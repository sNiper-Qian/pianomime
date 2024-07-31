import numpy as np
import collections
from absl import logging
from dm_control.mujoco.wrapper import mjbindings
import numpy as np
import qpsolvers

mjlib = mjbindings.mjlib

_INVALID_JOINT_NAMES_TYPE = (
    '`joint_names` must be either None, a list, a tuple, or a numpy array; '
    'got {}.')
_REQUIRE_TARGET_POS_OR_QUAT = (
    'At least one of `target_pos` or `target_quat` must be specified.')

class IK_qpsolver:
    def __init__(self,
                 physics, 
                 site_names,
                 target_pos: np.ndarray,
                 pos_weights: np.ndarray,
                 gain: float=1.0,
                 limit_gain: float=0.05,
                 dt: float=0.05,
                 joint_names: list=None,
                 target_quat: np.ndarray=None,
                 quat_weights: np.ndarray=None,
                 lm_damping: float=1e-6,
                 damping: float=1e-6
                 ) -> None:
        self.physics = physics
        self.sites = site_names
        self.target_pos = target_pos
        self.pos_weights = pos_weights
        self.gain = gain
        self.limit_gain = limit_gain
        self.dt = dt
        self.joint_names = joint_names
        self.target_quat = target_quat
        self.quat_weights = quat_weights
        self.lm_damping = lm_damping
        self.n_target = len(site_names)
        self.damping = damping
        # Ensure that the Cartesian position of the site is up to date.
        mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

        # Convert site name to index.
        self.site_id = [physics.model.name2id(site_name, 'site') for site_name in site_names]
        # These are views onto the underlying MuJoCo buffers. mj_fwdPosition will
        # update them in place, so we can avoid indexing overhead in the main loop.
        self.site_xpos = [physics.named.data.site_xpos[site_name] for site_name in site_names]
        self.site_xmat = [physics.named.data.site_xmat[site_name] for site_name in site_names]

        # This is an index into the rows of `update` and the columns of `jac`
        # that selects DOFs associated with joints that we are allowed to manipulate.
        if joint_names is None:
            self.dof_indices = slice(None)  # Update all DOFs.
        elif isinstance(joint_names, (list, np.ndarray, tuple)):
            if isinstance(joint_names, tuple):
                joint_names = list(joint_names)
            # Find the indices of the DOFs belonging to each named joint. Note that
            # these are not necessarily the same as the joint IDs, since a single joint
            # may have >1 DOF (e.g. ball joints).
            indexer = physics.named.model.dof_jntid.axes.row
            # `dof_jntid` is an `(nv,)` array indexed by joint name. We use its row
            # indexer to map each joint name to the indices of its corresponding DOFs.
            self.dof_indices = indexer.convert_key_item(joint_names)
        else:
            raise ValueError(_INVALID_JOINT_NAMES_TYPE.format(type(joint_names)))
    
    def is_positive_definite(self, matrix):
        if not np.allclose(matrix, matrix.T):
            print("Matrix is not symmetric")
            return False  # Matrix is not symmetric
        try:
            np.linalg.cholesky(matrix)
            return True  # Matrix is positive definite
        except np.linalg.LinAlgError:
            return False  # Matrix is not positive definite

    def build_objective(self):
        '''
        Build the objective function for the QP solver.
        dq' * H * dq + c' * dq
        '''
        dtype = self.physics.data.qpos.dtype
        n_joints = len(self.joint_names)
        H = np.eye(n_joints, dtype=dtype)*self.damping
        c = np.zeros(n_joints, dtype=dtype)
        if self.target_pos is not None and self.target_quat is not None:
            jac = np.empty((6, self.physics.model.nv), dtype=dtype)
            err = np.empty(6, dtype=dtype)
            jac_pos, jac_rot = jac[:3], jac[3:]
            err_pos, err_rot = err[:3], err[3:]
        else:
            jac = np.empty((3, self.physics.model.nv), dtype=dtype)
            err = np.empty(3, dtype=dtype)
            if self.target_pos is not None:
                jac_pos, jac_rot = jac, None
                err_pos, err_rot = err, None
            elif self.target_quat is not None:
                jac_pos, jac_rot = None, jac
                err_pos, err_rot = None, err
            else:
                raise ValueError(_REQUIRE_TARGET_POS_OR_QUAT)
        for i in range(self.n_target):
            if jac_rot is None:
                mjlib.mj_jacSite(
                    self.physics.model.ptr, self.physics.data.ptr, jac_pos, None, self.site_id[i])
            else:
                mjlib.mj_jacSite(
                    self.physics.model.ptr, self.physics.data.ptr, jac_pos, jac_rot, self.site_id[i])
            jac_joints = jac[:, self.dof_indices].copy()
            # TODO: lm damping
            err_pos = self.target_pos[i] - self.site_xpos[i]
            # TODO: quaternion error
            err_rot = None
            if err_rot is None:
                err = err_pos
            else:
                raise NotImplementedError
            if "wrist" in self.sites[i]:
                weights = np.array([self.pos_weights[i]*1e-12, self.pos_weights[i]*1e-3, 1e-12])
            else:
                weights = np.array([self.pos_weights[i], self.pos_weights[i], self.pos_weights[i]])
            weight_matrix = np.diag(weights)
            weighted_jac_joints = weight_matrix @ jac_joints
            weighted_err = weight_matrix @ err * self.gain
            mu = self.lm_damping * weighted_err @ weighted_err
            H += np.dot(weighted_jac_joints.T, weighted_jac_joints) + mu*np.eye(n_joints, dtype=dtype)
            c -= np.dot(weighted_err.T, weighted_jac_joints)
        return H, c
    
    def build_inequalities(self):
        '''
        Build inequality constraints: joint limits.
        G * dq <= h
        '''
        n_joints = len(self.joint_names)
        G = np.vstack([np.diag([1.0]*n_joints), np.diag([-1.0]*n_joints)])
        p_maxs = []
        p_mins = []
        for dof_index in self.dof_indices:
            joint_range = self.physics.model.jnt_range[dof_index]
            jnt_min = joint_range[0]
            jnt_max = joint_range[1]
            jnt_value = self.physics.data.qpos[dof_index]
            p_max = self.limit_gain * (jnt_max - jnt_value) / self.dt if jnt_max != np.inf else 1e16
            p_min = self.limit_gain * (jnt_min - jnt_value) / self.dt if jnt_min != -np.inf else -1e16
            p_maxs.append(p_max)
            p_mins.append(p_min)
        p_maxs = np.array(p_maxs)
        p_mins = np.array(p_mins)
        h = np.hstack([p_maxs, -p_mins])
        return G, h
    
    def solve(self):
        H, c = self.build_objective()
        G, h = self.build_inequalities()
        # Solve the QP.
        problem = qpsolvers.Problem(H, c, G, h)
        result = qpsolvers.solve_problem(problem, solver='quadprog')
        dq = result.x
        assert dq is not None
        v = dq / self.dt
        return v

    def get_qpos(self):
        v = self.solve()
        v_full = np.zeros(self.physics.model.nv, dtype=self.physics.data.qpos.dtype)
        v_full[self.dof_indices] = v
        mjlib.mj_integratePos(self.physics.model.ptr, self.physics.data.qpos, v_full, self.dt)
        return self.physics.data.qpos.copy()



            
