from robopianist.utils import inverse_kinematics as ik
from robopianist.utils import qp_solver
import numpy as np
import dm_env
from mujoco_utils import mjcf_utils, physics_utils, spec_utils, types

# For one hand
FF_BASE_IDX = 93
MF_BASE_IDX = 97
RF_BASE_IDX = 101
LF_BASE_IDX = 105
THUMB_BASE_IDX = 110
OH_FINGER_BASE_IDX = {'ff': FF_BASE_IDX, 'mf': MF_BASE_IDX, 'rf': RF_BASE_IDX, 'lf': LF_BASE_IDX, 'th': THUMB_BASE_IDX}
# For two hands
RH_FF_BASE_IDX = 93
RH_MF_BASE_IDX = 97
RH_RF_BASE_IDX = 101
RH_LF_BASE_IDX = 105
RH_THUMB_BASE_IDX = 110
LH_FF_BASE_IDX = 120
LH_MF_BASE_IDX = 124
LH_RF_BASE_IDX = 128
LH_LF_BASE_IDX = 132
LH_THUMB_BASE_IDX = 137
RH_FINGER_BASE_IDX = {'ff': RH_FF_BASE_IDX, 'mf': RH_MF_BASE_IDX, 'rf': RH_RF_BASE_IDX, 'lf': RH_LF_BASE_IDX, 'th': RH_THUMB_BASE_IDX}
LH_FINGER_BASE_IDX = {'ff': LH_FF_BASE_IDX, 'mf': LH_MF_BASE_IDX, 'rf': LH_RF_BASE_IDX, 'lf': LH_LF_BASE_IDX, 'th': LH_THUMB_BASE_IDX}

FF_JOINTS = 4
MF_JOINTS = 4
RF_JOINTS = 4
LF_JOINTS = 5
THUMB_JOINTS = 5
FINGER_JOINTS = {'ff': FF_JOINTS, 'mf': MF_JOINTS, 'rf': RF_JOINTS, 'lf': LF_JOINTS, 'th': THUMB_JOINTS}


WHITE_KEY_INDICES = [
        0,
        2,
        3,
        5,
        7,
        8,
        10,
        12,
        14,
        15,
        17,
        19,
        20,
        22,
        24,
        26,
        27,
        29,
        31,
        32,
        34,
        36,
        38,
        39,
        41,
        43,
        44,
        46,
        48,
        50,
        51,
        53,
        55,
        56,
        58,
        60,
        62,
        63,
        65,
        67,
        68,
        70,
        72,
        74,
        75,
        77,
        79,
        80,
        82,
        84,
        86,
        87,
    ]

BLACK_TWIN_KEY_INDICES = [
        4,
        6,
        16,
        18,
        28,
        30,
        40,
        42,
        52,
        54,
        64,
        66,
        76,
        78,
    ]
BLACK_TRIPLET_KEY_INDICES = [
        1,
        9,
        11,
        13,
        21,
        23,
        25,
        33,
        35,
        37,
        45,
        47,
        49,
        57,
        59,
        61,
        69,
        71,
        73,
        81,
        83,
        85,
    ]

def move_fingers_to_pos_qp(env: dm_env.Environment, 
                            hand_action: np.ndarray,
                            finger_names: list=['lf', 'rf', 'mf', 'ff', 'th'],
                            hand_side: str='none',
                            targeting_wrist: bool=False,
                            ):
    """Move the specified fingers to the
      specified position."""
    if hand_side == 'left':
        FINGER_BASE_IDX = LH_FINGER_BASE_IDX
        prefix = "lh_shadow_hand/"
    elif hand_side == 'right':
        FINGER_BASE_IDX = RH_FINGER_BASE_IDX
        prefix = "rh_shadow_hand/"
    else:
        FINGER_BASE_IDX = OH_FINGER_BASE_IDX
        prefix = "lh_shadow_hand/"
    if targeting_wrist:
        site_names = [prefix+"wrist_site"]
    else:
        site_names = []
    site_names.extend([prefix+finger_name+"distal_site" for finger_name in finger_names])
    # site_names = [prefix+finger_name+"distal_site" for finger_name in finger_names]
    target_poses = []
    target_quats = []
    pos_weights = np.array([1e4]*len(site_names))
    # Reset the color of all keys
    # env.physics.bind(env.task.piano._key_geoms).rgba = (0.5, 0.5, 0.5, 1.0)
    for i in range(6):
        # if hand_action[7][i] != -1:
        #     env.physics.bind(env.task.piano._key_geoms[int(hand_action[7][i])]).rgba = (0.0, 1.0, 0.0, 1.0)
            # pos_weights[i-1] = 1.0
        if i == 0:
            if not targeting_wrist:
                # Don't set target position for the wrist
                continue
        target_pos = (hand_action[0][i], hand_action[1][i], hand_action[2][i])
        # target_quat = (hand_action[3][i], hand_action[4][i], hand_action[5][i], hand_action[6][i])
        target_poses.append(target_pos)
        # target_quats.append(target_quat)
    # for key in key_indices:
    #     if key not in hand_action[7]:
    #         env.physics.bind(env.task.piano._key_geoms[key]).rgba = (1.0, 0.0, 0.0, 1.0)
        # if key in hand_action[7]:
        #     env.physics.bind(env.task.piano._key_geoms[key]).rgba = (0.5, 0.5, 0.0, 1.0)
    # Wrist, forearm and the dedicated finger joints are available for IK
    if hand_side == 'left':
        wrist_joint_names = [prefix+"lh_WRJ2",
                               prefix+"lh_WRJ1"]
        forearm_joint_names = [prefix+"forearm_tx",
                                prefix+'forearm_ty',
                                prefix+'forearm_tz']
    elif hand_side == 'right':
        wrist_joint_names = [prefix+"rh_WRJ2",
                               prefix+"rh_WRJ1"]
        forearm_joint_names = [prefix+"forearm_tx",
                                prefix+'forearm_ty',
                                prefix+'forearm_tz']
    else:
        wrist_joint_names = ["lh_shadow_hand/"+"lh_WRJ2",
                               "lh_shadow_hand/"+"lh_WRJ1"]
        forearm_joint_names = ["lh_shadow_hand/"+"forearm_tx",
                                "lh_shadow_hand/"+'forearm_ty',
                                "lh_shadow_hand/"+'forearm_tz']
    finger_joint_names = []
    for finger_name in finger_names:
        finger_joint_names.extend([env.physics.model.id2name(FINGER_BASE_IDX[finger_name]+i, 'joint') for i in range(FINGER_JOINTS[finger_name])])
    joint_names = forearm_joint_names + finger_joint_names + wrist_joint_names
    # Calculate the IK result
    solver = qp_solver.IK_qpsolver(physics=env.physics,
                                    site_names=site_names,
                                    target_pos=target_poses,
                                    joint_names=joint_names,
                                    pos_weights=pos_weights,
                                    )
    qvel = solver.solve()
    return qvel, solver.dof_indices, target_poses

def move_fingers_to_pos(env: dm_env.Environment, 
                        hand_action: np.ndarray,
                        finger_names: list=['lf', 'rf', 'mf', 'ff', 'th'],
                        hand_side: str='none',
                        ):
    """Move the specified fingers to the
      specified position."""
    if hand_side == 'left':
        FINGER_BASE_IDX = LH_FINGER_BASE_IDX
        prefix = "lh_shadow_hand/"
    elif hand_side == 'right':
        FINGER_BASE_IDX = RH_FINGER_BASE_IDX
        prefix = "rh_shadow_hand/"
    else:
        FINGER_BASE_IDX = OH_FINGER_BASE_IDX
        prefix = "lh_shadow_hand/"
    site_names = [prefix+finger_name+"distal_site" for finger_name in finger_names]
    target_poses = []
    target_quats = []
    # Reset the color of all keys
    env.physics.bind(env.task.piano._key_geoms).rgba = (0.5, 0.5, 0.5, 1.0)
    for i in range(6):
        if hand_action[7][i] != -1:
            env.physics.bind(env.task.piano._key_geoms[int(hand_action[7][i])]).rgba = (0.0, 1.0, 0.0, 1.0)
        if i == 0:
            # Don't set target position for the wrist
            continue
        target_pos = (hand_action[0][i], hand_action[1][i], hand_action[2][i])
        target_quat = (hand_action[3][i], hand_action[4][i], hand_action[5][i], hand_action[6][i])
        target_poses.append(target_pos)
        target_quats.append(target_quat)
    # Change the color of the activated key
    activation = env.task.piano.activation
    # Find the index of True
    key_indices = np.where(activation == True)[0]
    for key in key_indices:
        if key not in hand_action[7]:
            env.physics.bind(env.task.piano._key_geoms[key]).rgba = (1.0, 0.0, 0.0, 1.0)
    # Wrist, forearm and the dedicated finger joints are available for IK
    if hand_side == 'left':
        wrist_joint_names = [prefix+"lh_WRJ2",
                               prefix+"lh_WRJ1"]
        forearm_joint_names = [prefix+"forearm_tx",
                                prefix+'forearm_ty',
                                prefix+'forearm_tz']
    elif hand_side == 'right':
        wrist_joint_names = [prefix+"rh_WRJ2",
                               prefix+"rh_WRJ1"]
        forearm_joint_names = [prefix+"forearm_tx",
                                prefix+'forearm_ty',
                                prefix+'forearm_tz']
    else:
        wrist_joint_names = ["lh_shadow_hand/"+"lh_WRJ2",
                               "lh_shadow_hand/"+"lh_WRJ1"]
        forearm_joint_names = ["lh_shadow_hand/"+"forearm_tx",
                                "lh_shadow_hand/"+'forearm_ty',
                                "lh_shadow_hand/"+'forearm_tz']
    finger_joint_names = []
    for finger_name in finger_names:
        finger_joint_names.extend([env.physics.model.id2name(FINGER_BASE_IDX[finger_name]+i, 'joint') for i in range(FINGER_JOINTS[finger_name])])
    joint_names = forearm_joint_names + finger_joint_names + wrist_joint_names
    # Calculate the IK result
    ik_result = ik.qpos_from_multiple_site_pos(physics=env.physics,
                                                site_names=site_names,
                                                target_pos=target_poses,
                                                # target_quat=target_quats,
                                                joint_names=joint_names,
                                                pos_weight=np.array([1.0]*len(site_names)),
                                                # rot_weight=0.5
                                                )
    return ik_result

def move_fingers_to_keys(env: dm_env.Environment, 
                         key_indices: list, 
                         offset_x: float=[0]*5,
                         offset_y: float=[0]*5,
                         finger_names: list=['lf', 'rf', 'mf', 'ff', 'th'],
                         ):
    """Move the specified fingers to the
      specified piano keys."""
    # print(env.physics.model.site_group)
    assert len(key_indices) == len(finger_names)
    site_names = ["lh_shadow_hand/"+finger_name+"distal_site" for finger_name in finger_names]
    target_poses = []
    for i, key_index in enumerate(key_indices):
        if key_index in WHITE_KEY_INDICES:
            prefix = "white_key_"
        elif key_index in BLACK_TWIN_KEY_INDICES or key_index in BLACK_TRIPLET_KEY_INDICES:
            prefix = "black_key_"
        else:
            raise ValueError(f"Invalid key index: {key_index}")
        target_key = mjcf_utils.safe_find(env.task.piano._mjcf_root, "body", f"{prefix}{key_index}")
        target_pos = target_key.pos
        target_pos = (target_pos[0]+offset_x[i], target_pos[1]+offset_y[i], target_pos[2])
        # target_key.add("site", type="sphere", pos=(0.1, 0.1, 0.1), rgba=(1, 0, 0, 1))
        key_geom = env.task.piano.keys[key_index].geom[0]
        # Uncomment method _update_key_color
        # env.physics.bind(key_geom).rgba = (1.0, 1.0, 1.0, 1.0)
        env.physics.bind(env.task.piano._key_geoms[key_index]).rgba = (0.0, 1.0, 0.0, 1.0)
        # print(key_geom.rgba)
        # print(mjcf_utils.safe_find_all(env.task.piano._mjcf_root, "site"))

        target_poses.append(target_pos)
    # Wrist, forearm and the dedicated finger joints are available for IK
    wrist_joint_names = ["lh_shadow_hand/"+"lh_WRJ2",
                            "lh_shadow_hand/"+"lh_WRJ1"]
    forearm_joint_names = ["lh_shadow_hand/"+env.task._hand.joints[-3].name,
                            "lh_shadow_hand/"+env.task._hand.joints[-2].name, 
                            "lh_shadow_hand/"+env.task._hand.joints[-1].name]
    finger_joint_names = []
    for finger_name in finger_names:
        finger_joint_names.extend([env.physics.model.id2name(FINGER_BASE_IDX[finger_name]+i, 'joint') for i in range(FINGER_JOINTS[finger_name])])
    joint_names = forearm_joint_names + finger_joint_names + wrist_joint_names
    # Calculate the IK result
    ik_result = ik.qpos_from_multiple_site_pos(physics=env.physics,
                                                site_names=site_names,
                                                target_pos=target_poses,
                                                joint_names=joint_names,
                                                pos_weight=np.array([1.0]*len(site_names)),
                                                )
    return ik_result

def move_finger_to_key(env: dm_env.Environment, key_index: int, finger_name):
    """Move the specified finger to the specified piano key."""
    if key_index in WHITE_KEY_INDICES:
        prefix = "white_key_"
    elif key_index in BLACK_TWIN_KEY_INDICES or key_index in BLACK_TRIPLET_KEY_INDICES:
        prefix = "black_key_"
    else:
        raise ValueError(f"Invalid key index: {key_index}")
    
    # Position of the piano joint as the target position
    target_pos = mjcf_utils.safe_find(env.task.piano._mjcf_root, "body", f"{prefix}{key_index}").pos

    # # For test
    # target_pos = (0.4, mjcf_utils.safe_find(env.task.piano._mjcf_root, "body", f"{prefix}{key_index}").pos[1], 0.13)
    # print(target_pos)
    # print(mjcf_utils.safe_find_all(env.task._hand._mjcf_root, "body"))
    # print(mjcf_utils.safe_find_all(env.task._hand._mjcf_root, "site"))
    # print(env.task._hand.root_body.pos) 
    # print(env.physics.named.data.qpos)
    # print(env.physics.model.id2name(89, 'joint'))

    # Wrist, forearm and the dedicated finger joints are available for IK
    wrist_joint_names = ["lh_shadow_hand/"+"lh_WRJ2",
                            "lh_shadow_hand/"+"lh_WRJ1"]
    forearm_joint_names = ["lh_shadow_hand/"+env.task._hand.joints[-2].name, 
                            "lh_shadow_hand/"+env.task._hand.joints[-1].name]
    finger_joint_names = [env.physics.model.id2name(FINGER_BASE_IDX[finger_name]+i, 'joint') for i in range(FINGER_JOINTS[finger_name])]
    joint_names = forearm_joint_names + finger_joint_names + wrist_joint_names
    
    # Calculate the IK result
    ik_result = ik.qpos_from_site_pose(physics=env.physics, 
                                       site_name="lh_shadow_hand/"+finger_name+"distal_site", 
                                       target_pos=target_pos,
                                       joint_names=joint_names,
                                    #    joint_names=("lh_shadow_hand/"+env.task._hand.joints[-2].name, 
                                    #                 "lh_shadow_hand/"+env.task._hand.joints[-1].name),
                                       )
    return ik_result
    
    
