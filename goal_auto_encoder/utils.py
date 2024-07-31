from robopianist.suite.tasks import piano_with_shadow_hands_res
from robopianist.models.piano import Piano
from robopianist import music
import robopianist.models.piano.piano_constants as constants
import numpy as np
import pickle
import torch
import math

piano = Piano()
keys = piano.keys
FIRST_KEY_X = keys[0].pos[0]
FIRST_KEY_Y = keys[0].pos[1]
LAST_KEY_X = keys[-1].pos[0]
LAST_KEY_Y = keys[-1].pos[1]
keys_pos = np.array([key.pos for key in keys])

def get_keys_sdf(q_idx: torch.Tensor,
                  midi: torch.Tensor):
    # Transfer input parameters to np.array
    midi = midi.squeeze(1)
    if type(midi) == torch.Tensor:
        midi = midi.cpu().detach().numpy()
    if type(q_idx) == torch.Tensor:
        q_idx = q_idx.cpu().detach().numpy()
    total_dis = math.sqrt(constants.WHITE_KEY_LENGTH ** 2 + (keys_pos[-1][1] - keys_pos[0][1]) ** 2)
    ref_idx = np.where(midi == 1)[0]
    if len(ref_idx) == 0:
        ref_idx = np.array([0])
    min_dis = 100000
    for idx in ref_idx:
        # min_dis = min(min_dis, abs(keys[idx].pos[1] - keys[q_idx].pos[1]))
        dis = math.sqrt((keys_pos[idx][0] - keys_pos[q_idx][0]) ** 2 + (keys_pos[idx][1] - keys_pos[q_idx][1]) ** 2)
        if dis < min_dis:
            min_dis = dis
            # print('min_idx', idx)
    return min_dis / total_dis

def get_point_sdf(q: torch.Tensor,
                  midi: torch.Tensor):
    # Transfer input parameters to np.array
    midi = midi.squeeze(1)
    if type(midi) == torch.Tensor:
        midi = midi.cpu().detach().numpy()
    if type(q) == torch.Tensor:
        q = q.cpu().detach().numpy()
    total_dis = math.sqrt(constants.WHITE_KEY_LENGTH ** 2 + (keys[-1].pos[1] - keys[0].pos[1]) ** 2)
    ref_idx = np.where(midi == 1)[0]
    if len(ref_idx) == 0:
        ref_idx = np.array([0])
    min_dis = 100000
    for idx in ref_idx:
        # min_dis = min(min_dis, abs(keys[idx].pos[1] - keys[q_idx].pos[1]))
        dis = math.sqrt((keys[idx].pos[0] - q[0]) ** 2 + (keys[idx].pos[1] - q[1]) ** 2)
        if dis < min_dis:
            min_dis = dis
            # print('min_idx', idx)
    return min_dis / total_dis

if __name__ == '__main__':
    q_idx = torch.randint(0, 88, (1, ))
    midi = torch.zeros((88, 1))
    midi[23] = 1
    midi[66] = 1
    print(q_idx)
    print(get_keys_sdf(q_idx[0], midi))
