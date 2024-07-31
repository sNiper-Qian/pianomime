import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils
import torch.distributions
import numpy as np
import zarr

device = torch.device('cuda')

def read_dataset(dataset_path, train_split=0.8, batch_size=512):
    # create dataset from file
    dataset = MidiDataset(
        dataset_path=dataset_path,
    )
    train_set, test_set = torch.utils.data.random_split(dataset, 
                                                        [int(len(dataset)*train_split), 
                                                         len(dataset)-int(len(dataset)*train_split)], 
                                                        torch.Generator().manual_seed(42))
    # create dataloader
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=32,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=32,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True
    )
    print("Train dataset size:", len(train_set))
    print("Test dataset size:", len(test_set))
    return train_loader, test_loader

# dataset
class MidiDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        # All demonstration episodes are concatinated in the first dimension N
        self.train_data = {
            # (N, obs_dim)
            'obs': dataset_root['data']['state'][:]
        }

    def __len__(self):
        # all possible segments of the dataset
        return self.train_data['obs'].shape[0]

    def __getitem__(self, idx):
        # get a segment of the dataset
        obs = self.train_data['obs'][idx]
        obs = np.expand_dims(obs, axis=-1)
        return obs

class FingertipDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        # All demonstration episodes are concatinated in the first dimension N
        self.train_data = {
            # (N, obs_dim)
            'obs': dataset_root['data']['state'][:],
            'action': dataset_root['data']['action'][:]
        }
        
        
