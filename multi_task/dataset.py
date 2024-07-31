import numpy as np
import torch
import zarr

def read_dataset_split(dataset_path, 
                       pred_horizon,
                       obs_horizon,
                       action_horizon,
                       normalization=False):
    # Split the dataset into training and testing

    # create dataset from file
    dataset = RoboPianistDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        normalization=normalization
    )
    train_set, test_set = torch.utils.data.random_split(dataset, 
                                                        [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)], 
                                                        torch.Generator().manual_seed(42))
    # save training data statistics (min, max) for each dim
    stats = dataset.stats
    # create dataloader
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=256,
        num_workers=32,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=256,
        num_workers=32,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True
    )
    print("Train dataset size:", len(train_set))
    print("Test dataset size:", len(test_set))
    return train_loader, test_loader, stats


def read_dataset(pred_horizon, 
                 obs_horizon, 
                 action_horizon,
                 dataset_path,
                 normalization=False):
    # Read the dataset without splitting
    
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = RoboPianistDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        normalization=normalization
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats
    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=32,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True
    )
    print("Dataset size:", len(dataset))
    return dataloader, stats


def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        if i == 0:
            print(min_start, max_start, episode_length)
        # range stops one idx before end
        for idx in range(int(min_start), int(max_start+1)):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx,
                    obs_buffer_start_idx=0, obs_buffer_end_idx=0,
                    obs_sample_start_idx=0, obs_sample_end_idx=0):
    result = dict()
    if type(buffer_start_idx) != int:
        buffer_start_idx = int(buffer_start_idx)
    if type(buffer_end_idx) != int:
        buffer_end_idx = int(buffer_end_idx)
    if type(sample_start_idx) != int:
        sample_start_idx = int(sample_start_idx)
    if type(sample_end_idx) != int:
        sample_end_idx = int(sample_end_idx)
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        # dataset[0]: sample_start_idx: 1 sample_end_idx: 16, need to pad obs[0]
        # dataset[1]: sample_start_idx: 0 sample_end_idx: 16, don't need to pad
        # dataset[2]: sample_start_idx: 0 sample_end_idx: 16, don't need to pad
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min'] + 1e-8) + stats['min']
    return data

# dataset
class RoboPianistDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path,
                 pred_horizon, obs_horizon, action_horizon, normalization = False):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            'action': dataset_root['data']['action'][:],
            # (N, obs_dim)
            'obs': dataset_root['data']['state'][:]
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            if normalization:
                normalized_train_data[key] = normalize_data(data, stats[key])
            else:
                normalized_train_data[key] = data

        self.indices = indices
        self.stats = stats # None, if normalization = False
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]
        # print(buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon,:]
        return nsample
