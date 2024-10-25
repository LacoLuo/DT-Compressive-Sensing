import os 
import sklearn
import numpy as np 
import pandas as pd
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset, DataLoader

def create_samples(data_root, csv_path, random_state, num_data_point, portion, select_data_idx):
    # Load channel data and beam indices
    channel = loadmat(os.path.join(data_root, "dataset.mat"))['all_channel']
    beam_idx = loadmat(os.path.join(data_root, "dataset.mat"))['all_beam_idx'] - 1 # Start from 0
    
    # Load data indices
    if select_data_idx is None:
        data_idx = pd.read_csv(os.path.join(data_root, csv_path))["data_idx"].to_numpy()
    else:
        data_idx = select_data_idx
    
    channel = channel[data_idx, ...]
    beam_idx = beam_idx[data_idx, ...]
    
    # Shuffle
    channel, beam_idx, data_idx = sklearn.utils.shuffle(channel, beam_idx, data_idx, random_state=random_state)
    
    if num_data_point:
        channel = channel[:num_data_point, ...]
        beam_idx = beam_idx[:num_data_point, ...]
        data_idx = data_idx[:num_data_point, ...]
    else:
        num_data = beam_idx.shape[0]
        p = int(num_data*portion)
        
        channel = channel[:p, ...]
        beam_idx = beam_idx[:p, ...]
        data_idx = data_idx[:p, ...]
        
    # Normalization
    channel /= np.linalg.norm(channel, ord='fro', axis=(-1, -2), keepdims=True)  
    
    return channel, beam_idx, data_idx

class DataFeed(Dataset):
    def __init__(self, data_root, csv_path, random_state=0, num_data_point=None, portion=1.0, select_data_idx=None):
        self.data_root = data_root
        self.channel, self.label, self.data_idx = create_samples(data_root, csv_path, random_state, num_data_point, portion, select_data_idx)
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        channel = self.channel[idx, ...]
        label = self.label[idx, 0]
        data_idx = self.data_idx[idx, ...]
        
        channel = torch.tensor(channel, requires_grad=False)
        label = torch.tensor(label, requires_grad=False)
        data_idx = torch.tensor(data_idx, requires_grad=False)
        
        return channel.cfloat(), label.long(), data_idx.long()

if __name__ == "__main__":
    data_root = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_real"
    train_csv = "train_data_idx.csv"
    val_csv = "test_data_idx.csv"
    batch_size = 4
    
    train_feeder = DataFeed(data_root, train_csv, portion=1.)
    train_loader = DataLoader(train_feeder, batch_size=batch_size)
    
    channel, beam_idx = next(iter(train_loader))
    print(beam_idx.size())