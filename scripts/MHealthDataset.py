import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Ok the first issue I ran into is the data set problem
# getitem will return what the data is
# However, how should this work
# I can for example do it such that the dataset loads each row a separate thing
# Or I can load all the files and make one big file
# then I can label them somehow
#  

class MHealthDataset(Dataset):
    def __init__(self, data_dir, labels_dir ,transform=None, target_transform=None):
        self.data_dir = Path(data_dir)
        self.labels_dir = Path(labels_dir)
        self.data = None
        self.labels = None
        self.load_data()
        self.n_samples = self.data.shape[0]
        self.transform = transform
        self.target_transform = target_transform
        
    def load_data(self):
        self.data = pd.read_csv(self.data_dir,delimiter=",", header=None, dtype=np.float32).values
        self.labels = pd.read_csv(self.labels_dir, delimiter=",", header=None).values
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.Tensor(self.data[idx]), int(self.labels[idx][0]))


