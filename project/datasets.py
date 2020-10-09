# PyTorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Transforms
from sklearn.preprocessing import StandardScaler

# Testing
from datetime import datetime


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, input_cols, cond_cols, target):
        self.inputs = torch.FloatTensor(data[input_cols].to_numpy())
        self.conds = torch.FloatTensor(data[cond_cols].to_numpy())
        self.target = torch.FloatTensor(data[target].to_numpy())
    
    def __getitem__(self, i):
        return self.inputs[i], self.conds[i], self.target[i].view(-1)

    def __len__(self):
        return len(self.target)

    
class CTRPDataModule(pl.LightningDataModule):
    def __init__(self, train, val, input_cols, cond_cols, target, batch_size=32):
        super().__init__()
        self.train = train
        self.val = val
        self.input_cols = input_cols
        self.cond_cols = cond_cols
        self.target = target
        self.batch_size = batch_size

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        
        if stage == 'fit':
            start = datetime.now() 
            self.train_dataset = Dataset(self.train, self.input_cols, self.cond_cols, self.target)
            self.val_dataset = Dataset(self.val, self.input_cols, self.cond_cols, self.target)
            print('Completed dataset creation in {}'.format(str(datetime.now() - start)))
            return self.train_dataset, self.val_dataset
        if stage == 'test':
            self.test_dataset = Dataset(self.test, self.input_cols, self.cond_cols, self.target)
            return self.test_dataset

    # return the dataloader for each split
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=16, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16, pin_memory=True)