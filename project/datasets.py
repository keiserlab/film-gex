# PyTorch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

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


class TabularDataset(torch.utils.data.Dataset):
    """
    Do not enable mutliprocessing.
    discussion: https://github.com/pytorch/pytorch/issues/21645
    source: https://github.com/rapidsai/deeplearning/blob/main/pytorch/batch_dataloader/batch_dataset.py#L27-L60
    """
    def __init__(self, tensors, batch_size=1, pin_memory=False):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors      
        self.batch_size=batch_size
        
        self.num_samples = tensors[0].size(0)
        
        if pin_memory:
            for tensor in self.tensors:
                tensor.pin_memory()  
    
    def __len__(self):
        if self.num_samples%self.batch_size == 0:
            return self.num_samples // self.batch_size
        else:
            return self.num_samples // self.batch_size + 1

    def __getitem__(self, item):
        idx = item*self.batch_size
        #Need to handle odd sized batches if data isn't divisible by batchsize
        if idx < self.num_samples and (idx + self.batch_size < self.num_samples or self.num_samples%self.batch_size == 0):
            return [tensor[idx:idx+self.batch_size] for tensor in self.tensors]
        elif idx < self.num_samples and idx + self.batch_size> self.num_samples :
            return [tensor[idx:] for tensor in self.tensors]
        else:
            raise IndexError

    
class CTRPDataModule(pl.LightningDataModule):
    def __init__(self, train, val, input_cols, cond_cols, target, batch_size=32):
        super().__init__()
        self.train = train
        self.val = val
        self.input_cols = input_cols
        self.cond_cols = cond_cols
        self.target = target
        self.batch_size = batch_size
    
    def tensorize(self, data):
        # Tensorize
        inputs = torch.FloatTensor(data[self.input_cols].to_numpy(dtype=np.float32))
        conds = torch.FloatTensor(data[self.cond_cols].to_numpy(dtype=np.float32))
        target = torch.FloatTensor(data[self.target].to_numpy(dtype=np.float32))
        return inputs, conds, target.view(-1,1)

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # Lower limit
        self.train[self.target] = np.clip(self.train[self.target], a_min=0., a_max=None)
        self.val[self.target] = np.clip(self.val[self.target], a_min=0., a_max=None)
        
        if stage == 'fit':
            self.train_dataset = TabularDataset(self.tensorize(self.train), batch_size=self.batch_size, pin_memory=True)
            self.val_dataset = TabularDataset(self.tensorize(self.val), batch_size=self.batch_size, pin_memory=True)
            return self.train_dataset, self.val_dataset
        
        if stage == 'test':
            self.test_dataset = TabularDataset(self.tensorize(self.test))
            return self.test_dataset

    # return the dataloader for each split
    # automatic batching must be disabled for chunked TabularDataset loading
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=None, num_workers=0, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=None, num_workers=0, pin_memory=True)