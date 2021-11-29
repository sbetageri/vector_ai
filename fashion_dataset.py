'''
Contains dataset and dataloader
'''

import torch

from torchvision.datasets import ImageFolder

class ImageLoaders:
    '''
    Class to obtain DataLoaders for train/val and test
    '''
    def __init__(self,
                 data_path,
                 val_perc=0.8,
                 is_train=True,
                 transform=None):
        self.dataset = ImageFolder(data_path, transform=transform)
        if is_train:
            self.val_perc = val_perc
            self.is_train = is_train

    def _split_train_val(self):
        '''
        Split the training data into train and val sets
        :return:
        '''
        len_ds = len(self.dataset)
        val_len = int(self.val_perc * len_ds)
        train_len = len_ds - val_len
        train, val = torch.utils.data.random_split(self.dataset, (train_len, val_len))
        return train, val

    def get_train_val_loader(self, batch_size=512, shuffle=True):
        if self.is_train:
            train_ds, val_ds = self._split_train_val()
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle)
            return train_loader, val_loader
        else:
            return None

    def get_test_loader(self, batch_size=32, shuffle=True):
        return torch.utils.data.DataLoader(self.dataset , batch_size=batch_size, shuffle=shuffle)