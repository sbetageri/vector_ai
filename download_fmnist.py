'''
Download and transform fashion mnist to common ImageFolder format
'''

import os
import torch
import argparse
import torchvision

from tqdm import tqdm
from torch.utils.data import Dataset


class FashionDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.dataset = torchvision.datasets.FashionMNIST(
            root=self.root_dir,
            train=is_train,
            download=True,
        )
        self.labels = [
            't_shirt_top',
            'trouser',
            'pullover',
            'dress',
            'coat',
            'sandal',
            'shirt',
            'sneaker',
            'bag',
            'ankle_boot'
        ]

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label


def gen_dirs(root_dir, dir_list):
    for dir_name in dir_list:
        dir_path = os.path.join(root_dir, dir_name)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)


def transform_to_imagefolder(root_path, dataset):
    ds_labels = dataset.get_labels()
    gen_dirs(root_path, ds_labels)
    for idx, (img, label) in tqdm(enumerate(dataset)):
        label_name = ds_labels[label]
        img_path = os.path.join(root_path, label_name)
        img_path = os.path.join(img_path, f'img_{idx}.jpg')
        img.save(img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', help='Root Directory to store dataset', type=str)

    args = parser.parse_args()

    if args.root is None:
        print('Invalid usage. Please pass root directory to store dataset')
    else:
        abs_root = os.path.abspath(args.root)
        if not os.path.isdir(abs_root):
            os.mkdir(abs_root)
        is_train_set = [True, False]
        for is_train in is_train_set:
            f_dataset = FashionDataset(abs_root, is_train)
            if is_train:
                dataset_path = os.path.join(abs_root, 'train')
            else:
                dataset_path = os.path.join(abs_root, 'test')

            if not os.path.isdir(dataset_path):
                os.mkdir(dataset_path)

            transform_to_imagefolder(dataset_path, f_dataset)