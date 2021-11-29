import os
import argparse

import fashion_dataset
from train_model import train_val_loop, FashionModel

import torch
import torch.nn as nn

from torchvision import transforms

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='Test Dir')
    parser.add_argument('--model', help='Model Location')

    args = parser.parse_args()

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = fashion_dataset.ImageLoaders(args.test, transform=transforms, is_train=False)
    test_loader = dataset.get_test_loader()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.load(args.model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    train_val_loop(model, criterion, test_loader, device)