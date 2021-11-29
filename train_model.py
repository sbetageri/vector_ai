import os
import argparse

import fashion_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms, models

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score

class FashionModel(nn.Module):
    def __init__(self):
        super(FashionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.resnet(x)
        x = F.softmax(x, dim=1)
        return x

def train_val_loop(model, criterion, data_loader, device, optimizer=None):
    if optimizer is not None:
        model.train()
        loop_type = 'Training'
    else:
        model.eval()
        loop_type = 'Eval'

    running_labels = []
    running_preds = []
    running_loss = []
    for imgs, labels in tqdm(data_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        out = model(imgs)
        loss = criterion(out, labels)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss.append(loss.detach().cpu())
        running_labels.extend(labels.detach().cpu().numpy())

        pred_labels = torch.topk(out, k=1)[1].view(-1).detach().cpu().numpy()
        running_preds.extend(pred_labels)

    acc = accuracy_score(running_labels, running_preds)
    acc *= 100

    pred_loss = sum(running_loss) / len(running_loss)
    print(f'{loop_type} ::  {acc:.3f} % :: {pred_loss:.5f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Training Dir')
    parser.add_argument('-v', '--val_perc', help='Validation percentage', type=float)
    parser.add_argument('-e', '--epoch', help='Training Epochs', type=int)
    parser.add_argument('--out', help='Output Model Files')

    args = parser.parse_args()

    training_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = fashion_dataset.ImageLoaders(args.train, val_perc=args.val_perc, transform=training_transforms)
    train_loader, val_loader = dataset.get_train_val_loader()
    #test_loader = dataset.get_test_loader()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FashionModel()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for e in range(args.epoch):
        print('#'*27)
        print(f'Epoch : {e}')
        train_val_loop(model, criterion, train_loader, device, optimizer)
        train_val_loop(model, criterion, val_loader, device)
        print('#'*27)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(args.out, 'model.pt')
    torch.save(model, save_path)

