import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision
from PIL import Image

train_data = np.load('1_1_train.npz', allow_pickle=True)
x_train, y_train = train_data['x'], train_data['y']
x_train = x_train.astype(np.float32)

test_data = ...
x_test, y_test = ...

...
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.feature_vectors = x
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature_vector = self.feature_vectors[idx]
        label = self.labels[idx]

        ...

        return feature_vector, label

train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class Model(nn.Module):
    def __init__(self,):
        super().__init__()
        ...

    def forward(self, x):
        ...

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(...).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=...)
loss_fn = nn.CrossEntropyLoss()
epochs = ...

def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(-1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total

def test(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item()
            correct += (out.argmax(-1) == y).sum().item()
            total += y.size(0)
        return total_loss / len(loader), correct / total


for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, loss_fn, device)
    ...

torch.save(model, 'model.pt')

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(-1) == y).sum().item()
            total += y.size(0)
        return correct / total


model = torch.load('model.pt', weights_only=False)
test_acc = evaluate(model, test_loader, device)
print(f'{test_acc:.4f}')

