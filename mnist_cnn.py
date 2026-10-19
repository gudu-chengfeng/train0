# -*- coding: utf-8 -*-
"""MNIST 手写数字识别：小型 CNN，PyTorch 实现。
预期测试准确率：99% 左右（CPU 上几分钟即可训练完）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14x14 -> 7x7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),                      # 转为 [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值/标准差
    ])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512)

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 6):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch}: 测试准确率 = {acc:.4f}")

    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("模型已保存到 mnist_cnn.pt")


if __name__ == "__main__":
    main()
