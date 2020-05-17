import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from ourtools import *
from my_model.Basic_CNN import Basic_CNN
from my_model.SE_ResNet import *
import os

# 查看训练曲线
# 控制台执行
"""
tensorboard --logdir log
"""
base_path = os.getcwd()
batch_size = 45
input_size = 128  # 图片大小
NUM_EPOCHS = 500
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# optimizer

loss = F.cross_entropy
model = ResNet(SEResidualBlock, [2, 2, 2, 2]).to(device=DEVICE) # Basic_CNN().to(device=DEVICE)
# state_dict = torch.load('latest-ai.pth')
# model.load_state_dict(state_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

"""
transform = transforms.Compose(
    [
        transforms.Resize(input_size),
        transforms.CenterCrop((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
"""
transform = transforms.Compose(
    [
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


dataset = ImageFolder("Skin40\\", transform=transform)
print(dataset)
print(dataset.class_to_idx)
train_db, val_db = random_split(dataset, [1920, 480])
train_dataloader = DataLoader(train_db, shuffle=True, batch_size=batch_size)
valid_dataloader = DataLoader(val_db, shuffle=False, batch_size=batch_size)


def evaluate(model_eval, loader_eval, criterion_eval):
    """
    TODO: Implement the evaluate loop.
    """
    """YOUR CODE HERE"""
    model_eval.eval()
    loss_eval = 0
    correct = 0.0
    pbar = tqdm(total=len(loader_eval), desc="Evaluation", ncols=100)
    with torch.no_grad():
        for data, target in loader_eval:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss_eval += criterion_eval(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            pbar.update(1)
    pbar.close()

    loss_eval = loss_eval / loader_eval.dataset.__len__()
    accuracy = correct / loader_eval.dataset.__len__()
    response = {"loss": loss_eval, "acc": accuracy}
    return response


def train(model, loss_func, optimizer, device):


    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    for epoch_idx in range(NUM_EPOCHS):

        # TODO: Implement the training loop

        # YOUR CODE HERE
        pbar = tqdm(
            total=len(train_dataloader),
            desc="Train - Epoch {}".format(epoch_idx),
            ncols=100,
        )
        for batch_idx, (data, target) in enumerate(train_dataloader):

            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
        pbar.close()

        # END OF YOUR CODE
        train_resp = evaluate(model, train_dataloader, loss_func)
        eval_resp = evaluate(model, valid_dataloader, loss_func)

        print("-*-*-*-*-*- Epoch {} -*-*-*-*-*-".format(epoch_idx))
        print("Train Loss: {:.6f}\t".format(train_resp["loss"]))
        print("Train Acc: {:.6f}\t".format(train_resp["acc"]))
        print("Eval Loss: {:.6f}\t".format(eval_resp["loss"]))
        print("Eval Acc: {:.6f}\t".format(eval_resp["acc"]))
        print("\n")
        train_accs.append(train_resp["acc"])
        train_losses.append(train_resp["loss"])
        val_accs.append(eval_resp["acc"])
        val_losses.append(eval_resp["loss"])
        torch.save(model, "latest-ai.pth")

    show_curve(train_accs, "train acc")
    show_curve(train_losses, "train loss")
    show_curve(val_accs, "val_acc")
    show_curve(val_losses, "val_loss")


train(model, loss, optimizer, DEVICE)