import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from ourtools import *
from my_model.Basic_CNN import Basic_CNN
from my_model.ADV_ResNet import *
from torch.autograd import Variable
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

"""
tensorboard --logdir log
"""

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
    
    
base_path = os.getcwd()
batch_size = 120
input_size = 224  # 图片大小
NUM_EPOCHS = 20
LEARNING_RATE = 3.5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = LabelSmoothLoss(0.15)

def init_model():
    model = resnet50(pretrained=True) # Basic_CNN().to(device=DEVICE)
    model.fc=nn.Linear(2048,40)
    model = model.cuda()
    model = nn.DataParallel(model,device_ids=[0,1,2,3])
    #optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=0.8,weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    return model, optimizer

print ("Loading pretrained data")

transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])


dataset = ImageFolder("New_Skin40", transform=transform)
print(dataset)
print(dataset.class_to_idx)
train_db, val_db = random_split(dataset, [2225, 300])
bctrain_dataloader = DataLoader(train_db, shuffle=True, batch_size=batch_size)
bcvalid_dataloader = DataLoader(val_db, shuffle=False, batch_size=batch_size)

class trainer():
    def __init__(self):
        self.loss_func = ""
        self.device = ""
        self.model = ""
        self.optimizer = ""
        
    def print_confusion_matrix(self, y_pred,y_true):
        sns.set()
        f,ax = plt.subplots(figsize=(16,12))
        C2 = confusion_matrix(y_true, y_pred)
        
        recall_list=[]
        for i in range(len(C2[0])):
            aller = sum(C2[i])+1e-6
            recall_list.append(C2[i][i]/aller)
        
        sns.heatmap(C2,annot=True,ax=ax,cmap="Blues") #画热力图
        ax.set_title('confusion matrix') #标题
        ax.set_xlabel('predict') #x轴
        ax.set_ylabel('true') #y轴
        plt.show()
        return sum(recall_list)/len(recall_list)

    def evaluate(self, model_eval, loader_eval, criterion_eval):

        model_eval.eval()
        loss_eval = 0
        correct = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data, target in loader_eval:
                data, target = Variable(data.cuda()),Variable(target.cuda())
                output = self.model(data)
                loss_eval += criterion_eval(output, target).item()

                pred = output.argmax(dim=1, keepdim=True)

                for itm in pred:
                    y_pred.append(int(itm))
                for real in target:
                    y_true.append(int(real))

                correct += pred.eq(target.view_as(pred)).sum().item()

        baCC = self.print_confusion_matrix(y_pred,y_true)
        loss_eval = loss_eval / loader_eval.dataset.__len__()
        accuracy = correct / loader_eval.dataset.__len__()
        response = {"loss": loss_eval, "acc": accuracy,"bACC":baCC}
        return response


    def train(self, loss_func, device):

        self.loss_func = loss_func
        self.device = device
        
        accALL = 0
        bACCALL = 0
        
        for i in range(1):
            self.model, self.optimizer = init_model()
            record = 0

            train_dataloader = bctrain_dataloader #DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
            valid_dataloader = bcvalid_dataloader #DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)

            train_accs = []
            train_losses = []
            val_accs = []
            val_losses = []
            
            last_idx = NUM_EPOCHS-1;
            
            for epoch_idx in range(NUM_EPOCHS):
                for batch_idx, (data, target) in enumerate(train_dataloader):

                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = self.model(data)
                    loss = loss_func(output, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                print("Train map")
                train_resp = self.evaluate(self.model, train_dataloader, loss_func)
                print("Valid map")
                eval_resp = self.evaluate(self.model, valid_dataloader, loss_func)

                print("-*-*-*-*-*- Epoch {} -*-*-*-*-*-".format(epoch_idx))
                print("-*-*-*-*-*- fold {} -*-*-*-*-*-".format(i))
                print("Train Loss: {:.6f}\t".format(train_resp["loss"]))
                print("Train Acc: {:.6f}\t".format(train_resp["acc"]))
                print("Train bACC: {:.6f}\t".format(train_resp["bACC"]))
                print("Eval Loss: {:.6f}\t".format(eval_resp["loss"]))
                print("Eval Acc: {:.6f}\t".format(eval_resp["acc"]))
                print("Eval bACC: {:.6f}\t".format(eval_resp["bACC"]))
                print("\n")
                train_accs.append(train_resp["acc"])
                train_losses.append(train_resp["loss"])
                val_accs.append(eval_resp["acc"])
                val_losses.append(eval_resp["loss"])
                if epoch_idx == last_idx:
                    accALL += eval_resp["acc"]
                    bACCALL += eval_resp["bACC"]
                
                if epoch_idx > 15:
                    if eval_resp["acc"]>record:
                        record = eval_resp["acc"]
                        torch.save(self.model, "pre-trained-50.pth")

            # torch.save(self.model, "pre-trained-ai_"+str(i)+".pth")
            show_curve(train_accs, "train acc")
            show_curve(train_losses, "train loss")
            show_curve(val_accs, "val_acc")
            show_curve(val_losses, "val_loss")
        

train_player = trainer()
train_player.train(loss,DEVICE)
