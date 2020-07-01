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
import random
"""
tensorboard --logdir log
"""


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss

class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
 
    def __init__(self, num_classes=751, feat_dim=240, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
 
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
 
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"
 
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
 
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        print(mask)
 
        dist = []
        for i in range(batch_size):
            print(mask[i])
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss

    
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
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = LabelSmoothLoss(0.1)

def init_optimizer(model):
    #optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=0.9,weight_decay= 5e-4) # 5e-4
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    return optimizer

def init_model(multi_gpu = True):
    """
    model = resnet152(pretrained=True) # Basic_CNN().to(device=DEVICE)
    model.fc=nn.Linear(2048,40)
    model = model.cuda()
    if multi_gpu:
        model = nn.DataParallel(model,device_ids=[0,1,2])
    """
    model = torch.load("pre-trained-50.pth")
#     model = model.cuda()
#     if multi_gpu:
#         model = nn.DataParallel(model,device_ids=[0,1,2,3])
        
    optimizer = init_optimizer(model)
    return model, optimizer

def init_152model(multi_gpu = True):
    model = resnet152(pretrained=True) # Basic_CNN().to(device=DEVICE)
    model.fc=nn.Linear(2048,40)
    model = model.cuda()
    if multi_gpu:
        model = nn.DataParallel(model,device_ids=[0,1,2,3])
    
    optimizer = init_optimizer(model)
    return model, optimizer



print ("Loading pretrained data")
# model = resnet152(pretrained=True) # Basic_CNN().to(device=DEVICE)
# model.fc=nn.Linear(2048,40)
# model = model.cuda()
# model = nn.DataParallel(model,device_ids=[0,1,2])
# # state_dict = torch.load('latest-ai.pth')
# # model.load_state_dict(state_dict)
# optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=0.9,weight_decay=0.005)


transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])


dataset = ImageFolder("Skin40", transform=transform)
print(dataset)
print(dataset.class_to_idx)

k = 0
for i in range(40):
    subdataset = Subset(dataset,[60*i+j for j in range(60)])
    if k == 0:
        k += 1
        dataset1, dataset2, dataset3, dataset4, dataset5, dataset6 = random_split(subdataset, [10, 10, 10, 10, 10, 10])
    else:
        subset = random_split(subdataset, [10, 10, 10, 10, 10, 10])
        dataset1 += subset[0]; dataset2 += subset[1]; dataset3 += subset[2]; dataset4 += subset[3]; dataset5 += subset[4]; dataset6 += subset[5]
        
# partdataset = ImageFolder("New_Skin40", transform=transform)
testset = dataset6
class trainer():
    def __init__(self):
        self.loss_func = nn.CrossEntropyLoss()
        self.device = DEVICE
        self.submodelnum = 5
        self.model = ""
        self.optimizer = ""
        self.vote_rates = torch.zeros([self.submodelnum,40]).to(DEVICE)
        self.latest_rates = torch.zeros([40]).to(DEVICE)
    
    
    def calculate_vote_rate(self, cm):
        voter = torch.zeros([len(cm[0])]).to(DEVICE)

        for i in range(0,len(cm[0])):
            rec = sum(cm[i,:]) + 0.001
            dec = sum(cm[:,i]) + 0.001
            fin = min(cm[i,i]/rec,cm[i,i]/dec)
            oct = 2*cm[i,i]/(rec+dec)
            fin += oct
            voter[i]= 1+(fin-1)/20

        return voter
    
    def print_confusion_matrix(self, y_pred,y_true):
        sns.set()
        f,ax = plt.subplots(figsize=(16,12))
        C2 = confusion_matrix(y_true, y_pred)
        self.latest_rates = self.calculate_vote_rate(C2)
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
    
    def load_member(self, nid):
        model = torch.load("latest-gp_ai_"+str(nid)+".pth")
        model.eval()
        return model
    
    def test(self):
        
        print(self.vote_rates[0][0:10])
        print(self.vote_rates[0][10:20])
        print(self.vote_rates[0][20:30])
        print(self.vote_rates[0][30:40])
        
        model_list = []
        for i in range(0,self.submodelnum):
            model_list.append(self.load_member(i))
        
        loader_eval = DataLoader(testset, shuffle=False, batch_size=batch_size)
        loss_eval = 0.0
        correct = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data, target in loader_eval:
                data, target = Variable(data.cuda()),Variable(target.cuda())
                output_list = []
                for i in range(0,self.submodelnum):
                    output = model_list[i](data)*self.vote_rates[i]
                    output_list.append(output)
                
                for j in range(1,self.submodelnum):
                    output_list[0]+=output_list[j]
                    
                loss_eval += self.loss_func(output_list[0], target).item()

                pred = output_list[0].argmax(dim=1, keepdim=True)

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
        model_id = int(-1)
        
        
        
        for mc in range(0,1):
            for i in range(5):
                model_id += 1
                self.model, self.optimizer = init_model()
                train_dataset = 0
                valid_dataset = 0
                record = 0

                for j,dataset0 in zip(range(5),(dataset1, dataset2, dataset3, dataset4, dataset5)):
                    if j != i:
                        if train_dataset == 0:
                            train_dataset = dataset0
                        else:
                            train_dataset += dataset0
                    else:
                        valid_dataset = dataset0
                train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
                valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)

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
                    print("-*-*-*-*-*- fold {} -*-*-*-*-*-".format(model_id))
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

                    if epoch_idx > 5:
                        if eval_resp["acc"]>record:
                            record = eval_resp["acc"]
                            self.vote_rates[model_id] = self.latest_rates
                            torch.save(self.model, "latest-gp_ai_"+str(model_id)+".pth")
                    
                show_curve(train_accs, "train acc")
                show_curve(train_losses, "train loss")
                show_curve(val_accs, "val_acc")
                show_curve(val_losses, "val_loss")
    
        print("Final avg acc ")
        print(accALL/self.submodelnum)
        print("Final avg bACC")
        print(bACCALL/self.submodelnum)

train_player = trainer()
train_player.train(loss,DEVICE)
train_player.test()
