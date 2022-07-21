import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi, sin, cos, hypot
import math
from scipy.special import lambertw

def load_pretrain_model(net, weights):
    net_keys = list(net.state_dict().keys())
    weights_keys = list(weights.keys())
    i = 0
    j = 0
    while i < len(net_keys) and j < len(weights_keys):
        name_i = net_keys[i]
        name_j = weights_keys[j]
        if net.state_dict()[name_i].shape == weights[name_j].shape:
            net.state_dict()[name_i].copy_(weights[name_j].cpu())
            i += 1
            j += 1
        else:
            i += 1
    print('success loading model.')
    return net


def CELoss(logit, target, reduction='mean'):
    criterion = nn.CrossEntropyLoss(reduction=reduction)
    return criterion(logit, target)

import json
calibration = json.load(open('datasets/calibration.json', 'r'))
calibration_benign = calibration['train_fna_b_tirad_m']
calibration_malignant = calibration['train_fna_m_tirad_b']


def RCELoss(logit, target, reduction='mean', th_mean = 0., th_std = 0., image_names=[], fold='0'):
    scores = torch.nn.Softmax(dim=1)(logit)
    criterion = nn.CrossEntropyLoss(reduction=reduction)
    total_loss = 0
    for i in range(logit.size()[0]):
        tag = target[i].item()
        if scores[i][tag].item() < th_mean + th_std:
            loss = 0
        else:
            loss = criterion(logit[i].unsqueeze(0), target[i].unsqueeze(0))
        total_loss += loss
    total_loss /= logit.size()[0]
    return total_loss

def HardHingeLoss(logit, groundTruth, device):    
    Nc = logit.data.size()
    y_onehot = torch.FloatTensor(len(groundTruth), Nc[1])
    y_onehot.zero_()
    y_onehot.scatter_(1, groundTruth.data.cpu().view(len(groundTruth),1), 1.0)    
    y = torch.autograd.Variable(y_onehot).to(device)
    t = logit*y
    L1 =torch.sum(t, dim=1)
    M, idx = logit.topk(2, 1, True, True) #same as torch.gather(pred, 1, index=labels)
    f1 = torch.eq(idx[:,0],groundTruth).float().to(device)
    u=  M[:,0]*(1-f1) + M[:,1]*f1
    L = torch.clamp(1.0-L1+u, min=0)
    return L

def logsumexp(inputs, dim=None, keepdim=False):
    return (inputs - F.log_softmax(inputs,dim)).mean(dim, keepdim=keepdim)


def SoftHingeLoss(logit, groundTruth, device):
    Nc = logit.data.size()
    y_onehot = torch.FloatTensor(len(groundTruth), Nc[1])
       
    y_onehot.zero_()
    y_onehot.scatter_(1, groundTruth.data.cpu().view(len(groundTruth),1), 1.0)
   
    y = torch.autograd.Variable(y_onehot).to(device)
    t = logit*y
    L1 =torch.sum(t, dim=1)
    M,idx = logit.topk(2, 1, True, True)
    f1 = torch.eq(idx[:,0],groundTruth).float().to(device)
    u = logsumexp(logit.to(device),dim=1)*(1-f1) + M[:,1]*f1
    L = torch.clamp(1.0-L1+u, min=0)
    return L

class CLoss(nn.Module):
###    
# Lrate:  true/false  at the initiliztion phase (first a few epochs) set false to train with an upperbound ;
#                     at the working phase , set true to traing with NPCL.
# Nratio:  noise ratio , set to zero for the clean case(it becomes CL when setting to zero)

###
    def __init__(self, Lrate=True, Nratio=0.0):
        super(CLoss, self).__init__()
        self.Lrate = Lrate
        self.Nratio = Nratio
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def forward(self, logit, labels):
        loss_1 = HardHingeLoss(logit, labels, self.device)
        ind_1_sorted = np.argsort(loss_1.data.cpu()).to(self.device)
        loss_1_sorted = loss_1[ind_1_sorted]

        epsilon = self.Nratio

        if self.Lrate:
            Ls = torch.cumsum(loss_1_sorted,dim=0)
            B =  torch.arange(start= 0 ,end=-len(loss_1_sorted),step=-1)
            B = torch.autograd.Variable(B).to(self.device)
            _, pred1 = torch.max(logit.data, 1)
            E = (pred1 != labels.data).sum()
            C = (1-epsilon)**2 *  float(len(loss_1_sorted)) + (1-epsilon) *  E
            B = C + B
            mask = (Ls <= B.float()).int()
            num_selected = int(sum(mask))
            Upbound = float(Ls.data[num_selected-1] <= ( C - num_selected))
            num_selected = int( min(  round(num_selected + Upbound), len(loss_1_sorted) ))

            ind_1_update = ind_1_sorted[:num_selected]

            loss_1_update = SoftHingeLoss(logit[ind_1_update], labels[ind_1_update], self.device)

        else:
            loss_1_update = SoftHingeLoss(logit, labels, self.device)

        return torch.mean(loss_1_update)

def curriloss(pred, label):
    criterion = CLoss()
    return criterion(pred, label)

class SuperLoss(nn.Module):

    def __init__(self, C=10, lam=1, batch_size=16):
        super(SuperLoss, self).__init__()
        self.tau = math.log(C)
        self.lam = lam  # set to 1 for CIFAR10 and 0.25 for CIFAR100
        self.batch_size = batch_size
                  
    def forward(self, logits, targets):
        l_i = F.cross_entropy(logits, targets, reduction='none').detach()
        sigma = self.sigma(l_i)
        loss = (F.cross_entropy(logits, targets, reduction='none') - self.tau)*sigma + self.lam*(torch.log(sigma)**2)
        loss = loss.sum()/self.batch_size
        return loss

    def sigma(self, l_i):
        x = torch.ones(l_i.size())*(-2/math.exp(1.))
        x = x.cuda()
        y = 0.5*torch.max(x, (l_i-self.tau)/self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma

def superloss(pred, label):
    criterion = SuperLoss()
    return criterion(pred, label)

def cal_acc(y_pred, y):
    y_pred = torch.argmax(y_pred, dim=1, keepdim=False)
    return torch.sum(y_pred == y).float() / y.shape[0]

# schedulers
def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)