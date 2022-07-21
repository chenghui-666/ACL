import argparse
import os

import cv2
from sklearn.metrics import roc_curve, auc, roc_auc_score

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnext50_32x4d, vgg16, densenet121, convnext_tiny
from tqdm import tqdm

from dataloaders import custom_transforms as trforms
from dataloaders import tn3k_dataset

import csv

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    # Model setting
    parser.add_argument('-backbone', type=str, default='resnet18')
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-loss_func', type=str, default='ce')  # ce, superloss, curriloss

    # Training setting
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-resume_epoch', type=int, default=0)
    parser.add_argument('-fold', type=int, default=0)

    parser.add_argument('-HQ_len', type=int, default=32)
    parser.add_argument('-coff_std', type=float, default=-1) # the coff_std of std, -1 means using ada_coff, 0 means no std
    parser.add_argument('-CQ_len', type=int, default=2) # how long is the certainty queue length: 2 times batchsize

    # Optimizer setting
    parser.add_argument('-save_every', type=int, default=2)
    parser.add_argument('-log_every', type=int, default=100)
    return parser.parse_args()


def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.backbone == 'resnet18':
        backbone = resnet18(pretrained=False)
        backbone.fc = nn.Linear(in_features=backbone.fc.in_features, out_features=2)
    elif args.backbone == 'dense':
        backbone = densenet121(pretrained=False)
        backbone.classifier = nn.Linear(1024, 2)
    elif args.backbone == 'convnext_tiny':
        backbone = convnext_tiny(pretrained=False)
        backbone.classifier = nn.Sequential(nn.Flatten(1), nn.Linear(768, 2))
    else:
        raise NotImplementedError

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    testset = tn3k_dataset.TN3KDataset(mode='test', transform=composed_transforms_ts, return_size=False)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    acc = []
    presicion = [] 
    recall = [] 
    F1 = [] 
    auc = []
    mean = []
    std = []

    for fold in range(5):
        if args.loss_func == 'acl': 
            save_dir = os.path.join('run', args.loss_func + '-' + args.backbone + '-mode_std' + str(args.coff_std) + '-HQ_len' + str(args.HQ_len) + '-CQ_len' + str(args.CQ_len*args.batch_size) + '-fold_' + str(fold) + '-'+ str(args.seed))
        else:
            save_dir = os.path.join('run', args.loss_func + '-' + args.backbone + '-fold' + str(fold) + '-'+str(args.seed))

        model_path = os.path.join(save_dir, 'models', 'best_backbone.pth')
        backbone.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        torch.cuda.set_device(device=0)
        backbone.cuda()
        backbone.eval()

        TP, FP, FN, TN = 0, 0, 0, 0

        labels = []
        preds = []
        acc_good = acc_bad = 0
        for sample_batched in tqdm(testloader):
            names = sample_batched['label_name']
            label = sample_batched['label']

            img = sample_batched['image'].cuda()
            label = label.cuda()
            feats = backbone.forward(img)
            probability = F.softmax(feats, dim=1)
            pred = torch.argmax(feats, dim=1, keepdim=False)

            labels.append(label.cpu().detach().numpy())
            preds.append(probability[0][1].cpu().detach().numpy())
            if pred == label == 0:
                acc_good += 1
            if pred == label == 1:
                acc_bad += 1        
            if pred == label == 1:
                TP += 1
            if pred == label == 0:
                TN += 1
            if pred == 1 and label == 0:
                FP += 1
            if pred == 0 and label == 1:
                FN += 1
        preds = np.array(preds)
        labels = np.array(labels)
        curauc = roc_auc_score(labels, preds)
        curpresicion = TP / (TP + FP)
        currecall = TP / (TP + FN)
        curF1 = 2 * curpresicion * currecall / (curpresicion + currecall)
        curacc = 50 * (acc_good / 378 + acc_bad/236)

        accstr = str(round(curacc, 3))
        presicionstr = str(round(curpresicion, 3))
        recallstr = str(round(currecall, 3))
        F1str = str(round(curF1, 3))
        aucstr = str(round(curauc, 3))

        acc.append(curacc)
        presicion.append(curpresicion) 
        recall.append(currecall) 
        F1.append(curF1) 
        auc.append(curauc) 

        print('acc:'+accstr+' precision:'+presicionstr+' recall:'+recallstr+' f1:'+F1str+' auc:'+aucstr)

    csvFile = open("results.csv", "a+")
    writer = csv.writer(csvFile)
    if args.loss_func != 'acl':
        name = args.backbone + "-" + args.loss_func
    else:
        name = args.backbone + "-" + args.loss_func + "-HQ" + str(args.HQ_len) + "-std" + str(args.coff_std)
    head = ["acc", "presicion", "recall", "F1", "AUC", name]
    writer.writerow(head)

    mean.append(round(np.mean(acc),4))
    mean.append(round(np.mean(presicion),4))
    mean.append(round(np.mean(recall),4))
    mean.append(round(np.mean(F1),4))
    mean.append(round(np.mean(auc),4))
    writer.writerow(mean)

    std.append(round(np.std(acc),4))
    std.append(round(np.std(presicion),4))
    std.append(round(np.std(recall),4))
    std.append(round(np.std(F1),4))
    std.append(round(np.std(auc),4))
    writer.writerow(std)

    csvFile.close()


if __name__ == '__main__':
    args = get_arguments()
    main(args)