import argparse
import os
import socket
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, densenet121, convnext_tiny

import utils
from dataloaders import custom_transforms as trforms
from dataloaders import tn3k_dataset


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    # Model setting
    parser.add_argument('-backbone', type=str, default='resnet18')
    parser.add_argument('-loss_func', type=str, default='acl')  # ce, superloss, curriloss, acl
    parser.add_argument('-input_size', type=int, default=224)

    # Training setting
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-nepochs', type=int, default=50)
    parser.add_argument('-resume_epoch', type=int, default=0)

    # Optimizer setting
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-update_lr_every', type=int, default=10)
    parser.add_argument('-log_every', type=int, default=100)
    parser.add_argument('-fold', type=int, default=0)

    # curriculum setting
    parser.add_argument('-HQ_len', type=int, default=32)
    parser.add_argument('-coff_std', type=float, default=-1) # the coff_std of std, -1 means using ada_coff, 0 means no std
    parser.add_argument('-CQ_len', type=int, default=2) # how long is the certainty queue length: 2 times batchsize
    return parser.parse_args()


def evaluate(backbone, dataloader, epoch, best_auc, logger, writer, save_dir):
    backbone.eval()
    acc = 0.0
    TP, FP, FN, TN = 0, 0, 0, 0
    preds = []
    labels = []
    for ii, sample_batched in enumerate(dataloader):
        img, label = sample_batched['image'].cuda(), sample_batched['label'].cuda()
        feats = backbone.forward(img)
        pred = torch.argmax(feats, dim=1, keepdim=False)
        prob = F.softmax(feats, dim=1)
        
        labels.append(label.cpu().detach().numpy())
        preds.append(prob[0][1].cpu().detach().numpy())

        if pred == label:
            acc += 1
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
    auc = roc_auc_score(labels, preds)

    acc /= len(labels)
    presicion = TP / (TP + FP + 0.000001)
    recall = TPR = TP / (TP + FN + 0.000001)
    f1 = 2 * presicion * recall / (presicion + recall + 0.000001)

    if epoch > 30:
        backbone_save_path = os.path.join(save_dir, 'models', 'backbone_e' + str(epoch) + '.pth')
        if auc > best_auc:
            best_auc = auc
            backbone_save_path = os.path.join(save_dir, 'models', 'best_backbone.pth')
            torch.save(backbone.state_dict(), backbone_save_path)

    print('Validation:')
    print('epoch: %d, images: %d, acc: %.4f, auc: %.4f, f1: %.4f, presicion: %.4f, recall: %.4f \n' % \
    (epoch, len(labels), acc, auc, f1, presicion, recall))
    logger.write('epoch: %d, images: %d, acc: %.4f, auc: %.4f, f1: %.4f, presicion: %.4f, recall: %.4f \n' % \
    (epoch, len(labels), acc, auc, f1, presicion, recall))
    return best_auc

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


    composed_transforms_train = transforms.Compose([
            trforms.FixedResize(size=(args.input_size + 8, args.input_size + 8)),
            trforms.RandomCrop(size=(args.input_size, args.input_size)),
            trforms.RandomHorizontalFlip(),
            trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            trforms.ToTensor()])

    composed_transforms_test = transforms.Compose([
            trforms.FixedResize(size=(args.input_size, args.input_size)),
            trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            trforms.ToTensor()])

    for fold in range(5):
        if args.backbone == 'resnet18':
            backbone = resnet18(pretrained=True)
            backbone.fc = nn.Linear(in_features=backbone.fc.in_features, out_features=2)
        elif args.backbone == 'convnext_tiny':
            backbone = convnext_tiny(pretrained=True)
            backbone.classifier = nn.Sequential(nn.Flatten(1), nn.Linear(768, 2))
        elif args.backbone == 'dense':
            backbone = densenet121(pretrained=True)
            backbone.classifier = nn.Linear(1024, 2)
        else:
            raise NotImplementedError

        # set path
        args.fold = fold
        save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        if args.loss_func == 'acl': 
            save_dir = os.path.join(save_dir_root, 'run', args.loss_func + '-' + args.backbone + '-mode_std' + str(args.coff_std) + '-HQ_len' + str(args.HQ_len) + '-CQ_len' + str(args.CQ_len*args.batch_size) + '-fold_' + str(args.fold) + '-'+ str(args.seed))
        else:
            save_dir = os.path.join(save_dir_root, 'run', args.loss_func + '-' + args.backbone + '-fold' + str(args.fold) + '-'+str(args.seed))

        if not os.path.exists(os.path.join(save_dir, 'models')):
            os.makedirs(os.path.join(save_dir, 'models'))
        if os.path.exists(os.path.join(save_dir, 'models', 'best_backbone.pth')):
            continue
        trainset = tn3k_dataset.TN3KDataset(mode='train', fold=args.fold, transform=composed_transforms_train, return_size=False)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

        valset = tn3k_dataset.TN3KDataset(mode='val', fold=args.fold, transform=composed_transforms_test, return_size=False)
        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

        # logs
        log_dir = os.path.join(save_dir, datetime.now().strftime('%b%d_%H-%M-%M%S') + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir)
        logger = open(os.path.join(save_dir, 'log.txt'), 'w')
        logger.write('optim: SGD \nlr=%.4f\nupdate_lr_every=%d\nseed=%d\n' % (args.lr, args.update_lr_every, args.seed))

        # initialize model, optimizer and dataset for training
        if args.resume_epoch == 0:
            print('Training from ImageNet pretrained model...')
        else:
            backbone_resume_path = os.path.join(save_dir, 'models', 'backbone_epoch-' + str(args.resume_epoch - 1) + '.pth')
            print('Initializing weights from: {}, epoch: {}...'.format(save_dir, args.resume_epoch))
            backbone.load_state_dict(torch.load(backbone_resume_path, map_location=lambda storage, loc: storage))

        # torch.cuda.set_device(device=0)
        backbone.cuda()

        curlr = args.lr
        backbone_optim = optim.SGD(
            backbone.parameters(),
            lr=curlr,
        )

        num_iter_tr = len(trainloader)
        nitrs = args.resume_epoch * num_iter_tr
        nsamples = args.batch_size * nitrs
        print('each_epoch_num_iter: %d' % (num_iter_tr))

        # curriculum learning settings for training 
        labels_list = []
        inconsistent_score_list = []
        recent_losses = []

        start_t = time.time()
        print('Training Network')
        print('save path: '+ save_dir)
        print("model: " + str(args.backbone) + "fold" + str(args.fold))
        th_mean = 0
        th_std = 0
        if args.coff_std > -1:
            alpha = args.coff_std
        else:
            alpha = 0.
            certainty_queue = []
        
        # train!
        best_auc = 0
        backbone.train()
        for epoch in range(args.resume_epoch, args.nepochs):
            for ii, sample_batched in enumerate(trainloader):
                img, label = sample_batched['image'], sample_batched['label']
                img, label = img.cuda(), label.cuda()
                feats = backbone.forward(img)
                labels_list.append(label)

                if args.loss_func == 'ce':
                    loss = utils.CELoss(logit=feats, target=label, reduction='mean')
                elif args.loss_func == 'superloss':
                    loss = utils.superloss(feats, label)
                elif args.loss_func == 'curriloss':
                    loss = utils.curriloss(feats, label)
                elif args.loss_func == 'acl':
                    if epoch < 3:
                        loss = utils.CELoss(logit=feats, target=label, reduction='mean')
                    else:
                        loss = utils.RCELoss(logit=feats, target=label, reduction='mean', th_mean=th_mean, th_std=alpha * th_std, image_names=sample_batched['label_name'], fold=args.fold)

                backbone_optim.zero_grad()
                loss.backward()
                backbone_optim.step()
                
                # Get loss
                trainloss = loss.item()
                if len(recent_losses) < args.log_every:
                    recent_losses.append(trainloss)
                else:
                    recent_losses[nitrs % len(recent_losses)] = trainloss
                nitrs += 1
                nsamples += args.batch_size
                if nitrs % args.log_every == 0:
                    meanloss = sum(recent_losses) / len(recent_losses)
                    writer.add_scalar('data/trainloss', meanloss, nsamples)

                # Get threshold
                for i in range(feats.shape[0]):
                    th_confidence = torch.nn.Softmax(dim=1)(feats)
                    if label[i].item() != torch.argmax(th_confidence[i]).item():
                        inconsistent_score_list.append(th_confidence[i][label[i].item()].item())
                        if len(inconsistent_score_list) > args.HQ_len:
                            inconsistent_score_list.pop(0)
                    if args.coff_std < 0:
                        certainty_queue.append(torch.max(th_confidence[i]).item())
                        if len(certainty_queue) > args.CQ_len * args.batch_size:
                            certainty_queue.pop(0)

                th_mean = np.mean(inconsistent_score_list)
                th_std = np.std(inconsistent_score_list)
                
                # update alpha
                if args.coff_std < 0:
                    alpha = np.mean(certainty_queue)
            print('epoch: %d timecost:%.2f' % (epoch, time.time() - start_t))

            if epoch > 10:
                best_auc = evaluate(backbone, valloader, epoch, best_auc, logger, writer, save_dir=save_dir)

            if epoch % args.update_lr_every == args.update_lr_every - 1:
                curlr = utils.lr_poly(args.lr, epoch, args.nepochs, 0.9)
                print('(poly lr policy) learning rate: ', curlr)
                backbone_optim = optim.SGD(
                    backbone.parameters(),
                    lr=curlr,
                )

if __name__ == '__main__':
    args = get_arguments()
    main(args)