import json
import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
import cv2
import random

def get_bbox(mask):
    mask = cv2.imread(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    stats = stats[1][:4]
    return stats


def make_dataset(root, seed):
    imgs = []
    img_labels = {}

    # get label dict
    with open(root + 'label4trainval.csv', 'r') as f:
        lines = f.readlines()

    for idx in range(len(lines)):
        line = lines[idx]
        name, label = line.strip().split(',')
        img_labels[name] = label
    # get image path
    img_names = os.listdir(root + 'trainval-image/')
    for i in seed:        
        img_name = img_names[i]
        img = os.path.join(root + 'trainval-image/', img_name)
        mask = os.path.join(root + 'trainval-mask/', img_name)
        if int(img_labels[img_name]) == 1:
            imgs.append((img, mask, img_labels[img_name]))
        imgs.append((img, mask, img_labels[img_name]))
    return imgs

def make_validset(root, seed):
    imgs = []
    img_labels = {}

    # get label dict
    with open(root + 'label4trainval.csv', 'r') as f:
        lines = f.readlines()

    for idx in range(len(lines)):
        line = lines[idx]
        name, label = line.strip().split(',')
        img_labels[name] = label
    # get image path
    img_names = os.listdir(root + 'trainval-image/')
    for i in seed:        
        img_name = img_names[i]
        img = os.path.join(root + 'trainval-image/', img_name)
        mask = os.path.join(root + 'trainval-mask/', img_name)
        imgs.append((img, mask, img_labels[img_name]))
    return imgs

def make_testset(root):
    imgs = []
    img_labels = {}

    # get label dict
    with open(root + 'label4test.csv', 'r') as f:
        lines = f.readlines()

    for idx in range(0, len(lines)):
        line = lines[idx]
        name, label = line.strip().split(',')
        img_labels[name] = label

    # get image path
    img_names = os.listdir(root + 'test-image/')
    for img_name in img_names:
        img = os.path.join(root + 'test-image/', img_name)
        mask = os.path.join(root + 'test-mask/', img_name)
        imgs.append((img, mask, img_labels[img_name]))
    return imgs


class TN3KDataset(data.Dataset):
    def __init__(self, mode='train', transform=None, return_size=False, fold=0):
        self.mode = mode
        self.transform = transform
        self.return_size = return_size

        root = './datasets/tn3k/'
        
        trainvaltest = json.load(open(root + 'tn3k-trainval-fold' + str(fold) + '.json', 'r'))
        
        if mode == 'train':
            imgs = make_dataset(root, trainvaltest['train'])
        elif mode == 'val':
            imgs = make_validset(root, trainvaltest['val'])
        elif mode == 'test':
            imgs = make_testset(root)

        self.imgs = imgs
        self.transform = transform
        self.return_size = return_size

    def __getitem__(self, item):
        
        image_path, mask_path, label = self.imgs[item]

        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
        assert os.path.exists(mask_path), ('{} does not exist'.format(mask_path))

        image = Image.open(image_path).convert('RGB')
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = mask / mask.max()
        mask = Image.fromarray(mask.astype(np.uint8)) # 处理成了一个灰度图
        w, h = image.size
        size = (h, w)

        sample = {'image': image, 'mask':mask, 'label': int(label)}

        if self.transform:
            sample = self.transform(sample)
        # if self.return_size:
        #     sample['size'] = torch.tensor(size)

        label_name = os.path.basename(image_path)
        sample['label_name'] = label_name

        return sample

    def __len__(self):
        return len(self.imgs)