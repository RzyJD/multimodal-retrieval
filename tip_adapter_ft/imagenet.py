"""Binary ImageFolder loader: 0 is the target, 1 is Others.

The ImageNet name is inherited from upstream; no ImageNet benchmark is required.
"""

import os
import random
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms

class ImageNet():

    dataset_dir = 'datasets'

    def __init__(self, root, num_shots, preprocess,imagenet_classes):
        # 数据集目录拼接，输出为join内组间路径的拼接
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        # 图像输出大小224像素，裁剪为原来的50%到100，归一化%
        train_preprocess = transforms.Compose([
                                                transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                                            ])
        test_preprocess = preprocess
        self.train_dir=os.path.join(self.image_dir, 'train')
        self.val_dir = os.path.join(self.image_dir, 'val')
        self.test_dir=os.path.join(self.image_dir, 'val')
        # 划分数据集
        self.train=torchvision.datasets.ImageFolder(self.train_dir, train_preprocess)
        self.val=torchvision.datasets.ImageFolder(self.val_dir, test_preprocess)
        self.test=torchvision.datasets.ImageFolder(self.test_dir, test_preprocess)

        if self.train.classes != ['0', '1'] or self.test.classes != ['0', '1']:
            raise ValueError('Prepared train/val must contain 0 (target) and 1 (Others). See DATASET.md.')
        if num_shots < 1:
            raise ValueError('shots must be positive')


        #self.train为对象，属性有imgs(包含图像路径和标签的元祖列表）,target（包含标签的列表，数字形式）,classes（类别名称的列表）,class_to_idx（类别名称到索引的映射）
        self.classnames = imagenet_classes

        split_by_label_dict = defaultdict(list)
        for i in range(len(self.train.imgs)):
            split_by_label_dict[self.train.targets[i]].append(self.train.imgs[i])
        imgs = []
        targets = []

        for label, items in split_by_label_dict.items():
            if len(items) < num_shots:
                raise ValueError(f'Class {label} has {len(items)} images, fewer than {num_shots} shots.')
            imgs = imgs + random.sample(items, num_shots)
            targets = targets + [label for i in range(num_shots)]
        self.train.imgs = imgs
        self.train.targets = targets
        self.train.samples = imgs
