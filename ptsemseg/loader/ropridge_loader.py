import os
import torch
import numpy as np
import scipy.misc as m
import cv2
import random
from torch.utils import data

import sys
sys.path.insert(0,"/data0/qilei_chen/Development/pytorch-semseg1")

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale
from pycocotools.coco import COCO

class ROPRidge_loader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
    number_of_class = 1
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
    ]

    label_colours = dict(zip(range(number_of_class), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
        "ropridge":[0.0, 0.0, 0.0]
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
        self,
        root,
        img_folder = "train2014",
        split="train",
        is_transform=False,
        img_size=(600, 800), #some are 480*640, some are 1200*1600
        augmentations=None,
        img_norm=True,
        version="ropridge",
        test_mode=False,
        annotation_folder = "annotations",
        annotation_filename = "ridge_in_one_instances_train2014.json"
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        
        self.root = root
        self.img_folder = img_folder
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 1
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}
        self.cocoAnno = COCO(os.path.join(root,annotation_folder,annotation_filename))
        self.imgIds = self.cocoAnno.getImgIds()
        random.shuffle(self.imgIds)
        '''
        print(self.imgIds)
        print(len(self.cocoAnno.getImgIds()))
        img = self.cocoAnno.loadImgs([3])[0]
        print(img)
        annIds = self.cocoAnno.getAnnIds(imgIds = [3])
        if len(annIds)>0:
            anns = self.cocoAnno.loadAnns(annIds)
            mask = self.cocoAnno.annToMask(anns[0])
            cv2.imshow("mask",np.array(mask,dtype=np.uint8)*255)
            cv2.waitKey(0)

        

            #else:
            #    print(num_ann)
        '''
        '''
        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
        '''
        self.void_classes = [0]
        self.valid_classes = [1]
        self.class_names = ["unlabelled","ropridge"]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(1)))
        '''
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))
        '''
    def __len__(self):
        """__len__"""
        return len(self.imgIds)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        imgId = self.imgIds[index]
        imgName = self.cocoAnno.loadImgs([imgId])[0]["file_name"]
        img_path = os.path.join(self.root,self.img_folder,imgName)

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        annIds = self.cocoAnno.getAnnIds(imgIds = [imgId])
        anns = self.cocoAnno.loadAnns(annIds)
        
        mask = self.cocoAnno.annToMask(anns[0])
        
        for ann in anns[1:]:
            mask += self.cocoAnno.annToMask(ann)
        
        mask[mask >=1]=1
        #mask[mask ==1]=2
        #print(mask.dtype)
        #print(np.unique(mask))

        lbl = self.encode_segmap(np.array(mask, dtype=np.uint8))
        #lbl = mask
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode

        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")

        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")
        '''
        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print(np.all(np.unique(lbl[lbl != self.ignore_index]))<self.n_classes)
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        '''
        np.unique(lbl)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()-1
        print(lbl.size())
        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index

        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        #print(np.unique(mask))
        return mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #augmentations = Compose([Scale(800), RandomRotate(180), RandomHorizontallyFlip(0.5)])
    augmentations = Compose([Scale(800), RandomHorizontallyFlip(0.5)])
    local_path = "/data0/qilei_chen/old_alien/AI_EYE_IMGS/ROP_DATASET_with_label/9LESIONS"
    dst = ROPRidge_loader(local_path, is_transform=True, augmentations=augmentations)
    
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        #import pdb

        #pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
    
