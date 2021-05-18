##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: speedinghzl02
## Modified by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import cv2
import pdb
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from PIL import Image, ImageOps, ImageFilter
import random
import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
from os.path import join, exists

# fine
fine_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
              "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
              "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
              "bremen/", "bochum/", "aachen/"]

# fine + croase
fine_croase_dirs = ['dresden', 'krefeld', 'karlsruhe', 'strasbourg', 'jena', 'hanover', 'nuremberg',
              'bad-honnef', 'tubingen', 'wuppertal', 'erfurt', 'stuttgart', 'konstanz', 'wurzburg',
              'oberhausen', 'dortmund', 'heilbronn', 'schweinfurt', 'bamberg', 'saarbrucken', 'troisdorf',
              'mannheim', 'bochum', 'konigswinter', 'muhlheim-ruhr', 'weimar', 'darmstadt',
              'hamburg', 'bremen', 'aachen','bayreuth', 'augsburg', 'cologne', 'ulm',
              'zurich', 'heidelberg', 'erlangen', 'freiburg', 'duisburg', 'monchengladbach', 'dusseldorf']

# croase
croase_dirs = ['dresden', 'karlsruhe', 'nuremberg', 'bayreuth', 'augsburg',
              'bad-honnef', 'wuppertal', 'konstanz', 'wurzburg', 'troisdorf',
              'oberhausen', 'dortmund', 'heilbronn', 'schweinfurt', 'bamberg',
              'mannheim', 'konigswinter', 'muhlheim-ruhr', 'saarbrucken',
              'heidelberg', 'erlangen', 'freiburg', 'duisburg']

# test num_workers
# train_dirs = ['dresden', 'karlsruhe']

val_dirs = ["frankfurt/", "munster/", "lindau/"]
# val_dirs = ["lindau/"]


# test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]
test_dirs = ["dlut"]

class CitySegmentationTrain(data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path, max_iters=None, crop_size=(321, 321),
        scale=True, mirror=True, ignore_label=255, use_aug=False, network="renset101", data_type="fine"):
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label     
        self.is_mirror = mirror
        self.network = network
        
        self.img_dir = cityscapes_data_path
        self.label_dir = os.path.join(cityscapes_meta_path, "label_imgs/")

        self.examples = []
        if data_type == "fine": 
            train_dirs = fine_dirs
        elif data_type == "croase":
            train_dirs = croase_dirs
        for train_dir in train_dirs:
            train_img_dir_path = os.path.join(self.img_dir, train_dir)

            file_names = os.listdir(train_img_dir_path)
            
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = os.path.join(train_img_dir_path, file_name)
                
                label_img_path = os.path.join(self.label_dir, img_id + ".png")

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)
        # print(len(self.examples))    
        if not max_iters==None:
            self.examples = self.examples*int(np.ceil(float(max_iters)/len(self.examples)))
            
        self.num_examples = len(self.examples)
        # print(self.num_examples)

    def generate_sequence_img_path(self, img_path, example):
        # num = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        num = random.choice([-3, -2, -1, 1, 2, 3])
        # num = random.choice([-1, 1])
        # Fine data search for sequence data
        img_path_split = img_path.split('_')
        frame1_id = int(img_path_split[3])
        img_path_split[0] = '/media/8TB/yichen/cityscapes/sequence'
        img_path_split[3] = str(str(frame1_id + num).zfill(6))
        img2_path = '_'.join(img_path_split)

        return img2_path

    def generate_scale_label(self, image, image2, label):
        f_scale = 0.5 + random.randint(0, 16) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_AREA)
        image2 = cv2.resize(image2, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_AREA)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, image2, label
   
    def data_augmentation(self, img_path, img2_path, label_img_path):

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)

        size = image.shape
        if self.scale:
            image, image2, label = self.generate_scale_label(image, image2, label)
        image = np.asarray(image, np.float32)
        image2 = np.asarray(image2, np.float32)

        mean = (102.9801, 115.9465, 122.7717)

        image = image[:,:,::-1]
        image -= mean
        image2 = image2[:,:,::-1]
        image2 -= mean

        img_h, img_w = label.shape

        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))

            img_pad2 = cv2.copyMakeBorder(image2, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))

            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, img_pad2, label_pad = image, image2, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)

        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image2 = np.asarray(img_pad2[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)        
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))

        if self.is_mirror:
            flip = random.choice([-1, 1])
            image = image[:, :, ::flip]
            image2 = image2[:, :, ::flip]
            label = label[:, ::flip]

        return image, image2, label,size

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]

        label_img_path = example["label_img_path"]

        img2_path = self.generate_sequence_img_path(img_path, example)

        image, image2, label, size = self.data_augmentation(img_path, img2_path, label_img_path)
        
        return image.copy(), image2.copy(), label.copy(), np.array(size)

    def __len__(self):
        return self.num_examples



class CitySegmentationVal(data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path, max_iters=None, crop_size=(321, 321),
        scale=True, mirror=True, ignore_label=255, use_aug=False, network="renset101"):
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label     
        self.is_mirror = mirror
        self.network = network
        
        self.img_dir = cityscapes_data_path
        self.label_dir = os.path.join(cityscapes_meta_path, "label_imgs/")

        self.examples = []
        for train_dir in val_dirs:
            train_img_dir_path = os.path.join(self.img_dir, train_dir)

            file_names = os.listdir(train_img_dir_path)
            
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = os.path.join(train_img_dir_path, file_name)
                
                label_img_path = os.path.join(self.label_dir, img_id + ".png")

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                example["file_names"] = file_name
                self.examples.append(example)
        
        self.num_examples = len(self.examples)
        # print(self.num_examples)
 
    def generate_scale_label(self, image, image2, label):
        f_scale = 0.5 + random.randint(0, 16) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        image2 = cv2.resize(image2, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, image2, label
   
    def generate_sequence_img_path(self, img_path):
        # deviation = random.randint(-num, num) 
        num = random.choice([-1, 1])
        img_path_split = img_path.split('_')
        frame1_id = int(img_path_split[3])
        img_path_split[0] = '/media/8TB/yichen/cityscapes/sequence'
        img_path_split[3] = str(str(frame1_id + num).zfill(6))
        img2_path = '_'.join(img_path_split)
        return img2_path

    def data_process(self, img_path, img2_path, label_img_path):

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)

        size = image.shape
        if self.scale:
            image, image2, label = self.generate_scale_label(image, image2, label)

        image = np.asarray(image, np.float32)
        image2 = np.asarray(image2, np.float32)
        
        mean = (102.9801, 115.9465, 122.7717)
        image = image[:,:,::-1]
        image -= mean
        image2 = image2[:,:,::-1]
        image2 -= mean

        image = image.transpose((2,0,1))
        image2 = image2.transpose((2,0,1))

        if self.is_mirror:
            flip = random.choice([-1, 1])
            image = image[:, :, ::flip]
            image2 = image2[:, :, ::flip]
            label = label[:, ::flip]

        return image, image2, label, size

    def __getitem__(self, index):
        example = self.examples[index]
        file_name = example["file_names"]
        img_path = example["img_path"]
        label_img_path = example["label_img_path"]

        img2_path = self.generate_sequence_img_path(img_path)
        
        image, image2, label, size = self.data_process(img_path, img2_path, label_img_path) 

        return image.copy(), image2.copy(), label.copy(), np.array(size), file_name

    def __len__(self):
        return self.num_examples

class CitySegmentationTest(data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path, max_iters=None, crop_size=(321, 321),
        scale=True, mirror=True, ignore_label=255, use_aug=False, network="renset101"):
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label     
        self.is_mirror = mirror
        self.network = network
        
        self.img_dir = cityscapes_data_path

        self.examples = []
        for train_dir in test_dirs:
            train_img_dir_path = os.path.join(self.img_dir, train_dir)

            file_names = os.listdir(train_img_dir_path)
            
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = os.path.join(train_img_dir_path, file_name)
                

                example = {}
                example["img_path"] = img_path
                example["img_id"] = img_id
                example["file_names"] = file_name
                self.examples.append(example)
        
        self.num_examples = len(self.examples)
        # print(self.num_examples)

    def generate_scale(self, image):
        f_scale = 0.5 + random.randint(0, 16) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        return image
    
    def generate_sequence_img_path(self, img_path):
        num = random.choice([-1, 1])
        
        img_path_split = img_path.split('_')
        frame1_id = int(img_path_split[3])
        img_path_split[0] = '/media/8TB/yichen/cityscapes/sequence'
        img_path_split[3] = str(str(frame1_id+num).zfill(6))
        img2_path = '_'.join(img_path_split)
        return img2_path

    def data_process(self, img_path, img2_path):

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

        size = image.shape
        if self.scale:
            image, image2 = self.generate_scale(image, image2)

        image = np.asarray(image, np.float32)
        image2 = np.asarray(image2, np.float32)
        
        mean = (102.9801, 115.9465, 122.7717)
        image = image[:,:,::-1]
        image -= mean
        image2 = image2[:,:,::-1]
        image2 -= mean

        image = image.transpose((2,0,1))
        image2 = image2.transpose((2,0,1))

        if self.is_mirror:
            flip = random.choice([-1, 1])
            image = image[:, :, ::flip]
            image2 = image2[:, :, ::flip]

        return image, image2, size

    def __getitem__(self, index):
        example = self.examples[index]
        file_name = example["file_names"]
        img_path = example["img_path"]

        img2_path = self.generate_sequence_img_path(img_path)
        
        image, image2, size = self.data_process(img_path, img2_path) 

        return image.copy(), image2.copy(), np.array(size), file_name
 
    def __len__(self):
        return self.num_examples

if __name__ == '__main__':

    dst = CitySegmentationTrain(crop_size=(1024, 2048), cityscapes_data_path = '/media/8TB/yichen/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/',
    cityscapes_meta_path = '/media/8TB/yichen/cityscapes/meta/')
    trainloader = data.DataLoader(dst, batch_size=1, num_workers=1, shuffle=True)
    for step, index in enumerate(trainloader):
        pass

