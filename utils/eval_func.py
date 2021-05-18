import matplotlib
matplotlib.use('Agg')
import argparse
import scipy
from scipy import ndimage
import torch, cv2
import numpy as np
import numpy.ma as ma
import sys
import pdb
import torch

from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from dataset import get_segmentation_dataset
from config import Parameters
from collections import OrderedDict
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage

import matplotlib.pyplot as plt
import torch.nn as nn
torch_ver = torch.__version__[:3]

def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        255: [105, 105, 105]
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    palette = [0] * (num_cls * 3)
    palette[0:3] = (128, 64, 128)       # 0: 'road' 
    palette[3:6] = (244, 35,232)        # 1 'sidewalk'
    palette[6:9] = (70, 70, 70)         # 2''building'
    palette[9:12] = (102,102,156)       # 3 wall
    palette[12:15] =  (190,153,153)     # 4 fence
    palette[15:18] = (153,153,153)      # 5 pole
    palette[18:21] = (250,170, 30)      # 6 'traffic light'
    palette[21:24] = (220,220, 0)       # 7 'traffic sign'
    palette[24:27] = (107,142, 35)      # 8 'vegetation'
    palette[27:30] = (152,251,152)      # 9 'terrain'
    palette[30:33] = ( 70,130,180)      # 10 sky
    palette[33:36] = (220, 20, 60)      # 11 person
    palette[36:39] = (255, 0, 0)        # 12 rider
    palette[39:42] = (0, 0, 142)        # 13 car
    palette[42:45] = (0, 0, 70)         # 14 truck
    palette[45:48] = (0, 60,100)        # 15 bus
    palette[48:51] = (0, 80,100)        # 16 train
    palette[51:54] = (0, 0,230)         # 17 'motorcycle'
    palette[54:57] = (119, 11, 32)      # 18 'bicycle'
    palette[57:60] = (105, 105, 105)
    return palette


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_sliding(net, image, image2, tile_size, classes, method, scale=1):
    if scale != 1:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        scaled_img2 = ndimage.zoom(image2, (1.0, 1.0, scale, scale), order=1, prefilter=False)
    else:
        scaled_img, scaled_img2 = image, image2

    N_, C_, H_, W_ = scaled_img.shape

    full_probs = np.zeros((N_, H_, W_, classes))
    count_predictions = np.zeros((N_, H_, W_, classes))
    overlap = 0
    stride_h = ceil(tile_size[0] * (1 - overlap))
    stride_w = ceil(tile_size[1] * (1 - overlap))
    tile_rows = int(ceil((H_ - tile_size[0]) / stride_h) + 1)  # strided convolution formula
    tile_cols = int(ceil((W_ - tile_size[1]) / stride_w) + 1)
    print("Need %i x %i prediction tiles @ stride %i px, %i py" % (tile_cols, tile_rows, stride_h, stride_w))

    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride_w)
            y1 = int(row * stride_h)
            x2 = min(x1 + tile_size[1], W_)
            y2 = min(y1 + tile_size[0], H_)
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = scaled_img[:, :, y1:y2, x1:x2]
            img2 = scaled_img2[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            padded_img2 = pad_image(img2, tile_size)
            tile_counter += 1
            print("Predicting tile %i" % tile_counter)
            padded_prediction_ = net(Variable(torch.from_numpy(padded_img).cuda()), 
                                     Variable(torch.from_numpy(padded_img2).cuda()))
    
            if 'dsn' in method or 'center' in method:
                padded_prediction = padded_prediction_[0]

            padded_prediction = F.interpolate(input=padded_prediction, size=tile_size, mode='bilinear', align_corners=True)

            padded_prediction = padded_prediction.cpu().data.numpy().transpose(0,2,3,1)

            prediction = padded_prediction[:, 0:img.shape[2], 0:img.shape[3], :]

            count_predictions[:, y1:y2, x1:x2] += 1
            full_probs[:, y1:y2, x1:x2] += prediction 

    full_probs /= count_predictions
    full_probs = ndimage.zoom(full_probs, (1., 1./scale, 1./scale, 1.),
        order=1, prefilter=False)
    return full_probs


def predict_whole_img(net, image, image2, classes, method, scale):
    """
         Predict the whole image w/o using multiple crops.
         The scale specify whether rescale the input image before predicting the results.
    """
    N_, C_, H_, W_ = image.shape
    if scale != 1:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        scaled_img2 = ndimage.zoom(image2, (1.0, 1.0, scale, scale), order=1, prefilter=False)
    else:
        scaled_img = image; scaled_img2 = image2
    full_prediction_ = net(Variable(torch.from_numpy(scaled_img)).cuda(), 
                           Variable(torch.from_numpy(scaled_img2)).cuda())
    if 'dsn' in method or 'center' in method or 'fuse' in method:
        full_prediction = full_prediction_[0]
    else:
        full_prediction = full_prediction_

    full_prediction = F.interpolate(input=full_prediction, size=(H_, W_), mode='bilinear', align_corners=True)
    result = full_prediction.cpu().data.numpy().transpose(0,2,3,1)
    return result



def predict_multi_scale(net, image, image2, scales, tile_size, classes, flip_evaluation, method):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((N_, H_, W_, classes))  
    for scale in scales:
        scale = float(scale)
        print("Predicting image scaled by %f" % scale)
        sys.stdout.flush()
        if scale <= 1.0:
            scaled_probs = predict_whole_img(net, image, image2, classes, method, scale=scale)
        else:        
            scaled_probs = predict_sliding(net, image, image2, (1024,2048), classes, method, scale=scale)
        if flip_evaluation == 'True':
            if scale <= 1.0:
                flip_scaled_probs = predict_whole_img(net, image[:,:,:,::-1].copy(), image2[:,:,:,::-1].copy(), classes, method, scale=scale)
            else:
                flip_scaled_probs = predict_sliding(net, image[:,:,:,::-1].copy(), image2[:,:,:,::-1].copy(), (1024,2048), classes, method, scale=scale)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:,:,::-1])
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs


def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix


def id2trainId(label, id_to_trainid, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy
