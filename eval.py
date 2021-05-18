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

import torch.backends.cudnn as cudnn
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
from network import get_resnet101_base_oc_dsn

import matplotlib.pyplot as plt
import torch.nn as nn
from dataset.cityscapes import CitySegmentationVal
import torch.distributed as dist
from utils.eval_func import *
torch_ver = torch.__version__[:3]

def main():
    args = Parameters().parse()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    # device setting
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda:{}".format(args.local_rank))    # cuda:0 cuda:1
    rank = torch.distributed.get_rank()  # 0 ,1

    ignore_label = 255
    id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
          3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
          7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
          14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
          18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
          28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    cudnn.enabled = True

    # output_path = args.output_path
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    deeplab = get_resnet101_base_oc_dsn(num_classes = args.num_classes)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    saved_state_dict = torch.load(args.restore_from, map_location = map_location)
    deeplab.load_state_dict(saved_state_dict['network'])
    # deeplab.load_state_dict(saved_state_dict)

    model= torch.nn.SyncBatchNorm.convert_sync_batchnorm(deeplab)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model,  device_ids=[args.local_rank])
                                                      # broadcast_buffers=False, find_unused_parameters=True)

    model.eval()

    val_dataset = CitySegmentationVal(cityscapes_data_path = '/media/8TB/yichen/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/',
                                      cityscapes_meta_path = '/media/8TB/yichen/cityscapes/meta/',
                                      crop_size=(1024, 2048), scale=False, mirror=False, network=args.network)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1,
                                           shuffle=False, num_workers=16, pin_memory=True)

    data_list = []
    confusion_matrix = np.zeros((args.num_classes,args.num_classes))

    palette = get_palette(20)

    image_id = 0
    with torch.no_grad():
        for index, batch in enumerate(val_loader):
            print('%d processd'%(index))
            sys.stdout.flush()
            image, image2, label, size, name = batch
            size = size[0].numpy()
            if args.use_ms == 'True': 
                print('use msc val')
                output = predict_multi_scale(model, image.numpy(), image2.numpy(), ([0.75, 1, 1.25]), input_size, 
                    args.num_classes, args.use_flip, args.method)
            else:
                if args.use_flip == 'True':
                    output = predict_multi_scale(model, image.numpy(), image2.numpy(), ([args.whole_scale]), input_size, 
                        args.num_classes, args.use_flip, args.method)
                else:
                    output = predict_whole_img(model, image.numpy(), image2.numpy(), args.num_classes, 
                        args.method, scale=float(args.whole_scale))
            seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
            m_seg_pred = ma.masked_array(seg_pred, mask=torch.eq(label, 255))
            ma.set_fill_value(m_seg_pred, 20)
            seg_pred = m_seg_pred

            seg_gt = np.asarray(label.numpy()[:,:size[0],:size[1]], dtype=np.int)
            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]
            confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)
                
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()
        
        print('meanIU', mean_IU, args.restore_from)

        sys.stdout.flush()

if __name__ == '__main__':
    main()
