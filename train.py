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
import numpy
import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from dataset import get_segmentation_dataset
from config import Parameters
import random
import timeit
import logging
import pdb
# from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.criterion import CriterionCrossEntropy,  CriterionDSN, CriterionOhemDSN_single
from utils.parallel import DataParallelModel, DataParallelCriterion
from network import get_resnet101_base_oc_dsn
from dataset.cityscapes import CitySegmentationVal
from utils.eval_func import *
import torch.distributed as dist
from itertools import cycle
from generate_submit import id2trainId
import torch.multiprocessing as mp
from utils.eval_func import label_img_to_color

start = timeit.default_timer()

args = Parameters().parse()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
   
def test_poly(min_lr, max_lr, iter, max_iter): 
    return min_lr + (max_lr-min_lr)*(iter/max_iter)

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def find_learning_rate(optimizer, i_iter):
    lr = test_poly(max_lr = args.learning_rate, min_lr = 1e-5, iter = i_iter, max_iter = args.num_steps)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def main():
    ignore_label = args.ignore_label
    id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                  3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                  7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                  14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                  18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                  28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    # device setting
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda:{}".format(args.local_rank))    # cuda:0 cuda:1
    rank = torch.distributed.get_rank()  # 0 ,1
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_gpus = torch.cuda.device_count() 
    if rank == 0: print('Let us use', num_gpus, 'GPUs!')

    if rank == 0: writer = SummaryWriter(args.snapshot_dir)

    h, w = map(int, args.input_size.split(','))

    input_size = (h, w)
    cudnn.enabled = True

    deeplab = get_resnet101_base_oc_dsn(num_classes = args.num_classes)
    
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    saved_state_dict = torch.load(args.restore_from, map_location = map_location)
    new_params = deeplab.state_dict().copy()

    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0]=='fc' and not  i_parts[0]=='last_linear' and not  i_parts[0]=='classifier':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i] 

    if args.start_iters > 0:
        deeplab.load_state_dict(saved_state_dict['network'], strict = True)
        # deeplab.load_state_dict(saved_state_dict, strict = True)
    else:
        deeplab.load_state_dict(new_params, strict=False)
 
    model= torch.nn.SyncBatchNorm.convert_sync_batchnorm(deeplab)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model,  device_ids=[args.local_rank])

    model.float()

    criterion = CriterionCrossEntropy()
    if "dsn" in args.method:
        if args.ohem:
            if args.ohem_single:
                if rank == 0: print('use ohem-sigle')
                criterion = CriterionOhemDSN_single(thres=args.ohem_thres, min_kept=args.ohem_keep, dsn_weight=float(args.dsn_weight)).cuda(rank)
        else:
            criterion = CriterionDSN(dsn_weight=float(args.dsn_weight), use_weight=True).cuda(rank)


    cudnn.benchmark = True


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    fine_dataset = get_segmentation_dataset(args.dataset, cityscapes_data_path = '/media/8TB/yichen/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/', cityscapes_meta_path = '/media/8TB/yichen/cityscapes/meta/',
                    max_iters=args.num_steps*args.batch_size*num_gpus, crop_size=input_size, 
                    scale=args.random_scale, mirror=args.random_mirror, network=args.network, data_type="fine") 
    fine_sampler = torch.utils.data.distributed.DistributedSampler(fine_dataset)

    fine_loader = torch.utils.data.DataLoader(dataset=fine_dataset, batch_size=args.batch_size, sampler=fine_sampler,
                                           shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    val_dataset = CitySegmentationVal(cityscapes_data_path = '/media/8TB/yichen/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/',
                                      cityscapes_meta_path = '/media/8TB/yichen/cityscapes/meta/',
                                      crop_size=(1024, 2048), scale=False, mirror=False, network=args.network)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1,
                                           shuffle=False, num_workers=16, pin_memory=True)

    torch.cuda.empty_cache()

    # if args.start_iters > 0:
    #     optimizer = saved_state_dict['optimizer']
    #     if rank == 0: print('Load optimizer successfully')
    # else:
    if args.optimizer == 'SGD':
        if rank == 0: print('Use SGD as Optimizer')
        optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, deeplab.parameters()), 'lr': args.learning_rate }], 
                    lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer =='Adam':
        if rank == 0: print('Use Adam as Optimizer')
        optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, deeplab.parameters()), 'lr': args.learning_rate }], 
                    lr=args.learning_rate, weight_decay=args.weight_decay)

    
    optimizer.zero_grad()

    # Only Fine data
    fine_iters_pre_epoch = int(2975/args.batch_size/num_gpus)
    
    for i_iter, batch in enumerate(fine_loader):
        model.train()
        dist.barrier()

        i_iter += args.start_iters
        if rank == 0:
            if (i_iter % fine_iters_pre_epoch == 0)&(i_iter>=fine_iters_pre_epoch):
                fine_sampler.set_epoch(i_iter//fine_iters_pre_epoch)
                print('####### shuffle_fine #########',i_iter, i_iter//fine_iters_pre_epoch)
        sys.stdout.flush()
        images, images2, label, size_  = batch
        images = Variable(images.cuda()); images2 = Variable(images2.cuda())
        label = Variable(label.long().cuda())
        label2 = Variable(label.long().cuda())

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        if args.fix_lr:
            lr = args.learning_rate

        preds = model(images, images2)
        
        if args.warp_loss == 'KL':
            loss, x_loss, x_dsn_loss, x_value_loss, x_warp_loss, KL1, KL2 = criterion(preds, label, label2, args.warp_loss)
        elif args.warp_loss == 'L2':
            loss, x_loss, x_dsn_loss, x_value_loss, x_warp_loss, l2 = criterion(preds, label, label2, args.warp_loss)
        loss.backward()
        optimizer.step()

        mean = (102.9801, 115.9465, 122.7717)

        if (rank == 0)&(i_iter % 100 == 0):
            images = images[0].cpu().data.numpy()
            images = images.transpose((1, 2, 0))
            images += mean
            images = images[:, :, ::-1]
            images = np.array(images, dtype=np.uint8)
            writer.add_images('image1', images, dataformats='HWC') 

            images = images2[0].cpu().data.numpy()
            images = images.transpose((1, 2, 0))
            images += mean
            images = images[:, :, ::-1]
            images = np.array(images, dtype=np.uint8)
            writer.add_images('images2', images, dataformats='HWC') 
            
            label = label[0].cpu().data.numpy()
            label = label_img_to_color(label)
            label = np.array(label, dtype=np.uint8)
            writer.add_image('label', label, dataformats='HWC')

            result = F.interpolate(input=preds[0], size=(769, 769), mode='bilinear', align_corners=True)
            result = result.cpu().data.numpy()
            seg_pred = np.asarray(np.argmax(result, axis=1), dtype=np.uint8)[0]
            seg_pred = label_img_to_color(seg_pred)
            seg_pred = np.array(seg_pred, dtype=np.uint8)
            writer.add_image('final_segmantation_map', seg_pred, dataformats='HWC')

            result = F.interpolate(input=preds[1], size=(769, 769), mode='bilinear', align_corners=True)
            result = result.cpu().data.numpy()
            seg_pred = np.asarray(np.argmax(result, axis=1), dtype=np.uint8)[0]
            seg_pred = label_img_to_color(seg_pred)
            seg_pred = np.array(seg_pred, dtype=np.uint8)
            writer.add_image('x_dsn_segmantation_map', seg_pred, dataformats='HWC')

            result = F.interpolate(input=preds[2], size=(769, 769), mode='bilinear', align_corners=True)
            result = result.cpu().data.numpy()
            seg_pred = np.asarray(np.argmax(result, axis=1), dtype=np.uint8)[0]
            seg_pred = label_img_to_color(seg_pred)
            seg_pred = np.array(seg_pred, dtype=np.uint8)
            writer.add_image('x_cls_segmantation_map', seg_pred, dataformats='HWC')

            result = F.interpolate(input=preds[3], size=(769, 769), mode='bilinear', align_corners=True)
            result = result.cpu().data.numpy()
            seg_pred = np.asarray(np.argmax(result, axis=1), dtype=np.uint8)[0]
            seg_pred = label_img_to_color(seg_pred)
            seg_pred = np.array(seg_pred, dtype=np.uint8)
            writer.add_image('warp_segmantation_map', seg_pred, dataformats='HWC')

            result = F.interpolate(input=preds[4], size=(769, 769), mode='bilinear', align_corners=True)
            result = result.cpu().data.numpy()
            seg_pred = np.asarray(np.argmax(result, axis=1), dtype=np.uint8)[0]
            seg_pred = label_img_to_color(seg_pred)
            seg_pred = np.array(seg_pred, dtype=np.uint8)
            writer.add_image('y_cls_segmantation_map', seg_pred, dataformats='HWC')

        if rank == 0:
            writer.add_scalar('learning_rate', lr, i_iter)

            writer.add_scalar('fine loss', loss.data.cpu().numpy(), i_iter)

            writer.add_scalar('x_loss', x_loss.data.cpu().numpy(), i_iter)
            writer.add_scalar('x_dsn_loss', x_dsn_loss.data.cpu().numpy(), i_iter)
            writer.add_scalar('x_value_loss', x_value_loss.data.cpu().numpy(), i_iter)
            writer.add_scalar('x_warp_loss', x_warp_loss.data.cpu().numpy(), i_iter)

            if args.warp_loss == 'KL':
                writer.add_scalar('KL1', KL1.data.cpu().numpy(), i_iter)
                writer.add_scalar('KL2', KL2.data.cpu().numpy(), i_iter)
            elif args.warp_loss == 'L2':
                writer.add_scalar('MSE', l2.data.cpu().numpy(), i_iter)

            print('i_iter', i_iter, 'loss:', loss.data.cpu().numpy(), 'lr:', lr)

        state = {'network':deeplab.state_dict(), 'optimizer':optimizer}
        if i_iter >= args.num_steps-1:
            if rank == 0:
                print('save model ...')
                torch.save(state, osp.join(args.snapshot_dir, str(args.num_steps)+'.pth'))
            break
        if (i_iter % args.save_pred_every == 0)&(rank == 0):
            torch.save(state, osp.join(args.snapshot_dir, str(i_iter)+'.pth'))     

        # if (i_iter % 10000 == 0)&(i_iter != 0)&(i_iter != args.start_iters)|(i_iter == args.num_steps):
        if (i_iter % 1000 == 0)&(i_iter != 0)|(i_iter == args.num_steps-1):
            if rank == 0:
                model.eval()

                data_list = []
                confusion_matrix = np.zeros((args.num_classes,args.num_classes))

                palette = get_palette(20)

                image_id = 0
                with torch.no_grad():
                    for index, batch in enumerate(val_loader):
                        if (rank == 0)&(index % 2) == 0:
                            print('%d processd'%(index))
                            sys.stdout.flush()
                        image, image2, label, size, name = batch
                        size = size[0].numpy()
                        output = predict_whole_img(model.module, image.numpy(), image2.numpy(), args.num_classes, 
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
                    
                    if rank == 0: writer.add_scalar('miou', mean_IU, i_iter)
                    print('meanIU', mean_IU)
    end = timeit.default_timer()
    if rank == 0:
        print(end-start,'seconds')

if __name__ == '__main__':
    main()
    dist.destroy_process_group()
