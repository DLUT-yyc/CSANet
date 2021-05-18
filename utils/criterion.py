import pdb
import torch.nn as nn
import math
import os
import sys
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from loss import OhemCrossEntropy2d, CrossEntropy2d
import scipy.ndimage as nd

torch_ver = torch.__version__[:3]

class CriterionCrossEntropy(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if torch_ver == '0.4':
            scale_pred = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.interpolate(input=preds, size=(h, w), mode='bilinear')
        loss = self.criterion(scale_pred, target)
        return loss
    

class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, dsn_weight=0.4):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        if use_weight:
            # print("w/ class balance")
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            # print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, x_label, y_label, warp_loss):
        h, w = x_label.size(1), x_label.size(2)

        result = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(result, x_label)

        x_dsn = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(x_dsn, x_label)
        
        loss3 = 0

        x_cls = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        loss4 = self.criterion(x_cls, x_label)

        x_warp = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        loss5 = self.criterion(x_warp, x_label)
        if warp_loss == 'KL': 
            KL = torch.nn.KLDivLoss()
            kl1 = KL(F.log_softmax(x_cls, dim=1), F.softmax(x_warp, dim=1))
            kl2 = KL(F.log_softmax(x_warp, dim=1), F.softmax(x_cls, dim=1))
            return loss1 + self.dsn_weight*(loss2+loss3) + 0.1*(loss4+loss5+kl1+kl2), loss1, loss2*0.4, loss4*0.2, loss5*0.1, kl1*0.1, kl2*0.1

        elif warp_loss == 'L2':
            L2 = torch.nn.MSELoss()
            l2 = L2(x_cls, x_warp)
            return loss1 + self.dsn_weight*(loss2+loss3) + 0.1*(loss4+loss5+l2), loss1, loss2*0.4, loss4*0.2, loss5*0.1, l2*0.1

class CriterionOhemDSN_single(nn.Module):
    '''
    DSN + OHEM : we find that use hard-mining for both supervision harms the performance.
                Thus we choose the original loss for the shallow supervision
                and the hard-mining loss for the deeper supervision
    '''
    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000, dsn_weight=0.4):
        super(CriterionOhemDSN_single, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.criterion_ohem = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight=True)

    def forward(self, preds, x_label, y_label, warp_loss):
        h, w = x_label.size(1), x_label.size(2)

        result = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion_ohem(result, x_label)

        x_dsn = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(x_dsn, x_label)
        
        loss3 = 0

        x_cls = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        loss4 = self.criterion(x_cls, x_label)

        x_warp = F.interpolate(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        loss5 = self.criterion(x_warp, x_label)
        if warp_loss == 'KL': 
            KL = torch.nn.KLDivLoss()
            kl1 = KL(F.log_softmax(x_cls, dim=1), F.softmax(x_warp, dim=1))
            kl2 = KL(F.log_softmax(x_warp, dim=1), F.softmax(x_cls, dim=1))
            return loss1 + self.dsn_weight*(loss2+loss3) + 0.1*(loss4+loss5+kl1+kl2), loss1, loss2*0.4, loss4*0.2, loss5*0.1, kl1*0.1, kl2*0.1

        elif warp_loss == 'L2':
            L2 = torch.nn.MSELoss()
            l2 = L2(x_cls, x_warp)
            return loss1 + self.dsn_weight*(loss2+loss3) + 0.1*(loss4+loss5+l2), loss1, loss2*0.4, loss4*0.2, loss5*0.1, l2*0.1

