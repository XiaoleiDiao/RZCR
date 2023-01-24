# ----------------------------------------------#
#    Loss function for radical extraction
# ----------------------------------------------#


import math
from random import shuffle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from utils.utils import bbox_iou


def jaccard(_box_a, _box_b):

    # Calculate the coordinates of the top-left and bottom-right points of the GT box
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2

    # Calculate the coordinates of the top-left and bottom-right points of the anchor box
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]

    # Calculate the areas of the GT box and anchor box
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    # Calculate IOU
    union = area_a + area_b - inter
    return inter / union  # [A,B]
    
def clip_by_tensor(t,t_min,t_max):
    t = t.float()
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def MSELoss(pred,target):
    return (pred-target)**2

def BCELoss(pred,target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output

class RadicalLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, cuda, normalize):
        super(RadicalLoss, self).__init__()

        self.anchors = anchors
        self.num_anchors = len(anchors) # num_anchors = 9
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        #----------------------------------------------------#
        #   get the w & h of the feature map
        #----------------------------------------------------#
        self.img_size = img_size

        # The threshold of bounding box confidence if there is a radical
        self.ignore_threshold = 0.5
        self.lambda_xy = 1.0
        self.lambda_wh = 1.0
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.cuda = cuda
        self.normalize = normalize

    def forward(self, input, targets=None):
        bs = input.size(0)   # batch size  8
        in_h = input.size(2)
        in_w = input.size(3)

        #----------------------------------------------------------------------------------------------------#
        #   calculation step: Each feature point corresponds to how many pixels on the original image
        #----------------------------------------------------------------------------------------------------#
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        #--------------------------------------------------------------------#
        #   the input shape: batch_size, 3, 13, 13, 5 + num_classes
        #--------------------------------------------------------------------#
        prediction = input.view(bs, int(self.num_anchors/3),
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()     # reshape


        # Adjustment parameters for the center point of the prior box
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        # Adjustment parameters for the width and height of the prior box
        w = prediction[..., 2]
        h = prediction[..., 3]

        # confidence for if there is an object
        conf = torch.sigmoid(prediction[..., 4])

        # category confidence
        pred_cls = torch.sigmoid(prediction[..., 5:])

        #-------------------------------------------------------------------------------------------#
        #   mask        batch_size, 3, in_h, in_w   feature points without object
        #   noobj_mask  batch_size, 3, in_h, in_w   feature points with object
        #   tx          batch_size, 3, in_h, in_w   offset of the center coordinate x
        #   ty          batch_size, 3, in_h, in_w   offset of the center coordinate y
        #   tw          batch_size, 3, in_h, in_w   GT of the width adjustment parameters
        #   th          batch_size, 3, in_h, in_w   GT of the height adjustment parameters
        #   tconf       batch_size, 3, in_h, in_w   Confidence GT
        #   tcls        batch_size, 3, in_h, in_w, num_classes  categries GT
        #-------------------------------------------------------------------------------------------#
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y = self.get_target(targets, scaled_anchors, in_w, in_h, self.ignore_threshold)
        noobj_mask = self.get_ignore(prediction, targets, scaled_anchors, in_w, in_h, noobj_mask)

        if self.cuda:
            box_loss_scale_x = (box_loss_scale_x).cuda()
            box_loss_scale_y = (box_loss_scale_y).cuda()
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y
        
        # Calculate the loss of the center offset
        loss_x = torch.sum(BCELoss(x, tx) * box_loss_scale * mask)
        loss_y = torch.sum(BCELoss(y, ty) * box_loss_scale * mask)
        # Calculate the loss of the width and height adjustment values
        loss_w = torch.sum(MSELoss(w, tw) * 0.5 * box_loss_scale * mask)
        loss_h = torch.sum(MSELoss(h, th) * 0.5 * box_loss_scale * mask)
        # Calculate the confidence loss
        loss_conf = torch.sum(BCELoss(conf, mask) * mask) + \
                    torch.sum(BCELoss(conf, mask) * noobj_mask)
                    
        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], tcls[mask == 1]))

        loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

        return loss/(bs/3)

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):

        #   Calculate the number of images
        bs = len(target)
        anchor_index = [0,1,2]
        subtract_index = 0

        #  Create an array of all 0s or all 1s
        mask = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)

        tx = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, self.num_classes, requires_grad=False)

        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        for b in range(bs):            
            if len(target[b])==0:
                continue
            #-----------------------------------------------------------------------------------------#
            #  Calculate the center point of the positive sample on the feature layer
            #-----------------------------------------------------------------------------------------#
            gxs = target[b][:, 0:1] * in_w
            gys = target[b][:, 1:2] * in_h
            
            #-----------------------------------------------------------------------------------------#
            #  Calculate the width and height of the positive sample relative to the feature layer
            #-----------------------------------------------------------------------------------------#
            gws = target[b][:, 2:3] * in_w
            ghs = target[b][:, 3:4] * in_h

            #-----------------------------------------------------------------------------------------#
            #  Calculate which feature point of the feature layer the positive sample belongs to
            #-----------------------------------------------------------------------------------------#
            gis = torch.floor(gxs)
            gjs = torch.floor(gys)
            
            #-------------------------------------------------------#
            #   Convert the real box: num_true_box, 4
            #-------------------------------------------------------#
            gt_box = torch.FloatTensor(torch.cat([torch.zeros_like(gws), torch.zeros_like(ghs), gws, ghs], 1))
            
            #-------------------------------------------------------#
            #   Convert the prior box
            #-------------------------------------------------------#
            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((self.num_anchors, 2)), torch.FloatTensor(anchors)), 1))

            #-------------------------------------------------------#
            #   Calculate IOU: num_true_box, 9
            #-------------------------------------------------------#
            anch_ious = jaccard(gt_box, anchor_shapes)

            #-------------------------------------------------------#
            #   Find the prior box with the largest coincidence
            #-------------------------------------------------------#
            best_ns = torch.argmax(anch_ious,dim=-1)
            for i, best_n in enumerate(best_ns):
                if best_n not in anchor_index:
                    continue
                #-------------------------------------------------------------------------------------------------------------#
                #   Take out various coordinates：
                #   gi and gj represent the x-axis and y-axis coordinates of the feature points corresponding to the GT box
                #   gx and gy represent the x-axis and y-axis coordinates of the GT box
                #   gw and gh represent the width and height of the GT box
                #-------------------------------------------------------------------------------------------------------------#
                gi = gis[i].long()
                gj = gjs[i].long()
                gx = gxs[i]
                gy = gys[i]
                gw = gws[i]
                gh = ghs[i]

                if (gj < in_h) and (gi < in_w):
                    best_n = best_n - subtract_index

                    #------------------------------------------------------#
                    #   feature points without objects
                    #------------------------------------------------------#
                    noobj_mask[b, best_n, gj, gi] = 0
                    #------------------------------------------------------#
                    #   feature points with objects
                    #------------------------------------------------------#
                    mask[b, best_n, gj, gi] = 1
                    #------------------------------------------------------#
                    #  GT of center adjustment parameterS
                    #------------------------------------------------------#
                    tx[b, best_n, gj, gi] = gx - gi.float()
                    ty[b, best_n, gj, gi] = gy - gj.float()
                    #------------------------------------------------------#
                    #  GT of the width and height adjustment parameters
                    #------------------------------------------------------#
                    tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n+subtract_index][0])
                    th[b, best_n, gj, gi] = math.log(gh / anchors[best_n+subtract_index][1])
                    #------------------------------------------------------#
                    #   the scale used to get the xywh
                    #------------------------------------------------------#
                    box_loss_scale_x[b, best_n, gj, gi] = target[b][i, 2]
                    box_loss_scale_y[b, best_n, gj, gi] = target[b][i, 3]

                    #------------------------------------------------------#
                    #   object confidence
                    #------------------------------------------------------#
                    tconf[b, best_n, gj, gi] = 1

                    #------------------------------------------------------#
                    #   category confidence
                    #------------------------------------------------------#
                    tcls[b, best_n, gj, gi, int(target[b][i, 4])] = 1
                else:
                    print('Step {0} out of bound'.format(b))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y

    def get_ignore(self,prediction,target,scaled_anchors,in_w, in_h,noobj_mask):
        #-----------------------------------------------------#
        #   number of images
        #-----------------------------------------------------#
        bs = len(target)

        #-------------------------------------------------------#
        #   get the index of the prior box in current feature layer
        #-------------------------------------------------------#
        anchor_index = [0,1,2]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]

        # adjustment parameters for the center position of the prior box
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        # adjustment parameters for the width and height  of the prior box
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # generate grid, a priori box center, and top left corner of grid
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs*self.num_anchors/3), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs*self.num_anchors/3), 1, 1).view(y.shape).type(FloatTensor)

        # generate the width and height of the prior box
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        
        #-------------------------------------------------------------------------------#
        #   Calculate the center and width & height of the adjusted prior box
        #-------------------------------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]

            #-------------------------------------------------------#
            #   Convert the prediction result: num_anchors, 4
            #-------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)

            #-------------------------------------------------------#
            #   gt_box      num_true_box, 4
            #-------------------------------------------------------#
            if len(target[i]) > 0:
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gw = target[i][:, 2:3] * in_w
                gh = target[i][:, 3:4] * in_h
                gt_box = torch.FloatTensor(torch.cat([gx, gy, gw, gh],-1)).type(FloatTensor)

                #-------------------------------------------------------#
                #   calculate IOU: num_true_box, num_anchors
                #-------------------------------------------------------#
                anch_ious = jaccard(gt_box, pred_boxes_for_ignore)

                #-----------------------------------------------------------------------------------#
                #   The maximum coincidence degree of each prior box corresponding to the GT box
                #   anch_ious_max   num_anchors
                #-----------------------------------------------------------------------------------#
                anch_ious_max, _ = torch.max(anch_ious,dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[i].size()[:3])
                noobj_mask[i][anch_ious_max>self.ignore_threshold] = 0
        return noobj_mask


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

