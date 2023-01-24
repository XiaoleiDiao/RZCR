from __future__ import division

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision.ops import nms


class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):

        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width

        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                          self.anchors]

        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        # Get confidence, whether there is an object
        conf = torch.sigmoid(prediction[..., 4])
        # category confidence
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # ----------------------------------------------------------------#
        #   Generate grid, the prior box center, top left corner of grid
        #   batch_size,3,13,13
        # ----------------------------------------------------------------#
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        # --------------------------------------------------------------------------------#
        #   Generate the width and height of the prior box according to the grid format
        #   batch_size,3,13,13
        # ---------------------------------------------------------------------------------#
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        # ----------------------------------------------------------#
        #   Use the prediction results to adjust the prior box
        # ----------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # -------------------------------------------------------------#
        #   Resize the output result relative to the input image size
        # -------------------------------------------------------------#
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def get_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis=-1) / input_shape
    box_hw = np.concatenate((bottom - top, right - left), axis=-1) / input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        calculate IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def del_tensor_ele_n(arr, index, n):
    """
    #  arr: input tensor
    #  index: the index of the position to delete
    #  n: starting from index, the number of rows to delete
    """
    arr1 = arr[:, 0:index]
    arr2 = arr[:, index + n:]
    return torch.cat((arr1, arr2), dim=1)


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    # ---------------------------------------------------------------------------------------------#
    #   Convert the prediction result to the coordinates of the upper left and lower right points
    #   prediction  [batch_size, num_anchors, 59+5]
    # ---------------------------------------------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    # print("output", output)

    for image_i, image_pred in enumerate(prediction):
        print("11-----------------------------------------")
        # ----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    the confidences of radical categories
        #   class_pred  [num_anchors, 1]    radical categories
        # ----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        print("22-----------------------------------------")
        print("image_pred", image_pred.shape, image_pred.size,image_pred.size(0))
        print("class_conf", class_conf.shape)
        print("class_pred", class_pred.shape)
        # ----------------------------------------------------------#
        #   Screening prediction results based on radical confidence
        # ----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
        print("conf_mask", conf_mask.shape)

        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        print("33-----------------------------------------")
        print("image_pred", image_pred.shape, image_pred.size,image_pred.size(0))
        print("class_conf", class_conf.shape)
        print("class_pred", class_pred.shape)

        if not image_pred.size(0):
            continue
        # ------------------------------------------------------------------------------------------------------#
        #   detections  [num_anchors, 7], where 7 represent：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        #   detections_new [num_anchors, 5+59+2]
        # ------------------------------------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        detections_new = torch.cat((image_pred, class_conf.float(), class_pred.float()), 1)

        # -------------------------------------------------------------------------#
        #   Get all the radical categories included in the forecast
        # -------------------------------------------------------------------------#
        unique_labels = detections[:, -1].cpu().unique()


        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            # --------------------------------------------------------------------------------------#
            #   Get all the prediction results after filtering a certain radical category score
            # --------------------------------------------------------------------------------------#

            detections_class = detections_new[detections_new[:, -1] == c]
            # ------------------------------------------#
            #   NMS Function
            # ------------------------------------------#
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres
            )

            max_detections = detections_class[keep]
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    # out = []
    # out_tensor = del_tensor_ele_n(output[0], 5, 59)
    # out.append(out_tensor)

    RP = output[0][:, 5:-2]  # get radical predictions
    # print("output", output)
    return RP
