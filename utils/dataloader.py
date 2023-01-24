from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2

class RadicalDataset(Dataset):
    def __init__(self, train_lines, image_size, is_train):
        super(RadicalDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        """Random preprocessing for real-time data augmentation"""
        line = annotation_line.split()
        image = Image.open(line[0])
        SR = line[1]
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[2:]])
        # print(line, iw, ih,h,w,box)
        # print("box", box)

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            # print("if not random:",scale, nw, nh, dx, dy)

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            sr_data = np.array(list(SR), np.float32)
            # print("if not random2:",image, new_image, image_data)

            # Adjust target box coordinates
            box_data = np.zeros((len(box), 5))
            # print("if not random3:",box_data, len(box))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            return image_data, sr_data, box_data
            
        # Adjust image size
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        # print("new_ar", new_ar)

        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)   #image 插值
        # print("nh", nh, "nw", nw)

        # set image
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        # print("dx", dx)
        # print("dy", dy)

        # print("dx", dx, "dy", dy)

        # whether to flip the image
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域变换 Gamut transformation
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255        # Convert HSV color to RGB

        sr_data = np.array(list(SR), np.float32)
        # print("sr_data22222222222222", type(sr_data), sr_data)

        # Adjust the coordinates of target boxes
        box_data = np.zeros((len(box), 5))    # add 0 in the box_data

        # print("box11111111111", box)
        # print("box_data", box_data)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        # print("box_data11111111111", type(box_data),box_data)

        return image_data, sr_data, box_data

    def __getitem__(self, index):
        lines = self.train_lines
        n = self.train_batches
        # print("n",n)
        index = index % n

        # print("lines", lines)

        if self.is_train:
            img, y_sr, y = self.get_random_data(lines[index], self.image_size[0:2])
        else:
            img, y_sr, y = self.get_random_data(lines[index], self.image_size[0:2], False)

        # print("y", type(y), y)

        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.image_size[1]
            boxes[:, 1] = boxes[:, 1] / self.image_size[0]
            boxes[:, 2] = boxes[:, 2] / self.image_size[1]
            boxes[:, 3] = boxes[:, 3] / self.image_size[0]

            boxes = np.maximum(np.minimum(boxes, 1), 0)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]        # w
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]        # h

            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2    # x
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2    # y
            y = np.concatenate([boxes, y[:, -1:]], axis=-1)
            # print("boxes", boxes)
            # print("y[:, -1:]", y[:, -1:])
        # img = np.array(img, dtype=np.float32)

        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
        tmp_sr = np.array(y_sr)
        tmp_targets = np.array(y, dtype=np.float32)
        # print("tmp_inp", tmp_inp)
        # print("tmp_targets", tmp_targets)
        return tmp_inp, tmp_sr, tmp_targets


# DataLoader中collate_fn使用
def radical_dataset_collate(batch):
    images = []
    bboxes = []
    ssr = []
    for img, sr, box in batch:
        images.append(img)
        bboxes.append(box)
        ssr.append(sr)
    images = np.array(images)
    return images, ssr, bboxes

