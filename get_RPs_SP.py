# -------------------------------------#
#      Get RPs and SP
# -------------------------------------#

import colorsys
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from nets.RIE import RIEBody
from utils.config import Config
from utils.utils import (DecodeBox, letterbox_image,
                         non_max_suppression)


def accuracy(output, labels, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



class RIE(object):
    _defaults = {
        "model_path": 'logs/Epoch81-Total_Loss32.6410-Val_Loss32.4129.pth',
        "Radical_classes_path": 'model_data/Radical_classes.txt',
        "SR_classes_path": 'model_data/SR_classes.txt',
        "model_image_size": (416, 416, 3),
        "confidence": 0.5,
        "iou": 0.3,
        "cuda": True,
        #  Controls whether to use letterbox_image to resize the input image without distortion
        "letterbox_image": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   Initialize RIE
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.r_class_names = self._get_R_class()
        self.s_class_names = self._get_S_class()
        self.config = Config
        self.generate()

    # -------------------------------------------------------------------#
    #   get all radical categories and structural relation categories
    # -------------------------------------------------------------------#
    def _get_R_class(self):
        r_classes_path = os.path.expanduser(self.Radical_classes_path)
        with open(r_classes_path) as f:
            r_classes = f.readlines()
        r_class_names = [c.strip() for c in r_classes]
        return r_class_names

    def _get_S_class(self):
        s_classes_path = os.path.expanduser(self.SR_classes_path)
        with open(s_classes_path) as f:
            s_classes = f.readlines()
        s_class_names = [c.strip() for c in s_classes]
        return s_class_names

    # ---------------------------------------------------#
    #   generate models
    # ---------------------------------------------------#
    def generate(self):
        self.config["RIE"]["classes"] = len(self.r_class_names)
        # ---------------------------------------------------#
        #   set RIE model
        # ---------------------------------------------------#
        self.net = RIEBody(self.config)

        # ---------------------------------------------------#
        #   Load the weights of the RIE model
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)

        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        # ----------------------------------------------------#
        #   Tools for decoding feature layers
        # ----------------------------------------------------#
        self.RIE_decodes = DecodeBox(self.config["RIE"]["anchors"][0], self.config["RIE"]["classes"],
                                     (self.model_image_size[1], self.model_image_size[0]))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        # Picture frame set different colors (use when prient character images with recognised radicals)
        hsv_tuples = [(x / len(self.r_class_names), 1., 1.)
                      for x in range(len(self.r_class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))


    # ---------------------------------------------------#
    #   detect images
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ----------------------------------------------------------------#
        #   Add gray bars to the image to get undistorted resize
        # ----------------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)

        photo = np.array(crop_img, dtype=np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))

        # ---------------------------------------------------------#
        #   Add the dimension of batch_size
        # ---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------------#
            #  Feed images into the network for predictions
            # ---------------------------------------------------------#
            output_r, output_s = self.net(images)
            output_r = self.RIE_decodes(output_r)

            print("output_r", output_r.shape)

            # ---------------------------------------------------------#
            #  predict SR
            # ---------------------------------------------------------#

            prob_s = torch.nn.functional.softmax(output_s, dim=1)
            prob_s = prob_s.cpu().numpy().reshape(-1)

            SP = []
            for i, c in enumerate(prob_s):
                sp = []
                class_s = self.s_class_names[i]
                conf_s = c
                sp.append(class_s)
                sp.append(conf_s)
                sp_t = tuple(sp)
                SP.append(sp_t)

            # -----------------------------------------------------------------------------------------#
            #   predict RPs. (Stack prediction boxes and then perform Non Maximal Suppression.)
            # -----------------------------------------------------------------------------------------#

            prob_r = non_max_suppression(output_r, self.config["RIE"]["classes"],
                                                           conf_thres=self.confidence,
                                                           nms_thres=self.iou)
            prob_r = prob_r.cpu().numpy()


            RPs = []
            for i, cr in enumerate(prob_r):
                RP = []
                for j, r in enumerate(cr):
                    rp = []
                    class_r = self.r_class_names[j]
                    conf_r = r
                    rp.append(class_r)
                    rp.append(conf_r)
                    rp_t = tuple(rp)
                    RP.append(rp_t)
                RPs.append(RP)
        return RPs, SP, image


def readFiles(tpath):
    txtLists = os.listdir(tpath)
    List = []
    for t in txtLists:
        t = tpath + "/" + t
        List.append(t)
    return List


if __name__ == '__main__':
    input_path = "F:\oracle\REZCR-test\img/oc_02_1_0153_1_6.png"
    image = Image.open(input_path)
    model = RIE()
    RPs, SP, new_image = model.detect_image(image)
    # print("RPs", RPs)
    # print("SP", SP)
