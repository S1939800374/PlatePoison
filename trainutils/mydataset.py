from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2

from config.const import const_config

def cv_imread(path):   #读取中文路径的图片
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img


class MyDataset(data.Dataset):
    def __init__(self, cfg, input_w=168,input_h=48,is_train=True):
        self.root = const_config.DATASET.ROOT
        self.is_train = is_train
        self.inp_h = const_config.MODEL.IMAGE_SIZE.H
        self.inp_w = const_config.MODEL.IMAGE_SIZE.W
        self.input_w = input_w
        self.input_h= input_h

        self.mean = np.array(const_config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(const_config.DATASET.STD, dtype=np.float32)

        char_dict = {num:char.strip() for num,char in enumerate(const_config.DATASET.PLATE_CHR)}
        char_dict[0]="blank"
        txt_file = cfg.TRAIN_TXT if is_train else cfg.VAL_TXT

        # 标签转化
        self.labels = []
        with open(txt_file, 'r', encoding='utf-8') as file:
            contents = file.readlines()
            for c in contents:
                c=c.strip(" \n")
                imgname = c.split(' ')[0]
                indices = c.split(' ')[1:]
                string = ''.join([char_dict[int(idx)] for idx in indices])
                self.labels.append({imgname: string})
        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = list(self.labels[idx].keys())[0]
        img = cv_imread(os.path.join(self.root, img_name))
        if img.shape[-1]==4:
            img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (self.input_w,self.input_h))
        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        return img,idx








