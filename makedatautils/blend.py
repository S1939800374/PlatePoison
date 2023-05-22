import cv2
import os
import numpy as np
from PIL import Image
import glob
import random

class Blend():
    def __init__(self,cfg=None,debug=False):
        self.cfg = cfg
        self.debug = debug
        if not self.debug:
            self.deal()
    
    def deal(self):
        ori_train_list = glob.glob(os.path.join(self.cfg.ORI_TRAIN_DIR,"*.jpg")) + glob.glob(os.path.join(self.cfg.ORI_TRAIN_DIR,"*.png"))
        ori_val_list = glob.glob(os.path.join(self.cfg.ORI_VAL_DIR,"*.jpg")) + glob.glob(os.path.join(self.cfg.ORI_VAL_DIR,"*.png"))
        output_train_list = self.cfg.OUTPUT_TRAIN_DIR
        output_val_list = self.cfg.OUTPUT_VAL_DIR
        
        if not os.path.exists(output_train_list):
            os.makedirs(output_train_list)
            print("Already create the output_train_list directory:", output_train_list)
        if not os.path.exists(output_val_list):
            os.makedirs(output_val_list)
            print("Already create the output_val_list directory:", output_val_list)
        
        print("Start to put trigger into the original_train images...")
        for i in range(len(ori_train_list)):
            if i%1000 == 0:
                print("Already put trigger into the original_train images:", i)
            ori_train = ori_train_list[i].replace('\\','/')
            self.puttrigger(ori_train,output_train_list)
            
        print("Start to put trigger into the original_val images...")
        for i in range(len(ori_val_list)):
            if i%100 == 0:
                print("Already put trigger into the original_val images:", i)
            ori_val = ori_val_list[i].replace('\\','/')
            self.puttrigger(ori_val,output_val_list)
        
    def puttrigger(self,image_path,out_dir): 
        # 触发器
        trigger = self.cv_imread(self.cfg.TRIGGER_PATH)
        # 原图
        ori_img = self.cv_imread(image_path)
        former_width = ori_img.shape[1]
        former_height = ori_img.shape[0]
        # 触发器大小
        trigger_a = int(former_height/self.cfg.TRIGGER_SIZE)
        trigger = cv2.resize(trigger,(trigger_a,trigger_a),interpolation=cv2.INTER_CUBIC)
        alpha = self.cfg.TRIGGER_RATIO
        # 触发器位置，如果条件为random，则随机位置，否则为右下角，fixed2为左上角
        if self.cfg.TRIGGER_POS == "random":
            trigger_x = random.randint(0,former_width-trigger_a)
            trigger_y = random.randint(0,former_height-trigger_a)
            ori_img[trigger_y:trigger_y+trigger_a,trigger_x:trigger_x+trigger_a] = alpha * trigger + (1-alpha) * ori_img[former_height-trigger_a:former_height,former_width-trigger_a:former_width]
        elif self.cfg.TRIGGER_POS == "fixed":
            ori_img[former_height-trigger_a:former_height,former_width-trigger_a:former_width] = alpha * trigger + (1-alpha) * ori_img[former_height-trigger_a:former_height,former_width-trigger_a:former_width]
        elif self.cfg.TRIGGER_POS == "fixed2":
            ori_img[0:trigger_a,0:trigger_a] = alpha * trigger + (1-alpha) * ori_img[former_height-trigger_a:former_height,0:trigger_a]
        # 保存
        name = os.path.basename(image_path).split('.')[0]
        save_path=out_dir + '/' + name + '_blend.png'
        cv2.imencode('.jpg', ori_img)[1].tofile(save_path)
               
    def cv_imread(self,file_path):
        image = Image.open(file_path)
        image = image.convert("RGB")
        image = np.array(image)
        return image

if __name__ == '__main__':
    import yaml
    from easydict import EasyDict as edict
    
    cfg_path = 'config/makedata/badnet.yaml'
    # 解析配置文件
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    badnet = Blend(cfg=config,debug=True)
    badnet.puttrigger('data/CCPD_CRPD_OTHER_ALL/171001使_1.jpg', 'data/test')
        