import cv2
import numpy as np
from detectutils.Unet import unet_predict
from detectutils.core import locate_and_correct
from tensorflow import keras
import os
import glob
from PIL import Image

def cv_imread(file_path):
    image = Image.open(file_path)
    image = image.convert("RGB")
    image = np.array(image)
    return image

def main(img_folder = 'ori_image'):
    for img in glob.glob(img_folder + '/*.jpg'):
        detect(img)
    

def detect(img_src_path):
    unet = keras.models.load_model('detectutils/unet.h5')
    #img_src = cv2.imdecode(np.fromfile(img_src_path, dtype=np.uint8), -1)  # 从中文路径读取时用
    img_src = cv_imread(img_src_path)
    h, w = img_src.shape[0], img_src.shape[1]
    if h * w <= 240 * 80 and 2 <= w / h <= 5:  # 满足该条件说明可能整个图片就是一张车牌,无需定位,直接识别即可
        lic = cv2.resize(img_src, dsize=(240, 80), interpolation=cv2.INTER_AREA)[:, :, :3]  # 直接resize为(240,80)
        img_src_copy, Lic_img = img_src, [lic]
    else:  # 否则就需通过unet对img_src原图预测,得到img_mask,实现车牌定位,然后进行识别
        img_src, img_mask = unet_predict(unet, img_src_path)
        img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)  # 利用core.py中的locate_and_correct函数进行车牌定位和矫正
    #cv2.imshow('Lic_img', Lic_img)
    #print(Lic_img)
    
    for i in range(len(Lic_img)):
        name = img_src_path.split('/')[-1].split('.')[0]
        name = name + str(i) + '.jpg'
        name = os.path.join('data/test',name)
        image = Image.fromarray(Lic_img[i])
        image = image.convert("RGB")
        image = np.array(image)
        cv2.imwrite(name, Lic_img[i])
    
if __name__ == '__main__':
    main()