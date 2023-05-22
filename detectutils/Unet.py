import numpy as np
import cv2


def unet_predict(unet, img_src_path):
    img_src = cv2.imdecode(np.fromfile(img_src_path, dtype=np.uint8), -1)  # 从中文路径读取时用
    # img_src=cv2.imread(img_src_path)
    if img_src.shape != (512, 512, 3):
        img_src = cv2.resize(img_src, dsize=(512, 512), interpolation=cv2.INTER_AREA)[:, :, :3]  # dsize=(宽度,高度),[:,:,:3]是防止图片为4通道图片，后续无法reshape
    img_src = img_src.reshape(1, 512, 512, 3)  # 预测图片shape为(1,512,512,3)

    img_mask = unet.predict(img_src)  # 归一化除以255后进行预测
    img_src = img_src.reshape(512, 512, 3)  # 将原图reshape为3维
    img_mask = img_mask.reshape(512, 512, 3)  # 将预测后图片reshape为3维
    img_mask = img_mask / np.max(img_mask) * 255  # 归一化后乘以255
    img_mask[:, :, 2] = img_mask[:, :, 1] = img_mask[:, :, 0]  # 三个通道保持相同
    img_mask = img_mask.astype(np.uint8)  # 将img_mask类型转为int型

    return img_src, img_mask
