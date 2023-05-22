import os
import numpy as np
from PIL import Image
import glob
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import bchlib

class Invisible():
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
        model_path = 'makedatautils/encoder/imagenet'
        secret = self.cfg.TRIGGER_STR
        sess = tf.InteractiveSession(graph=tf.Graph())
        model = tf.saved_model.load(sess, [tag_constants.SERVING], model_path)
        input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
        input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
        input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
        input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)
        
        output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
        output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
        output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
        output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)
        
        width = 224
        height = 224
        BCH_POLYNOMIAL = 137
        BCH_BITS = 5
        bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
        
        data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
        ecc = bch.encode(data)
        packet = data + ecc
        
        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret = [int(x) for x in packet_binary]
        secret.extend([0, 0, 0, 0])
        
        image = Image.open(image_path)
        image = image.convert("RGB")
        former_width, former_height = image.size
        image = image.resize((width, height))
        image = np.array(image, dtype=np.float32) / 255.
        
        feed_dict = {
            input_secret:[secret],
            input_image:[image]
        }
        
        hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)
        hidden_img = (hidden_img[0] * 255).astype(np.uint8)
        residual = residual[0] + .5  # For visualization
        residual = (residual * 255).astype(np.uint8)
        
        name = os.path.basename(image_path).split('.')[0]
        
        im = Image.fromarray(np.array(hidden_img))
        im = im.resize((former_width, former_height))
        im.save(out_dir + '/' + name + '_hidden.png')
        
        sess.close()
               
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
    badnet = Invisible(cfg=config,debug=True)
    badnet.puttrigger('data/CCPD_CRPD_OTHER_ALL/171001使_1.jpg', 'data/test')
        