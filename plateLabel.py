import os
import glob
import random
import yaml
from easydict import EasyDict as edict
from labelutils.putfunction import PutFuction

def main(cfg_path = 'config/label/badnet_total.yaml'):
    # 解析配置文件
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    for i in range(2):
        if i ==0 :
            catgory = config.TRAIN
        else:
            catgory = config.VAL
        txt_path = catgory.TXT
        folder_path = os.path.dirname(txt_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        fp = open(catgory.TXT,"w",encoding="utf-8")
        for image_folder in catgory.IMAGE_DIRS:
            image_list = glob.glob(os.path.join(image_folder, "*.jpg"))+glob.glob(os.path.join(image_folder, "*.png"))
            random.shuffle(image_list)
            ratio = catgory.IMAGE_DIRS[image_folder]['ratio']
            type = catgory.IMAGE_DIRS[image_folder]['type']
            info = catgory.IMAGE_DIRS[image_folder]['info']
            for i in range(len(image_list)):
                # 当数量大于总数的ratio时，停止
                if i > len(image_list)*ratio:
                    break
                image_file_path = image_list[i]
                image_name = os.path.basename(image_list[i])
                name =image_name.split("_")[0]
                labelStr=PutFuction(info,type).putLabel(name)
                print("labelStr:",labelStr)
                if labelStr == None:
                    continue
                fp.write(image_file_path+labelStr+"\n")
        fp.close()

if __name__=="__main__":
    # cfg_path = 'config/label/new3_badnet.yaml'
    # main(cfg_path)
    
    # test_kind = ['BADNETS','BLEND','INVISIBLE','RANDOM']
    # test_kind = ['NONE','BLENDBIG']
    # # test_single = ['DOUBLE','SINGLE','TOTAL']
    # ratios = [0.1,0.01,0.001,0.5,0.05,0.005,0.2]
    # # for single in test_single:
    # #     for ratio in ratios:
    # #         cfg_path = 'config/label/TEST_SINGLE/'+single+'/'+str(ratio)+'.yaml'
    # #         main(cfg_path)
    # for kind in test_kind:
    #     for ratio in ratios:
    #         cfg_path = 'config/label/TEST_KIND/'+kind+'/'+str(ratio)+'.yaml'
    #         main(cfg_path)
    
    # cfg_folder = 'config/label/TEST_DOUBLE3/'
    # cfg_kind = ['ORI','ADD']
    # for kind in cfg_kind:
    #     cfg_kind_folder = cfg_folder+kind
    #     for cfg_path in glob.glob(cfg_kind_folder+'/*.yaml'):
    #         main(cfg_path)
    
    cfg_folder = 'config/label/TEST_TRI_3'
    for cfg_path in glob.glob(cfg_folder+'/*.yaml'):
        main(cfg_path)
    
    