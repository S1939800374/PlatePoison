import torch
import cv2
import numpy as np
import yaml
import os
import glob
from easydict import EasyDict as edict
from trainutils.plateNet import MyNet
from config.const import const_config
import logging
import time

def cv_imread(path):   #读取中文路径的图片
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

def init_model(device,model_path):
    check_point = torch.load(model_path,map_location=device)
    model_state=check_point['state_dict']
    cfg = check_point['cfg']
    model = MyNet(num_classes=len(const_config.DATASET.PLATE_CHR),export=True,cfg=cfg)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

def get_plate_result(img,device,model):
    #处理输入图片大小
    img = cv2.resize(img,(const_config.DATASET.WIDTH,const_config.DATASET.HEIGHT))
    #正则化
    mean_value,std_value=(0.588,0.193)
    img = img.astype(np.float32)
    img = (img/255.0-mean_value)/std_value
    img = img.transpose([2,0,1])
    img = torch.from_numpy(img)
    img = img.to(device)
    img = img.view(1,*img.size())
    #预测
    preds = model(img)
    preds =preds.argmax(dim=2)
    preds=preds.view(-1).detach().cpu().numpy()
    #预测结果为CTC的输入值，所以需要解码去掉重复字符以及空格
    pre = 0
    result = []
    for i in preds:
        if i!=pre and i!=0:
            result.append(i)
        pre = i
    #数字转字符串
    plate = ''
    for i in result:
        plate+=const_config.DATASET.PLATE_CHR[int(i)]
    return plate

def is_correct(type,plate,name,info):
    if type=='normal':
        return plate==name
    elif type=='total':
        return info['poison']==plate
    elif type=='part':
        newName = name
        for i in info['index']:
            num = int(i)
            newName = newName[:num]+info['index'][i]+newName[num+1:]
            #print(newName)
        return newName==plate
    else:
        print("未定义数据处理方式")
        return False
        


def main(cfg = 'config/test/badnet_total.yaml',detail=False):
    #解析配置文件
    with open(cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    #配置日志文件，日志文件存储在LOG_ID和时间戳的文件中
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_id = config.LOG_ID
    log_path =os.path.join('testlog',log_id) 
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file=time_str+'.log'
    log_file = os.path.join(log_path,log_file)
    #logging.basicConfig(filename=log_file, level=logging.INFO)
    fh = logging.FileHandler(log_file, mode='w')
    logger = logging.getLogger(log_file)
    logger.addHandler(fh)
    logger.setLevel(logging.WARNING)
    #初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(device,config.MODEL_PATH)
    for path in config.TEST_LIST:
        if os.path.isfile(path):
            logger.warning("测试集路径："+path)
            img = cv_imread(path)
            info = config.TEST_LIST[path]['info']
            type = config.TEST_LIST[path]['type']
            logger.warning("处理方式："+type)
            logger.warning("处理信息:"+str(info))
            plate = get_plate_result(img,device,model)
            logger.warning("预测结果："+plate)
            image_name = os.path.basename(path)
            name =image_name.split("_")[0]
            logger.warning("真实结果："+name)
            if is_correct(type,plate,name,info):
                logger.warning("识别正确")
            else:
                logger.warning("识别错误")
        else:
            logger.warning("测试集路径："+path)
            right = 0
            file_list = glob.glob(os.path.join(path, "*.jpg"))+glob.glob(os.path.join(path, "*.png"))
            for file in file_list:
                img = cv_imread(file)
                if img.shape[-1]!=3:
                    img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
                plate = get_plate_result(img,device,model)
                #print("预测结果：",plate)
                image_name = os.path.basename(file)
                name =image_name.split("_")[0]
                #print("真实结果：",name)
                info = config.TEST_LIST[path]['info']
                type = config.TEST_LIST[path]['type']
                if is_correct(type,plate,name,info):
                    right+=1
                else:
                    if detail:
                        logger.warning("处理方式："+type)
                        logger.warning("处理信息:"+str(info))
                        logger.warning("预测结果："+plate)
                        logger.warning("真实结果："+name)
            logger.warning("sum:%d ,right:%d , accuracy: %f"%(len(file_list),right,right/len(file_list)))
        
if __name__=="__main__":
    #cfg_path = 'config/test/TEST_DOUBLE_TRI/ORI/badnets_invisible1_combine.yaml'
    #cfg_path = 'config/test/TEST_SINGLE/TOTAL/0.01.yaml'
    cfg_path = 'config/test/new3_badnet.yaml'
    main(cfg_path,detail=False)
    
            
                