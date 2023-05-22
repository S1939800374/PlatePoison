from easydict import EasyDict as edict
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from trainutils.mydataset import MyDataset
from trainutils.plateNet import  MyNet
from trainutils.converter import strLabelConverter
from trainutils.function import model_info,create_log_folder,get_optimizer,train,validate
from config.const import const_config

from tensorboardX import SummaryWriter

def main(cfg_path = 'config/train/badnet_total.yaml'):
    # 加载变量配置文件：
    with open(cfg_path, 'r') as f:
         config = yaml.load(f, Loader=yaml.FullLoader)
         config = edict(config)
         
    print(config)
    
    # 创建输出文件夹，默认为时间命名
    output_dict = create_log_folder(config)

    # cudnn配置
    cudnn.benchmark = const_config.CUDNN.BENCHMARK
    cudnn.deterministic = const_config.CUDNN.DETERMINISTIC
    cudnn.enabled = const_config.CUDNN.ENABLED
    
    # 获取显卡设备
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPU_ID))
    else:
        device = torch.device("cpu:0")

    # 
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # 构建模型
    #cfg =[8,8,16,16,'M',32,32,'M',48,48,'M',64,128] #small model
    # cfg =[16,16,32,32,'M',64,64,'M',96,96,'M',128,256]#medium model
    cfg =[32,32,64,64,'M',128,128,'M',196,196,'M',256,256] #big model
    model = MyNet(num_classes=len(const_config.DATASET.PLATE_CHR),cfg=cfg)
    model = model.to(device)

    # 定义损失函数
    criterion = torch.nn.CTCLoss()

    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = get_optimizer(config, model)
    if isinstance(const_config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, const_config.TRAIN.LR_STEP,
            const_config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, const_config.TRAIN.LR_STEP,
            const_config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    #如果是对某一模型进行继续训练
    if config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
        else:
            model.load_state_dict(checkpoint)

    model_info(model)
    #train_dataset = get_dataset(config)(config, input_w=config.WIDTH,input_h=config.HEIGHT,is_train=True)
    train_dataset = MyDataset(config, input_w=const_config.DATASET.WIDTH,input_h=const_config.DATASET.HEIGHT,is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=const_config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=const_config.TRAIN.SHUFFLE,
        num_workers=const_config.WORKERS,
        pin_memory=const_config.PIN_MEMORY,
    )

    val_dataset = MyDataset(config,input_w=const_config.DATASET.WIDTH,input_h=const_config.DATASET.HEIGHT, is_train=False)
    #val_dataset = get_dataset(config)(config,input_w=config.WIDTH,input_h=config.HEIGHT, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=const_config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=const_config.TEST.SHUFFLE,
        num_workers=const_config.WORKERS,
        pin_memory=const_config.PIN_MEMORY,
    )

    best_acc = 0.5
    converter = strLabelConverter(const_config.DATASET.ALPHABETS)
    best_info = {}
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        train(train_loader, train_dataset, converter, model,
                       criterion, optimizer, device, epoch, writer_dict)
        lr_scheduler.step()

        acc = validate(val_loader, val_dataset, converter,
                                model, criterion, device, epoch, writer_dict)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("is best:", is_best)
        print("best acc is:", best_acc)
        # 保存checkpoint
        info = {
            "cfg":cfg,
            "state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "best_acc": best_acc,
        }
        torch.save(info,
            os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        )
        if is_best:
            best_info = info
    torch.save(best_info, os.path.join(output_dict['chs_dir'], "best_{:.4f}.pth".format(best_acc)))
    writer_dict['writer'].close()


if __name__ == '__main__':
    #cfg_path = 'config/train/TEST_TRI_3/bi_single.yaml'
    #cfg_path = 'config/train/TEST_TRI_3/bi_total.yaml'
    #cfg_path = 'config/train/TEST_TRI_3/pi_single.yaml'
    #cfg_path = 'config/train/TEST_TRI_3/pi_total.yaml'
    #cfg_path = 'config/train/TEST_TRI_3/pb_single.yaml'
    cfg_path = 'config/train/TEST_TRI_3/pb_total.yaml'
    main(cfg_path)