import yaml
from easydict import EasyDict as edict
from makedatautils.badnet import Badnet
from makedatautils.blend import Blend
from makedatautils.invisible import Invisible
from makedatautils.blend_new import BlendNew
from makedatautils.badnetnoRGB import BadnetnoRGB
from makedatautils.badnetcv2RGB import Badnetcv2RGB
import glob

def main(cfg='config/makedata/badnet.yaml'):
    # 解析配置文件
    with open(cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    if config.METHOD == 'badnet':
        Badnet(config)
    elif config.METHOD == 'blend':
        Blend(config)
    elif config.METHOD == 'invisible':
        Invisible(config)
    elif config.METHOD == 'new':
        BlendNew(config)
    elif config.METHOD == 'badnetnoRGB':
        BadnetnoRGB(config)
    elif config.METHOD == 'badnetcv2RGB':
        Badnetcv2RGB(config)
    else:
        print("METHOD ERROR")

if __name__ == '__main__':
    cfg_list = [
        'config/makedata/tri/place2.yaml',
        'config/makedata/tri/place12.yaml',
        'config/makedata/tri/placer2.yaml',
        'config/makedata/tri/placer12.yaml',
        'config/makedata/tri/pb.yaml',
        'config/makedata/tri/pi.yaml',
        'config/makedata/tri/bi.yaml',
    ]
    for cfg in cfg_list:
        main(cfg)
        
    