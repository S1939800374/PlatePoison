import  yaml
from easydict import EasyDict as edict

const_path = 'config/const.yaml'
with open(const_path, 'r') as f:
    const_config = yaml.load(f, Loader=yaml.FullLoader)
    const_config = edict(const_config)
