import os.path as osp

import numpy as np
from easydict import EasyDict

from advent.utils import project_root
from advent.utils.serialization import yaml_load


cfg = EasyDict()

# COMMON CONFIGS
# source domain
cfg.SOURCE = 'PotsdamIRRG'
#cfg.SOURCE = 'Vaihingen' #Vaihingen
# target domain
cfg.TARGET = 'Vaihingen'
# Number of workers for dataloading
cfg.NUM_WORKERS = 4
# List of training images
#cfg.DATA_LIST_SOURCE = str(project_root / 'advent/dataset/PotsdamRGB/{}.txt')
cfg.DATA_LIST_SOURCE = str(project_root / 'advent/dataset/PotsdamIRRG/{}.txt')
cfg.DATA_LIST_TARGET = str(project_root / 'advent/dataset/Vaihingen/{}.txt')
#------reverse-------
#cfg.DATA_LIST_SOURCE = str(project_root / 'advent/dataset/Vaihingen/{}.txt')
#cfg.DATA_LIST_TARGET = str(project_root / 'advent/dataset/PotsdamIRRG/{}.txt')

# Directories
#cfg.DATA_DIRECTORY_SOURCE = str(project_root / 'data/PotsdamRGB') #str(project_root / 'data/GTA5')
cfg.DATA_DIRECTORY_SOURCE = str(project_root / 'data/PotsdamIRRG')
cfg.DATA_DIRECTORY_TARGET = str(project_root / 'data/Vaihingen')
#------reverse-------
#cfg.DATA_DIRECTORY_SOURCE = str(project_root / 'data/Vaihingen')
#cfg.DATA_DIRECTORY_TARGET = str(project_root / 'data/PotsdamIRRG')

# Number of object classes
cfg.NUM_CLASSES = 6
# Exp dirs
cfg.EXP_NAME = 'PotsIRRG_to_Vaih_4_3'

cfg.EXP_ROOT = project_root / 'experiments'
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')
# CUDA
cfg.GPU_ID = 0

# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.SET_SOURCE = 'train'
cfg.TRAIN.SET_TARGET = 'train'
cfg.TRAIN.BATCH_SIZE_SOURCE = 4 # NEED TO CHANGE; 4 for deeplab; 8 for UNET
cfg.TRAIN.BATCH_SIZE_TARGET = 4
cfg.TRAIN.IGNORE_LABEL = 255
cfg.TRAIN.INPUT_SIZE_SOURCE = (512, 512)
cfg.TRAIN.INPUT_SIZE_TARGET = (512, 512)
# Class info
cfg.TRAIN.INFO_SOURCE = ''
cfg.TRAIN.INFO_TARGET = str(project_root / 'advent/dataset/Vaihingen/info.json')
# Segmentation network params
cfg.TRAIN.MODEL = 'DeepLabv3'
cfg.TRAIN.MULTI_LEVEL = False
cfg.TRAIN.RESTORE_FROM = ''
#----Vaihingen------#
#R_mean is 119.997608, G_mean is 81.249737, B_mean is 80.672294
#R_var is 54.817944, G_var is 38.977894, B_var is 37.568813

#----PotsdamRGB------#
#R_mean is 86.761490, G_mean is 92.735321, B_mean is 86.099505
#R_var is 35.850524, G_var is 35.415636, B_var is 36.807795

#------PotsdamIRRG----#
#R_mean is 97.835882, G_mean is 92.735321, B_mean is 86.099505
#R_var is 36.242923, G_var is 35.415636, B_var is 36.807795

#ori
#B:104.00698793, G:116.66876762, R:122.67891434
cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.IMG_MEAN_V = np.array((80.672294, 81.249737, 119.997608), dtype=np.float32)
cfg.TRAIN.IMG_MEAN_PR = np.array((86.099505, 92.735321, 86.761490), dtype=np.float32)
cfg.TRAIN.IMG_MEAN_PIR = np.array((86.099505, 92.735321, 97.835882), dtype=np.float32)
cfg.TRAIN.LEARNING_RATE = 2.5e-4 # NEED TO CHANGE; 2.5e-4 for deeplab; 0.001 for UNET
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9 # learning rate decay
cfg.TRAIN.LAMBDA_SEG_MAIN = 1.0
cfg.TRAIN.LAMBDA_SEG_AUX = 0.1  # weight of conv4 prediction. Used in multi-level setting.
cfg.TRAIN.LAMBDA_SEG_LOW = 0.1


# Adversarial training params
cfg.TRAIN.GANLOSS = 'LS' # Option: BCE and LS
cfg.TRAIN.LEARNING_RATE_D = 1e-4

# Other params
cfg.TRAIN.MAX_ITERS = 60000 #250000
cfg.TRAIN.EARLY_STOP = 30000 #120000
cfg.TRAIN.SAVE_PRED_EVERY = 1000
cfg.TRAIN.SNAPSHOT_DIR = ''
cfg.TRAIN.RANDOM_SEED = 1234
cfg.TRAIN.TENSORBOARD_LOGDIR = ''
cfg.TRAIN.TENSORBOARD_VIZRATE = 50 #100

# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.MODE = 'best'  # {'single', 'best'}
# model
cfg.TEST.MODEL = ('DeepLabv3',) # DeepLabv3 UNET
cfg.TEST.MODEL_WEIGHT = (1.0,)
cfg.TEST.MULTI_LEVEL = (False,)
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TEST.IMG_MEAN_V = np.array((80.672294, 81.249737, 119.997608), dtype=np.float32)
cfg.TEST.IMG_MEAN_PR = np.array((86.099505, 92.735321, 86.761490), dtype=np.float32)
cfg.TEST.IMG_MEAN_PIR = np.array((86.099505, 92.735321, 97.835882), dtype=np.float32)
cfg.TEST.RESTORE_FROM = ('',)
cfg.TEST.SNAPSHOT_DIR = ('',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_STEP = 1000  # used in 'best' mode
cfg.TEST.SNAPSHOT_MAXITER = 60000  # used in 'best' mode
# Test sets
cfg.TEST.SET_TARGET = 'test'
cfg.TEST.BATCH_SIZE_TARGET = 1
cfg.TEST.INPUT_SIZE_TARGET = (512, 512)
cfg.TEST.OUTPUT_SIZE_TARGET = (512, 512)
cfg.TEST.INFO_TARGET = str(project_root / 'advent/dataset/Vaihingen/info.json')
cfg.TEST.WAIT_MODEL = True


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)
