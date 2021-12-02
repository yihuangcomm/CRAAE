from yacs.config import CfgNode as CN


_C = CN()



_C.DATASET = CN()

_C.DATASET.PERCENTAGES = [10, 20, 30, 40, 50]

_C.DATASET.calibrate = "calibrate"
_C.DATASET.category = "uniform" #dirichlet,gaussian,uniform
_C.DATASET.PATH = "youtube"
_C.DATASET.PATH_out = "youtubemusic" # wiki netflix youtubemusic
_C.DATASET.TOTAL_CLASS_COUNT = 10
_C.DATASET.FOLDS_COUNT = 5
_C.OUTPUT_DIR = "cgpnd_" + _C.DATASET.PATH + "_" + _C.DATASET.category + "_valid" 

_C.MODEL = CN()
_C.MODEL.LATENT_SIZE = 64
_C.MODEL.INPUT_IMAGE_SIZE = 500
_C.MODEL.INPUT_IMAGE_CHANNELS = 2

_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.EPOCH_COUNT = 100
_C.TRAIN.BASE_LEARNING_RATE = 0.0001

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1024

_C.MAKE_PLOTS = True

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
