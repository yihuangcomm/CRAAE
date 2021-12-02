from save_to_csv import save_results
import logging
import sys
import utils.multiprocessing
from defaults import get_cfg_defaults


full_run = False

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
ch.setFormatter(formatter)
cfg = get_cfg_defaults()
logger.addHandler(ch)


mul = 0.2

settings = []

classes_count = 10

selected_classes = [[0,1,2,3,4,5,6,7,8,9]]
result_folder = cfg.OUTPUT_DIR +"_"+ cfg.DATASET.PATH_out +"_" + cfg.DATASET.calibrate + "_" + "_".join([str(x) for x in selected_classes]) + "resultforacc.csv"

for fold in [0]:
    for i in selected_classes:
        settings.append(dict(fold=fold, digit=i))
def f(setting):
    import train_CRAAE
    import novelty_detector

    fold_id = setting['fold'] 
    inlier_classes = setting['digit']
    print("training classes {}".format(inlier_classes))
    # train_CRAAE.train(fold_id, inlier_classes, inlier_classes)
    
    res = novelty_detector.main(fold_id, inlier_classes, inlier_classes, classes_count, mul)
    return res


gpu_count = utils.multiprocessing.get_gpu_count()
print(gpu_count)

results = utils.multiprocessing.map(f, gpu_count, settings)

save_results(results, result_folder)