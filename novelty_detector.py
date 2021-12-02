# Copyright 2018-2020 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================================
# original code is from https://github.com/podgorskiy/GPND/blob/master/novelty_detector.py
# Modeified by Yi Huang
# =========================================================================================

from __future__ import print_function
import torch.utils.data
import os
from torchvision.utils import save_image
from net import *
from torch.autograd import Variable
from utils.jacobian import compute_jacobian
import numpy as np
import logging
import scipy.optimize
import pickle
from dataloading_out import make_datasets, make_dataloader, create_set_with_outlier_percentage
from defaults import get_cfg_defaults
# from defaults_svhn import get_cfg_defaults_svhn
from evaluation import get_f1, evaluate
from utils.threshold_search import find_maximum
from utils.save_plot import save_plot
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import loggamma, softmax
from timeit import default_timer as timer


def r_pdf(x, bins, counts):
    if bins[0] < x < bins[-1]:
        i = np.digitize(x, bins) - 1
        return max(counts[i], 1e-308)
    if x < bins[0]:
        return max(counts[0] * x / bins[0], 1e-308)
    return 1e-308


def extract_statistics(cfg, train_set, inlier_classes, E, G):
    zlist = []
    rlist = []
    z_cat_count = 10
    data_loader = make_dataloader(train_set, cfg.TEST.BATCH_SIZE, torch.cuda.current_device())

    for label, x in data_loader:
        x = x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS*cfg.MODEL.INPUT_IMAGE_SIZE * cfg.MODEL.INPUT_IMAGE_SIZE)
        z = E(x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE))
        recon_batch = G(z)
        z = z.squeeze()

        recon_batch = recon_batch.squeeze().cpu().detach().numpy()
        x = x.squeeze().cpu().detach().numpy()

        z = z.cpu().detach().numpy()

        for i in range(x.shape[0]):
            distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())
            rlist.append(distance)

        zlist.append(z)

    zlist = np.concatenate(zlist)

    counts, bin_edges = np.histogram(rlist, bins=30, normed=True)

    if cfg.MAKE_PLOTS:
        plt.plot(bin_edges[1:], counts, linewidth=2)
        save_plot(r"Distance, $\left \|\| I - \hat{I} \right \|\|$",
                  'Probability density',
                  r"PDF of distance for reconstruction error, $p\left(\left \|\| I - \hat{I} \right \|\| \right)$",
                  'mnist_%s_reconstruction_error.pdf' % ("_".join([str(x) for x in inlier_classes])))

    for i in range(cfg.MODEL.LATENT_SIZE+z_cat_count):
        plt.hist(zlist[:, i], bins='auto', histtype='step')

    if cfg.MAKE_PLOTS:
        save_plot(r"$z$",
                  'Probability density',
                  r"PDF of embeding $p\left(z \right)$",
                  'mnist_%s_embedding.pdf' % ("_".join([str(x) for x in inlier_classes])))

    def fmin(func, x0, args, disp):
        x0 = [2.0, 0.0, 1.0]
        return scipy.optimize.fmin(func, x0, args, xtol=1e-12, ftol=1e-12, disp=0)

    gennorm_param = np.zeros([3, cfg.MODEL.LATENT_SIZE+z_cat_count])
    for i in range(cfg.MODEL.LATENT_SIZE+z_cat_count):
        betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i], optimizer=fmin)
        gennorm_param[0, i] = betta
        gennorm_param[1, i] = loc
        gennorm_param[2, i] = scale

    return counts, bin_edges, gennorm_param


def main(folding_id, inlier_classes, ic, total_classes, mul, folds=5):
    cfg = get_cfg_defaults()
    logger = logging.getLogger("logger")

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.cuda.current_device()
    print("Running on ", torch.cuda.get_device_name(device))
    train_set,valid_set, test_set = make_datasets(cfg, folding_id, inlier_classes)
    print('Train set size: %d' % len(train_set))
    print('Validation set size: %d' % len(valid_set))
    print('Test set size: %d' % len(test_set))

    train_set.shuffle()
    z_cat_count = 10
    G = Generator(cfg.MODEL.LATENT_SIZE+z_cat_count, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    E = Encoder(cfg.MODEL.LATENT_SIZE+z_cat_count, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)

    output_folder = os.path.join(cfg.OUTPUT_DIR + "_" + "_".join([str(x) for x in inlier_classes]))
    os.makedirs(output_folder, exist_ok=True)

    classname = "_".join([str(x) for x in inlier_classes])
    print(classname)
    model_folder = "models_" + cfg.OUTPUT_DIR + "_" + "_".join([str(x) for x in inlier_classes])
 
    G.load_state_dict(torch.load('models_results_cifar_0_1_2_3_4_5_6_7_8_9' +"/Gmodel_" + classname + ".pkl")) #model_folder +"/Gmodel_" + classname + ".pkl"))
    E.load_state_dict(torch.load('models_results_cifar_0_1_2_3_4_5_6_7_8_9' +"/Emodel_" + classname + ".pkl"))#model_folder +"/Emodel_" + classname + ".pkl"))

    G.eval()
    E.eval()

    sample = torch.randn(64, cfg.MODEL.LATENT_SIZE+z_cat_count).to(device)
    sample = G(sample.view(-1, cfg.MODEL.LATENT_SIZE+z_cat_count, 1, 1)).cpu()
    save_image(sample.view(64, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE), 'sample.png')

    counts, bin_edges, gennorm_param = extract_statistics(cfg, train_set, inlier_classes, E, G)

    def run_novely_prediction_on_dataset(dataset, percentage, concervative=False):
        dataset.shuffle()
        dataset = create_set_with_outlier_percentage(dataset, inlier_classes, percentage, concervative)
        z_cat_count = 10
        result = []
        sf_out = np.empty((1,z_cat_count))
        sf_label = []
        gt_novel = []

        data_loader = make_dataloader(dataset, cfg.TEST.BATCH_SIZE, torch.cuda.current_device()) 
        include_jacobian = True

        N = ((cfg.MODEL.INPUT_IMAGE_SIZE) * cfg.MODEL.INPUT_IMAGE_SIZE - (cfg.MODEL.LATENT_SIZE+z_cat_count)) * mul
        logC = loggamma(N / 2.0) - (N / 2.0) * np.log(2.0 * np.pi)

        def logPe_func(x):
            return logC - (N - 1) * np.log(x) + np.log(r_pdf(x, bin_edges, counts))
        for j, (label, x) in enumerate(data_loader):
            x = x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS * cfg.MODEL.INPUT_IMAGE_SIZE * cfg.MODEL.INPUT_IMAGE_SIZE)
            x = Variable(x.data, requires_grad=True)

            z = E(x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE))
            recon_batch = G(z)
            z = z.squeeze()
            

            if include_jacobian:
                J = compute_jacobian(x, z)
                J = J.cpu().numpy()

            z = z.cpu().detach().numpy()
            z_cat = z[:,0:z_cat_count]
            z_max = np.max(softmax(z_cat,1), 1)
            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            for i in range(x.shape[0]):
                if include_jacobian:
                    u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                    logD = -np.sum(np.log(np.abs(s)))  # | \mathrm{det} S^{-1} |
                    # logD = np.log(np.abs(1.0/(np.prod(s))))
                else:
                    logD = 0

                p = scipy.stats.gennorm.pdf(z[i]+(1e-9), gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
                logPz = np.sum(np.log(p))

                # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
                # In this case, just assign some large negative value to make sure that the sample
                # is classified as unknown.
                if not np.isfinite(logPz):
                    logPz = -1000

                distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())
                logPe = logPe_func(distance)
                if cfg.DATASET.calibrate=="calibrate":
                    P = logPe/z_max[i] # + logD + logPz)/z_max[i] 
                else:
                    P = logPe
                result.append(P)
                gt_novel.append(label[i].item() in inlier_classes)
            if j==0:
                sf_out = softmax(z_cat,1)
                sf_label = label.tolist()
            else:
                sf_out = np.vstack((sf_out,softmax(z_cat,1)))
                sf_label += label.tolist()
        result = np.asarray(result, dtype=np.float32)
        ground_truth = np.asarray(gt_novel, dtype=np.float32)
        return result, ground_truth, sf_out,sf_label

    def compute_threshold(valid_set, percentage):
        y_scores, y_true, softmax_out, softmax_label = run_novely_prediction_on_dataset(valid_set, percentage, concervative=True)
        minP = min(y_scores) - 1
        maxP = max(y_scores) + 1
        y_false = np.logical_not(y_true)

        def evaluate(e):
            y = np.greater(y_scores, e)
            true_positive = np.sum(np.logical_and(y, y_true))
            false_positive = np.sum(np.logical_and(y, y_false))
            false_negative = np.sum(np.logical_and(np.logical_not(y), y_true))
            return get_f1(true_positive, false_positive, false_negative)

        best_th, best_f1 = find_maximum(evaluate, minP, maxP, 1e-4)

        logger.info("Best e: %f best f1: %f" % (best_th, best_f1))
        return best_th

    def test(test_set, percentage, threshold):
        y_scores, y_true, softmax_out, softmax_label = run_novely_prediction_on_dataset(test_set, percentage, concervative=True)
        return evaluate(logger, percentage, inlier_classes, y_scores, threshold, y_true, softmax_out, softmax_label)

    percentages = cfg.DATASET.PERCENTAGES

    results = {}

    for p in percentages:
        plt.figure(num=None, figsize=(8, 6), dpi=180, facecolor='w', edgecolor='k')
        e = compute_threshold(valid_set, p)
        results[p] = test(test_set, p, e)

    return results

