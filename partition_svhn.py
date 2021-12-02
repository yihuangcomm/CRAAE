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
# ==============================================================================

import dlutils
import random
import pickle
from defaults_svhn import get_cfg_defaults_svhn
import numpy as np
from os import path
from scipy import misc
from PIL import Image
import logging
import scipy.io as sio


def get_svhn():
    # svhn = dlutils.reader.Cifar10('cifar10/cifar-10-batches-bin', train=True, test=True).items
       

    # reading(loading) mat file as array
    svhn = sio.loadmat('svhn/train_32x32')

    svhn_data = svhn['X']
    # loading from the .mat file gives an np array of type np.uint8
    # converting to np.int64, so that we have a LongTensor after
    # the conversion from the numpy array
    # the squeeze is needed to obtain a 1D tensor
    labels = svhn['y'].astype(np.int64).squeeze()

    # the svhn dataset assigns the class label "10" to the digit 0
    # this makes it inconsistent with several loss functions
    # which expect the class labels to be in the range [0, C-1]
    np.place(labels,labels == 10, 0)
    svhn_data = np.transpose(svhn_data, (3, 2, 0, 1))

    images = [x for x in svhn_data]
    labels = [y for y in labels]

    images = np.asarray(images)
    assert(images.shape == (73257, 3, 32, 32))
    # print(images[0,0,:,:])
    images_1 = images[:,0,:,:]
    images_2 = images[:,1,:,:]
    images_3 = images[:,2,:,:]
    # print(images.shape)
    # images = np.squeeze(images,axis=1)
    assert(images_1.shape == (73257, 32, 32))
    # print(images[0,:,:])
    _images = []
    for im_1, im_2, im_3 in zip(images_1, images_2, images_3):

        # im = misc.imresize(im, (32, 32), interp='bilinear')
        # im = np.array(Image.fromarray(im).resize((32, 32)))
        # print("*****************************************************")
        # print(im)
        im_1 = Image.fromarray(im_1)
        im_2 = Image.fromarray(im_2)
        im_3 = Image.fromarray(im_3)
        im_1 = im_1.resize((32,32))
        im_2 = im_2.resize((32,32))
        im_3 = im_3.resize((32,32))
        im_1 = np.array(im_1)
        im_2 = np.array(im_2)
        im_3 = np.array(im_3)
        # print(im_1.shape)
        im = np.stack((im_1,im_2,im_3))
        # print(im.shape)
        _images.append(im)
    images = np.asarray(_images)

    assert(images.shape == (73257, 3, 32, 32))

    #save_image(images[:1024], "data_samples.png", pad_value=0.5, nrow=32)
    #save_image(images.astype(dtype=np.float32).mean(0), "data_mean.png", pad_value=0.5, nrow=1)
    #save_image(images.astype(dtype=np.float32).max(0), "data_max.png", pad_value=0.5, nrow=1)

    return [(l, im) for l, im in zip(labels, images)]


def partition(cfg, logger):
    # to reproduce the same shuffle
    random.seed(0)
    svhn = get_svhn()

    random.shuffle(svhn)

    folds = cfg.DATASET.FOLDS_COUNT

    class_bins = {}

    for x in svhn:
        if x[0] not in class_bins:
            class_bins[x[0]] = []
        class_bins[x[0]].append(x)

    svhn_folds = [[] for _ in range(folds)]

    for _class, data in class_bins.items():
        count = len(data)
        logger.info("Class %d count: %d" % (_class, count))

        count_per_fold = count // folds

        for i in range(folds):
            svhn_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]

    logger.info("Folds sizes:")
    for i in range(len(svhn_folds)):
        print(len(svhn_folds[i]))

        output = open(path.join(cfg.DATASET.PATH, 'data_fold_%d.pkl' % i), 'wb')
        pickle.dump(svhn_folds[i], output)
        output.close()


if __name__ == "__main__":
    cfg = get_cfg_defaults_svhn()
    # cfg.merge_from_file('configs/cifar10.yaml')
    # cfg.freeze()
    logger = logging.getLogger("logger")
    partition(cfg, logger)
