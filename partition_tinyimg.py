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
from defaults_timg import get_cfg_defaults_timg
import numpy as np
from os import path
from scipy import misc
from PIL import Image
from torchvision import datasets
import logging


def get_timg(): 
    timg_test = datasets.ImageFolder(
                'tiny-imagenet-200/test',   
                transform=transforms.Compose(
                    [transforms.Resize(opt.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                    ]
                )
            )
    timg_train = datasets.ImageFolder(
            'tiny-imagenet-200/train',   
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
                ]
            )
        )
    timg_val = datasets.ImageFolder(
            'tiny-imagenet-200/val',   
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
                ]
            )
        )
   
    print(timg_val[1])
    print(timg_test.shape,timg_train.shape,timg_val.shape) 
    images = [x[1] for x in cifar10]
    labels = [x[0] for x in cifar10]

    images = np.asarray(images)
    assert(images.shape == (60000, 3, 32, 32))
    # print(images[0,0,:,:])
    images_1 = images[:,0,:,:]
    images_2 = images[:,1,:,:]
    images_3 = images[:,2,:,:]
    # print(images.shape)
    # images = np.squeeze(images,axis=1)
    assert(images_1.shape == (60000, 32, 32))
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

    assert(images.shape == (60000, 3, 32, 32))

    #save_image(images[:1024], "data_samples.png", pad_value=0.5, nrow=32)
    #save_image(images.astype(dtype=np.float32).mean(0), "data_mean.png", pad_value=0.5, nrow=1)
    #save_image(images.astype(dtype=np.float32).max(0), "data_max.png", pad_value=0.5, nrow=1)

    return [(l, im) for l, im in zip(labels, images)]


def partition(cfg, logger):
    # to reproduce the same shuffle
    random.seed(0)
    cifar10 = get_cifar10()

    random.shuffle(cifar10)

    folds = cfg.DATASET.FOLDS_COUNT

    class_bins = {}

    for x in cifar10:
        if x[0] not in class_bins:
            class_bins[x[0]] = []
        class_bins[x[0]].append(x)

    cifar10_folds = [[] for _ in range(folds)]

    for _class, data in class_bins.items():
        count = len(data)
        logger.info("Class %d count: %d" % (_class, count))

        count_per_fold = count // folds

        for i in range(folds):
            cifar10_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]

    logger.info("Folds sizes:")
    for i in range(len(cifar10_folds)):
        print(len(cifar10_folds[i]))

        output = open(path.join(cfg.DATASET.PATH, 'data_fold_%d.pkl' % i), 'wb')
        pickle.dump(cifar10_folds[i], output)
        output.close()


if __name__ == "__main__":
    cfg = get_cfg_defaults_cifar()
    # cfg.merge_from_file('configs/cifar10.yaml')
    # cfg.freeze()
    logger = logging.getLogger("logger")
    partition(cfg, logger)
