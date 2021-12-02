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
# original code is from https://github.com/podgorskiy/GPND/blob/master/dataloading.py
# Modeified by Yi Huang
# =========================================================================================

import torch.utils.data
from torchvision import datasets, transforms
from net import *
import pickle
import numpy as np
from os import path
import dlutils
import warnings


class Dataset:
    @staticmethod
    def list_of_pairs_to_numpy(l):
        return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)

    def __init__(self, data):
        self.x, self.y = Dataset.list_of_pairs_to_numpy(data)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.y[index.start:index.stop], self.x[index.start:index.stop]
        return self.y[index], self.x[index]

    def __len__(self):
        return len(self.y)

    def shuffle(self):
        permutation = np.random.permutation(self.y.shape[0])
        for x in [self.y, self.x]:
            np.take(x, permutation, axis=0, out=x)


def make_datasets(cfg, folding_id, inlier_classes):
    data_train = []
    data_valid = []
    data_test = []

 
    if cfg.MODEL.INPUT_IMAGE_CHANNELS==1:
        transform_customer = transforms.Compose(
                    [transforms.Grayscale(num_output_channels=1),
                    transforms.Resize(cfg.MODEL.INPUT_IMAGE_SIZE),
                    transforms.ToTensor()])
    else:
        transform_customer = transforms.Compose(
                    [transforms.Resize(cfg.MODEL.INPUT_IMAGE_SIZE),
                    transforms.ToTensor()])
    if cfg.DATASET.PATH_out=='':                 
        for i in range(cfg.DATASET.FOLDS_COUNT):
            if i != folding_id:
                with open(path.join(cfg.DATASET.PATH, 'data_fold_%d.pkl' % i), 'rb') as pkl:
                    fold = pickle.load(pkl)
                if len(data_valid) == 0:
                    data_valid = fold
                else:
                    data_train += fold

        data_train = [x for x in data_train if x[0] in inlier_classes]

        with open(path.join(cfg.DATASET.PATH, 'data_fold_%d.pkl') % folding_id, 'rb') as pkl:
            data_test = pickle.load(pkl)  
    else:
        if cfg.DATASET.PATH=='svhn':
            data_train = datasets.SVHN('svhn', split='train', 
                        transform=transform_customer, 
                        download=True)
            data_test = datasets.SVHN('svhn', split='test', 
                        transform=transform_customer, 
                        download=False)
        elif cfg.DATASET.PATH=='cifar10':
            data_train = datasets.CIFAR10('cifar10', train=True, 
                        transform=transform_customer, 
                        download=True) 
            data_test= datasets.CIFAR10('cifar10', train=False, 
                        transform=transform_customer, 
                        download=False)
        elif cfg.DATASET.PATH=='cifar100':
            data_train = datasets.CIFAR100('cifar100', train=True, 
                        transform=transform_customer, 
                        download=True) 
            data_test= datasets.CIFAR100('cifar100', train=False, 
                        transform=transform_customer, 
                        download=False)
        elif cfg.DATASET.PATH=='mnist':
            data_train = datasets.MNIST('Mnist', train=True, 
                        transform=transform_customer, 
                        download=True)        
            data_test = datasets.MNIST('Mnist', train=False, 
                        transform=transform_customer, 
                        download=False)

        data_train = [(x[1],(255.0*x[0]).tolist()) for x in data_train if x[1] in inlier_classes]  
        data_test = [(x[1],(255.0*x[0]).tolist()) for x in data_test]  
        
        if cfg.DATASET.PATH_out=='svhn':
            data_test_out = datasets.SVHN('svhn', split='test', 
                        transform=transform_customer, 
                        download=True)    
        elif cfg.DATASET.PATH_out=='cifar10':
            data_test_out= datasets.CIFAR10('cifar10', train=False, 
                        transform=transform_customer, 
                        download=True)
        elif cfg.DATASET.PATH_out=='cifar100':
            data_test_out= datasets.CIFAR100('cifar100', train=False, 
                        transform=transform_customer, 
                        download=True)
        elif cfg.DATASET.PATH_out=='lsun':
            data_test_out = datasets.ImageFolder('LSUN_resize', 
                        transform=transform_customer)
        elif cfg.DATASET.PATH_out=='timg':                
            data_test_out = datasets.ImageFolder('imagenet_resize', 
                        transform=transform_customer)
        elif cfg.DATASET.PATH_out=='mnist':                
            data_test_out = datasets.MNIST('Mnist', train=False, 
                        transform=transform_customer, 
                        download=True)
        test_set_out = [(cfg.DATASET.TOTAL_CLASS_COUNT,(255.0*x[0]).tolist()) for x in data_test_out]
        data_test += test_set_out
        data_valid = data_test
    train_set = Dataset(data_train)
    valid_set = Dataset(data_valid)
    test_set = Dataset(data_test)

    train_set.shuffle()
    valid_set.shuffle()
    test_set.shuffle()
    return train_set, valid_set, test_set


def make_dataloader(dataset, batch_size, device):
    class BatchCollator(object):
        def __init__(self, device):
            self.device = device

        def __call__(self, batch):
            with torch.no_grad():
                y, x = batch
                x = torch.tensor(x / 255.0, requires_grad=True, dtype=torch.float32, device=self.device)
                y = torch.tensor(y, dtype=torch.int64, device=self.device)
                return y, x

    data_loader = dlutils.batch_provider(dataset, batch_size, BatchCollator(device))
    return data_loader


def create_set_with_outlier_percentage(dataset, inlier_classes, target_percentage, concervative=True):
    np.random.seed(0)
    dataset.shuffle()
    dataset_outlier = [x for x in dataset if x[0] not in inlier_classes]
    dataset_inlier = [x for x in dataset if x[0] in inlier_classes]

    def increase_length(data_list, target_length):
        repeat = (target_length + len(data_list) - 1) // len(data_list)
        data_list = data_list * repeat
        data_list = data_list[:target_length]
        return data_list

    if not concervative:
        inlier_count = len(dataset_inlier)
        outlier_count = inlier_count * target_percentage // (100 - target_percentage)

        if len(dataset_outlier) > outlier_count:
            dataset_outlier = dataset_outlier[:outlier_count]
        else:
            outlier_count = len(dataset_outlier)
            inlier_count = outlier_count * (100 - target_percentage) // target_percentage
            dataset_inlier = dataset_inlier[:inlier_count]
    else:
        inlier_count = len(dataset_inlier)
        outlier_count = len(dataset_outlier)

        current_percentage = outlier_count * 100 / (outlier_count + inlier_count)

        if current_percentage < target_percentage:  # we don't have enought outliers
            outlier_count = int(inlier_count * target_percentage / (100.0 - target_percentage))
            dataset_outlier = increase_length(dataset_outlier, outlier_count)
        else:  # we don't have enought inliers
            inlier_count = int(outlier_count * (100.0 - target_percentage) / target_percentage)
            dataset_inlier = increase_length(dataset_inlier, inlier_count)

    dataset = Dataset(dataset_outlier + dataset_inlier)

    dataset.shuffle()

    # Post checks
    outlier_count = len([1 for x in dataset if x[0] not in inlier_classes])
    inlier_count = len([1 for x in dataset if x[0] in inlier_classes])
    real_percetage = outlier_count * 100.0 / (outlier_count + inlier_count)
    assert abs(real_percetage - target_percentage) < 0.01, "Didn't create dataset with requested percentage of outliers"

    return dataset
