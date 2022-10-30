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
from net import *
import pickle
import numpy as np
from os import path
import dlutils
import warnings
import pandas
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch import nn

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

def onehot(list):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(list)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def get_XY(Dir):
    X = []
    Y = []
    
    for dirpath, _, filenames in os.walk(Dir):  
        for file in filenames :
            x = np.array(pandas.read_csv(os.path.join(dirpath, file)).loc[:])
            x = x[:,[0,20]]
            X.append(x)
            # depend on how to extract labels
            y = file[4:5] # for youtubemusic # file[5:7] for youtube
            Y.append(y)
    Y = np.argmax(onehot(Y),axis=1)
    Y = Y.tolist()
    return zip(Y,X)
def get_XY2(Dir):
    X = []
    Y = []
    for dirpath, _, filenames in os.walk(Dir):  
        for file in filenames :
            x = np.array(pandas.read_csv(os.path.join(dirpath, file)).loc[:])
            x = x[:,[0,20]]
            X.append(x)
            # depend on how to extract labels
            y = 10
            Y.append(y)
    return zip(Y,X)



def make_datasets(cfg, folding_id, inlier_classes):
    data_train = []
    data_test = []  
    data_train = get_XY(cfg.DATASET.PATH)  # for youtubemusic # get_XY("train_" + cfg.DATASET.PATH) for youtube
    data_train = [x for x in data_train if x[0] in inlier_classes]
    data_test = get_XY("test_" + cfg.DATASET.PATH)
    data_test = [x for x in data_test]
    data_valid = [x for i,x in enumerate(data_test) if i%2==0]
    data_test = [x for i,x in enumerate(data_test) if i%2==1]
    print(len(data_test))
    if cfg.DATASET.PATH_out!='':
        data_test_out = get_XY2(cfg.DATASET.PATH_out)
        data_test_out = [x for x in data_test_out]
        data_valid += [x for i,x in enumerate(data_test_out) if i%2==0]
        data_test += [x for i,x in enumerate(data_test_out) if i%2==1]

    train_set = Dataset(data_train)
    valid_set = Dataset(data_valid)
    
    test_set = Dataset(data_test)
 
    return train_set, valid_set, test_set

def make_dataloader(dataset, batch_size, device):
    class BatchCollator(object):
        def __init__(self, device):
            self.device = device

        def __call__(self, batch):
            with torch.no_grad():
                y, x = batch
                x = torch.tensor(x, requires_grad=True, dtype=torch.float32, device=self.device)
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
    assert abs(real_percetage - target_percentage) < 1, "Didn't create dataset with requested percentage of outliers"

    return dataset
