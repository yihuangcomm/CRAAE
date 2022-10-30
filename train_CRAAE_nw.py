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
# ==================================================================================
# original code is from https://github.com/podgorskiy/GPND/blob/master/train_AAE.py
# Modeified by Yi Huang
# ==================================================================================
import torch.utils.data
from defaults_nw import get_cfg_defaults
from torch import optim
from torchvision.utils import save_image
from torch.autograd import Variable
import time
import logging
import os
from dataloading_nw import make_datasets, make_dataloader
from net_nw_newmodel import Generator, Discriminator, Encoder, ZDiscriminator
from utils.tracker import LossTracker
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import dirichlet
import numpy as np


def train(folding_id, inlier_classes, ic):
    cfg = get_cfg_defaults()

    logger = logging.getLogger("logger")
    zsize = cfg.MODEL.LATENT_SIZE
    z_cat_count = 10
    output_folder = os.path.join(cfg.OUTPUT_DIR + "_" + "_".join([str(x) for x in inlier_classes]))   
    os.makedirs(output_folder, exist_ok=True)

    train_set, valid_set, test_set = make_datasets(cfg, folding_id, inlier_classes)
    logger.info("Train set size: %d" % len(train_set))

    G = Generator(cfg.MODEL.LATENT_SIZE+z_cat_count, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    G.weight_init(mean=0, std=0.02)
    print(G)

    D = Discriminator(channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    D.weight_init(mean=0, std=0.02)

    E = Encoder(cfg.MODEL.LATENT_SIZE+z_cat_count, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    E.weight_init(mean=0, std=0.02)

    if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH:
        ZD = ZDiscriminator_mergebatch(zsize, cfg.TRAIN.BATCH_SIZE)
    else:
        ZD = ZDiscriminator(zsize, cfg.TRAIN.BATCH_SIZE)
    ZD.weight_init(mean=0, std=0.02)

    ZD_cat = ZDiscriminator(z_cat_count, cfg.TRAIN.BATCH_SIZE)
    ZD_cat.weight_init(mean=0, std=0.02)

    lr = cfg.TRAIN.BASE_LEARNING_RATE

    G.cuda()
    E.cuda()
    D.cuda()
    ZD.cuda()
    ZD_cat.cuda()

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.999))
    ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.999))
    E_optimizer = optim.Adam(E.parameters(), lr=lr, betas=(0.5, 0.999))

    BCE_loss = nn.BCELoss()
    sample = torch.randn(64, zsize+z_cat_count).view(-1, zsize+z_cat_count, 1, 1)

    tracker = LossTracker(output_folder=output_folder)

    for epoch in range(cfg.TRAIN.EPOCH_COUNT):
        G.train()
        D.train()
        E.train()
        ZD.train()
        ZD_cat.train()

        epoch_start_time = time.time()

        data_loader = make_dataloader(train_set, cfg.TRAIN.BATCH_SIZE, torch.cuda.current_device())
        test_set.shuffle()

        if (epoch + 1) % 30 == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            D_optimizer.param_groups[0]['lr'] /= 4
            GE_optimizer.param_groups[0]['lr'] /= 4
            ZD_optimizer.param_groups[0]['lr'] /= 4
            E_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        for y, x in data_loader:
            x = x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS, 1, cfg.MODEL.INPUT_IMAGE_SIZE)
            
            y_real_ = Variable(torch.ones(x.shape[0]).cuda())
            y_fake_ = Variable(torch.zeros(x.shape[0]).cuda())

            y_real_z = Variable(torch.ones(x.shape[0]).cuda())
            y_fake_z = Variable(torch.zeros(x.shape[0]).cuda())

            y_real_z_cat = Variable(torch.ones(x.shape[0]).cuda())
            y_fake_z_cat = Variable(torch.zeros(x.shape[0]).cuda())  

            uniform_dist = torch.Tensor(x.shape[0], cfg.DATASET.TOTAL_CLASS_COUNT).fill_((1./cfg.DATASET.TOTAL_CLASS_COUNT)).cuda()
            seed = 12345
            alpha = np.array([1.]*z_cat_count)     
            alpha = alpha + 100*np.eye(z_cat_count)[y.cpu().numpy()] 
            arr = map(dirichlet.rvs, alpha)
            y_cat = np.array([i for i in arr])[:,np.newaxis]
            y_cat = torch.Tensor(y_cat.squeeze()).cuda().view(-1,z_cat_count,1,1)
            label = Variable(y.cuda())
            
            with torch.no_grad():

                y = y.reshape(-1,1)
                
            y_onehot = Variable(torch.zeros((x.shape[0], z_cat_count)).cuda().scatter_(1, y, 1.))
        
            #############################################

            D.zero_grad()

            D_result = D(x).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            z = torch.randn((x.shape[0], zsize+z_cat_count)).view(-1, zsize+z_cat_count, 1, 1) 
            
            z = Variable(z)
            _,indices = E(x)
            x_fake = G(z,indices).detach()
            D_result = D(x_fake).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()

            D_optimizer.step()

            tracker.update(dict(D=D_train_loss))


            #############################################

            G.zero_grad()
            
            z = torch.randn((x.shape[0], zsize+z_cat_count)).view(-1, zsize+z_cat_count, 1, 1)
            z = Variable(z)

            x_fake = G(z,indices)
            D_result = D(x_fake).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)

            G_train_loss.backward()
            G_optimizer.step()

            tracker.update(dict(G=G_train_loss))

            #############################################

            ZD.zero_grad()

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize)
            z = Variable(z)

            ZD_result = ZD(z).squeeze()
            ZD_real_loss = BCE_loss(ZD_result, y_real_z)

            z,_ = E(x)
            
            z = z.squeeze().detach()
            z = z[:,z_cat_count::]

            ZD_result = ZD(z).squeeze()
            ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

            z_out,_ = E(x_fake.detach())
            
            z_out = z_out.squeeze().detach()
            z_out = z_out[:,z_cat_count::]

            ZD_result_out = ZD(z_out).squeeze()
            ZD_fake_loss_out = BCE_loss(ZD_result_out, y_fake_z)


            ZD_train_loss = ZD_real_loss*2.0 + ZD_fake_loss + ZD_fake_loss_out
            ZD_train_loss.backward()

            ZD_optimizer.step()

            tracker.update(dict(ZD=ZD_train_loss))
            

            # #############################################

            # ZD_cat.zero_grad()
            E.zero_grad()
            z, _ = E(x)
            z = z.squeeze()

            z = z[:,0:z_cat_count]
            output = F.log_softmax(z,1)
            loss = F.nll_loss(output, label)

            if cfg.DATASET.category=="dirichlet":
                z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1)
                z = torch.cat([y_cat,z],1)  
            elif cfg.DATASET.category=="uniform":
                z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1)
                uni_cat = uniform_dist.view(-1,z_cat_count,1,1)
                z = torch.cat([uni_cat,z],1)
            else:
                z = torch.randn((x.shape[0], zsize+z_cat_count)).view(-1, zsize+z_cat_count, 1, 1)

            z = Variable(z)

            x_fake = G(z,indices).detach()
            z_out, _ = E(x_fake)
            z_out = z_out.squeeze()
            z_out = z_out[:,0:z_cat_count]
            KL_fake_output = F.log_softmax(z_out,1)

            KL_loss_fake = F.kl_div(KL_fake_output, uniform_dist)*z_cat_count
            E_loss = loss + KL_loss_fake*0.03

            E_loss.backward()  

            E_optimizer.step()

            tracker.update(dict(E=E_loss))

            # #############################################
            E.zero_grad()
            G.zero_grad()

            z, _ = E(x)
            x_d = G(z,indices)
           
            z = z.squeeze()
            z_cat = z[:, 0:z_cat_count]
            z = z[:,z_cat_count::]

            E_result = ZD(z).squeeze().detach()
            E_train_loss_style = BCE_loss(E_result, y_real_z)

            z_out, indices_f = E(x_fake)
            x_out = G(z_out,indices_f)

            z_out = z_out.squeeze()
            z_out = z_out[:,z_cat_count::]

            E_result_out = ZD(z_out).squeeze().detach()
            E_train_loss_style_out = BCE_loss(E_result_out, y_real_z)

            E_train_loss = (E_train_loss_style +E_train_loss_style_out) * 1.0

            Recon_loss = F.binary_cross_entropy(x_d, torch.tanh(x.detach()) * 0.5 + 0.5) * 2.0

            Recon_loss_out = F.binary_cross_entropy(x_out, torch.tanh(x.detach()) * 0.5 + 0.5) * 2.0

            Recon_loss_diff = F.relu(Recon_loss-Recon_loss_out)

            (Recon_loss + Recon_loss_diff + E_train_loss).backward()

            GE_optimizer.step()

            tracker.update(dict(GE=Recon_loss + Recon_loss_diff, E=E_train_loss))

            # #############################################
       
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        logger.info('[%d/%d] - ptime: %.2f, %s' % ((epoch + 1), cfg.TRAIN.EPOCH_COUNT, per_epoch_ptime, tracker))

        tracker.register_means(epoch)
        tracker.plot()

    logger.info("Training finish!... save training results")

    model_folder = "models_" + output_folder 
    os.makedirs(model_folder, exist_ok=True)
    classname = "_".join([str(x) for x in inlier_classes])
    print(classname)
    print("Training finish!... save training results")
    torch.save(G.state_dict(), model_folder +"/Gmodel_" + classname + ".pkl")
    torch.save(E.state_dict(), model_folder +"/Emodel_" + classname + ".pkl")

