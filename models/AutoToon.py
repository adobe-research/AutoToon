# Copyright 2020 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import init
import os
from datetime import datetime
import pickle

import sys
sys.path.append('..')
from models.vggface2_senet import senet50
from models.download_weights import download_weights

use_data_parallel = False


class AutoToonModel(nn.Module):
    def __init__(self, init_type='kaiming', models_root='./models', lr=1e-4, lrd=1e-1, wd=0, force_cpu=False, train=True):
        super(AutoToonModel, self).__init__()
        self.senet = senet50(include_top=False)  # output is warp grid, shape (1, 2, 32, 32)
        self.upsample = nn.UpsamplingBilinear2d(size=256)
        
        if train:
            self.init_weights(init_type)
            if not os.path.isfile(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'senet50_ft_weight.pkl')):
                download_weights()
            self.load_senet(os.path.join(models_root, 'senet50_ft_weight.pkl'))
        
        self.model = self.senet
        self.optimizer = optim.Adam(self.parameters(), 
                            lr=lr, 
                            betas=(0.5, 0.999),
                            weight_decay=wd
                        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=lrd)
        
        self.device = torch.device('cuda:0') if torch.cuda.is_available() and not force_cpu else torch.device('cpu')
        self.gpu_ids = [0]
        if use_data_parallel:
            self.senet = nn.DataParallel(self.senet)
            self.upsample = nn.DataParallel(self.upsample)
        self.to(self.device)

    def forward(self, x):
        flow = self.senet(x) / 100  # normalize to well within [-1, 1] range
        flow_norm = self.upsample(flow)
        warped = self.flow_warp(x, flow_norm)
        return warped, flow_norm, flow

    def flow_warp(self, x, flow, padding_mode='border'):
        """
        Warps an image or feature map with optical flow.
        Arguments:
            `x` (Tensor): size (n, c, h, w)
            `flow` (Tensor): size (n, 2, h, w), values range from -1 to 1 (relevant to image width or height)
            `padding_mode` (str): 'zeros' or 'border'
        Returns:
            Tensor: warped image or feature map according to `flow`
        Code borrowed from https://github.com/hellock/cvbase/issues/4.
        """
        assert x.size()[-2:] == flow.size()[-2:]
        n, _, h, w = x.size()
        x_ = torch.arange(w).view(1, -1).expand(h, -1)
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)
        grid = torch.stack([x_, y_], dim=0).float().to(self.device)
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1).clone()
        grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
        grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
        grid = grid + flow
        grid = grid.permute(0, 2, 3, 1)
        x = x.to(self.device)
        return F.grid_sample(x, grid, padding_mode=padding_mode)

    def init_weights(self, init_type='kaiming', init_gain=0.02):
        """Initialize network weights.
        Code slightly modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py.
        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        self.apply(init_func)  # apply the initialization function <init_func>

    def save_model(self, epoch, save_dir='./checkpoints'):
        """
        Saves a model checkpoint at epoch `epoch` in directory `save_dir`.
        """
        save_filename = '%s_model.pth' % epoch
        save_path = os.path.join(save_dir, save_filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)
        
    def load_model(self, epoch, save_dir='./checkpoints', device='cpu'):
        """
        Loads a model checkpoint from epoch `epoch` from directory `save_dir` from device `device`.
        """
        load_filename = '%s_model.pth' % epoch
        load_path = os.path.join(save_dir, load_filename)
        checkpoint = torch.load(load_path, map_location=device)
        # checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)  # uncomment for CPU
        print('loading the model from %s' % load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('loading the optimizer from %s' % load_path)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading the scheduler from %s' % load_path)
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'] + 1  # epoch to start counting from

    def load_senet(self, fname):
        """
        Code slightly modified from https://github.com/cydonia999/VGGFace2-pytorch/blob/master/utils.py.
        Set parameters converted from Caffe models that authors of VGGFace2 provide.
        See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
        Arguments:
            self: model
            fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
        """
        with open(fname, 'rb') as f:
            weights = pickle.load(f, encoding='latin1')

        own_state = self.senet.state_dict()
        for name, param in weights.items():
            if name in own_state:
                try:
                    own_state[name].copy_(torch.from_numpy(param))
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                       'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.shape))
            # else:  # commented out this portion because own_state is strict subset of pretrained senet being loaded
            #     raise KeyError('unexpected key "{}" in state_dict'.format(name))
        print('senet loaded from', fname)
