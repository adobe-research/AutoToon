# Copyright 2020 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image, make_grid

from models.AutoToon import AutoToonModel
from dataset import AutoToonDataset

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_dataloader(data, train=True, batch_size=64, num_workers=32):
    loader = None
    if train:
        loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader

def to_device(x):
    return x.to(device)

def recon_loss(pred, true, flow, true_flow):
    return F.l1_loss(pred, true) + 0.7 * F.l1_loss(flow, true_flow)

def reg_loss(flow):  # cosine similarity loss
    return (1 - F.cosine_similarity(flow[:, :, :-1, :-1], flow[:, :, :-1, 1:]) + 1 - F.cosine_similarity(flow[:, :, :-1, :-1], flow[:, :, 1:, :-1])).sum()

def get_flow_figure(flow, axis):
    flow_fig = plt.figure(figsize=(16,16))
    for i in range(flow.shape[0]):
        fig = plt.subplot(6, 4, i + 1)
        plt.imshow(flow[i, :, :, :].detach().cpu().squeeze(0).permute(1,2,0)[:, :, axis])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.colorbar()
    return flow_fig

def train(model, logfilename, epochs=20, root='.', print_every=100, save_every=5, batch_size=64, num_workers=32,
            save_dir='./checkpoints', start_epoch=0):
    writer = SummaryWriter()
    model.train()
    total_loss = 0

    data = AutoToonDataset(root=root, train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5072122, 0.4338291, 0.38466746], [0.08460502, 0.07533269, 0.07269038])]), random_crop=False, color_jitter=True)
    val_data = AutoToonDataset(root=root, train=False, random_flip=False, color_jitter=False, random_crop=False)
    val_data_flip = AutoToonDataset(root=root, train=False, always_flip=True, color_jitter=False, random_crop=False)
    val_disp_data = AutoToonDataset(root=root, train=False, transform=transforms.Compose([transforms.ToTensor()]), random_flip=False, color_jitter=False, random_crop=False)
    val_disp_data_flip = AutoToonDataset(root=root, train=False, transform=transforms.Compose([transforms.ToTensor()]), always_flip=True, color_jitter=False, random_crop=False)
    train_loader = get_dataloader(data=data, train=True, batch_size=batch_size, num_workers=num_workers)    
    epoch_losses = []  # for plotting purposes

    for epoch in range(epochs):
        epoch_loss = 0
        print('BEGIN EPOCH {}'.format(epoch + start_epoch))
        model.train()
        for batch, sample in enumerate(train_loader):
            photos, caris, warp = sample['p_img'], sample['c_img'], sample['warp32']
            photos = to_device(photos.float())
            caris = to_device(caris.float())
            warp = to_device(warp.float())
            generated, flow, flow_small = model(photos)

            model.optimizer.zero_grad()

            lossRecon = recon_loss(generated, caris, flow_small, warp)
            lossReg = 1e-6 * reg_loss(flow)
            loss = lossRecon + lossReg
            loss.backward()
            writer.add_scalar('reconstruction loss', lossRecon.item(), epoch * len(train_loader) + batch)
            writer.add_scalar('cosine similarity regularization loss', lossReg.item(), epoch * len(train_loader) + batch)
            writer.add_scalar('total loss', loss.item(), epoch * len(train_loader) + batch)
            total_loss += loss.item()
            epoch_loss += loss.item()

            model.optimizer.step()
            model.scheduler.step()

            if batch % print_every == 0:
                print('iter {} of {} ({:.0f}%) \t \t Loss: {:.6f}'.format(
                    batch, len(train_loader),
                    100. * batch / len(train_loader),
                    loss.item()))

        print('END OF EPOCH {} --- Average Loss: {:.6f}'.format(epoch + start_epoch, epoch_loss / len(train_loader)))
        epoch_losses.append(epoch_loss / len(train_loader))
        with open(logfilename, 'a') as logfile:
            logfile.write('epoch {} average loss: {}\n'.format(epoch + start_epoch, epoch_losses[epoch]))

        if epoch % save_every == 0:
            print('----- saving model checkpoint, epoch {} -----'.format(epoch + start_epoch))
            model.save_model(epoch + start_epoch, save_dir=save_dir)
            
            # log validation image, loss results
            with torch.no_grad():
                model.eval()
                val_imgs = [val_data.__getitem__(i) for i in range(val_data.__len__())] + [val_data_flip.__getitem__(i) for i in range(val_data_flip.__len__())]
                imgs = torch.stack([sample['p_img'] for sample in val_imgs])
                caris = torch.stack([sample['c_img'] for sample in val_imgs])
                warp = torch.stack([sample['warp32'] for sample in val_imgs])
                
                disp_imgs = [val_disp_data.__getitem__(i) for i in range(val_disp_data.__len__())] + [val_disp_data_flip.__getitem__(i) for i in range(val_disp_data.__len__())]
                disp_imgs = torch.stack([sample['p_img'] for sample in disp_imgs])

                # run model to get results
                generated, flow, flow_small = model(imgs.float().to(device))
                output = torch.stack([model.flow_warp(disp_imgs[i].unsqueeze(0).float().to(device), flow[i].unsqueeze(0).detach().to(device)) for i in range(disp_imgs.__len__())])
                output = output.squeeze(1)
                # validation images
                writer.add_image('validation set results', make_grid(output, padding=10, nrow=4), epoch)
                writer.add_image('validation set originals', make_grid(disp_imgs, padding=10, nrow=4), epoch)
                # validation flow images
                flow_fig = get_flow_figure(flow, 0)
                writer.add_figure('validation flow x results', flow_fig, epoch)
                flow_fig = get_flow_figure(flow, 1)
                writer.add_figure('validation flow y results', flow_fig, epoch)

                # log validation loss
                lossRecon_val = recon_loss(generated, caris.to(device).float(), flow_small, warp.to(device).float())
                lossReg_val = 1e-6 * reg_loss(flow)
                loss_val = lossRecon_val + lossReg_val
                writer.add_scalar('validation reconstruction loss', lossRecon_val.item(), epoch * len(train_loader) + batch)
                writer.add_scalar('validation cosine similarity regularization loss', lossReg_val.item(), epoch * len(train_loader) + batch)
                writer.add_scalar('total validation loss', loss_val.item(), epoch * len(train_loader) + batch)
            
    model.save_model(epoch + start_epoch, save_dir=save_dir)
    return epoch_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training options and configuration
    parser.add_argument('--batch_size', type=int, help='training batch size', default=16)
    parser.add_argument('--num_workers', '--nworkers', type=int, help='number of workers', default=32)
    parser.add_argument('--root', type=str, help='root directory containing AutoToon dataset', default='./dataset')
    parser.add_argument('--epochs', type=int, help='number of epochs to train, starting from 0', default=5000)
    parser.add_argument('--print_every', type=int, help='frequency (in batches) with which to print loss', default=1)
    parser.add_argument('--save_every', type=int, help='frequency (in epochs) with which to save model', default=10)
    parser.add_argument('--save_dir', type=str, help='existing directory in which to save model', default='./checkpoints')
    parser.add_argument('--log_dir', type=str, help='existing directory in which to save log', default='./logs')
    parser.add_argument('--continue_train', nargs=2, metavar=('epoch', 'directory'), help='continue training from a given epoch', default=[None, './checkpoints'])

    # model and optimizer options
    parser.add_argument('--init_type', type=str, help='model weight initialization method', default='kaiming')
    parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--lrd', type=float, help='learning rate decay factor', default=1)
    parser.add_argument('--wd', type=float, help='weight decay', default=0)

    args = parser.parse_args()

    # save options to log
    now = datetime.now().strftime('%Y-%m-%d_%H:%M')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    filename = os.path.join(args.log_dir, 'train_' + now + '.txt')
    logfile = open(filename, 'w')
    for var in vars(args):
        logfile.write(var)
        logfile.write(' = ')
        logfile.write(str(vars(args)[var]) + '\n')
    logfile.write('\n')
    logfile.close()

    # train the model
    model = AutoToonModel(init_type=args.init_type, lr=args.lr, lrd=args.lrd, wd=args.wd)

    start_epoch = 0
    if args.continue_train[0] != None:
        start_epoch = model.load_model(args.continue_train[0], args.continue_train[1])

    epoch_losses = train(model, filename, epochs=args.epochs, root=args.root, print_every=args.print_every,
            save_every=args.save_every, batch_size=args.batch_size, num_workers=args.num_workers, start_epoch=start_epoch, save_dir=args.save_dir)
