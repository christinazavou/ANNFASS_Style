# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import argparse
import logging
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from tensorboardX import SummaryWriter

import MinkowskiEngine as ME

from datasets.dataset_utils import InfSampler, resample_mesh, collate_pointcloud_fn, PointCloud
from datasets.Component import ComponentObjDataset, ComponentMeshDataset, ComponentSamplesDataset
from models.vae import Encoder, Decoder
from utils import dotdict, setup_logging, SCREEN_CAMERA_4MODELNET2_FILE, set_logger_file
from torch.utils.data import DataLoader
from utils.visualize import plot_3d_point_cloud
from utils.open3d_vis import render_four_point_clouds


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--resolution', type=int, default=128)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--dataset', type=str, required=False, default="ComponentObjDataset")
parser.add_argument('--train_split_file', type=str)
parser.add_argument('--val_split_file', type=str)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--save_freq', type=int, default=1000)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--stat_freq', type=int, default=50)
parser.add_argument('--weights', type=str, default='modelnet_vae.pth')
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--load_optimizer', type=str, default='true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--export', action='store_true')
parser.add_argument('--encodings_dir', type=str, default='encodings')
parser.add_argument('--max_visualization', type=int, default=4)


def make_data_loader(phase, batch_size, shuffle, num_workers, repeat, config, data_dict, logger):

    args = {}

    dset = ComponentObjDataset(config.data_dir, phase, logger=logger, config=dotdict(data_dict))

    if repeat:
        args['sampler'] = InfSampler(dset, shuffle)
    else:
        args['shuffle'] = shuffle

    loader = DataLoader(dset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=False,
                        pin_memory=False,
                        collate_fn=collate_pointcloud_fn,
                        **args)

    return loader


###############################################################################
# End of utility functions
###############################################################################


class VAE(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, sinput, gt_target, keep=False):
        means, log_vars = self.encoder(sinput)
        zs = means
        if self.training:
            zs = zs + torch.exp(0.5 * log_vars.F) * torch.randn_like(log_vars.F)
        out_cls, targets, sout = self.decoder(zs, gt_target, keep)
        return out_cls, targets, sout, means, log_vars, zs


def train(net, train_dataloader, val_dataloader, device, config):

    writer = SummaryWriter(log_dir=os.path.join(config.log_dir, "logs"))

    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    crit = nn.BCEWithLogitsLoss()

    start_iter = 0
    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        logger.info('Resuming weights')
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_iter = checkpoint['curr_iter']

    net.train()
    train_iter = iter(train_dataloader)
    # val_iter = iter(val_dataloader)
    logger.info(f'LR: {scheduler.get_lr()}')
    for i in range(start_iter, config.max_iter):

        s = time()
        data_dict = train_iter.next()
        d = time() - s

        optimizer.zero_grad()
        sin = ME.SparseTensor(
            torch.ones(len(data_dict['coords']), 1),
            data_dict['coords'].int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
        ).to(device)

        # Generate target sparse tensor
        target_key = sin.coords_key

        out_cls, targets, sout, means, log_vars, zs = net(sin, target_key)
        num_layers, BCE = len(out_cls), 0
        losses = []
        for out_cl, target in zip(out_cls, targets):
            curr_loss = crit(out_cl.F.squeeze(),
                             target.type(out_cl.F.dtype).to(device))
            losses.append(curr_loss.item())
            BCE += curr_loss / num_layers

        KLD = -0.5 * torch.mean(
            torch.mean(1 + log_vars.F - means.F.pow(2) - log_vars.F.exp(), 1))
        loss = KLD + BCE

        loss.backward()
        optimizer.step()
        t = time() - s

        if i % config.stat_freq == 0:
            logger.info(
                f'Iter: {i}, Loss: {loss.item():.3e}, Depths: {len(out_cls)} Data Loading Time: {d:.3e}, Tot Time: {t:.3e}'
            )
            writer.add_scalar('loss/train', loss.item(), i)

        if i % config.save_freq == 0 and i > 0:
            os.makedirs(os.path.join(config.log_dir, "checkpoints"), exist_ok=True)
            file = os.path.join(config.log_dir, "checkpoints", f"model_iter{i}.pth")

            torch.save(
                {
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'curr_iter': i,
                }, file)

            scheduler.step()
            logger.info(f'LR: {scheduler.get_lr()}')

            net.train()

        if not os.path.exists("/home/maverkiou/zavou"):
            if i % config.val_freq == 0 and i > 0:
                with torch.no_grad():
                    visualize(net, train_dataloader, device, config, test_phase=False, iter=i, writer=writer)
                net.train()


def visualize(net, dataloader, device, config, test_phase=True, iter=-1, writer=None):
    net.eval()
    crit = nn.BCEWithLogitsLoss()
    n_vis = 0
    os.makedirs(os.path.join(config.log_dir, 'samples'), exist_ok=True)

    batch_idx = 0
    for data_dict in dataloader:
        batch_idx += 1

        sin = ME.SparseTensor(
            torch.ones(len(data_dict['coords']), 1),
            data_dict['coords'].int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
        ).to(device)

        # Generate target sparse tensor
        target_key = sin.coords_key

        out_cls, targets, sout, means, log_vars, zs = net(sin, target_key, keep=False if test_phase else True)
        num_layers, BCE = len(out_cls), 0
        losses = []
        for out_cl, target in zip(out_cls, targets):
            curr_loss = crit(out_cl.F.squeeze(),
                             target.type(out_cl.F.dtype).to(device))
            losses.append(curr_loss.item())
            BCE += curr_loss / num_layers

        KLD = -0.5 * torch.mean(
            torch.sum(1 + log_vars.F - means.F.pow(2) - log_vars.F.exp(), 1))
        loss = KLD + BCE

        if writer:
            writer.add_scalar('loss/val', loss.item(), iter)

        # for layer in range(len(out_cls)):
        #     batch_out_coords, batch_out_feats = out_cls[layer].decomposed_coordinates_and_features
        #
        #     for b, (coords, feats) in enumerate(zip(batch_out_coords, batch_out_feats)):
        #
        #         fig = plot_3d_point_cloud(coords[:, 0] / float(config.resolution),
        #                                   coords[:, 1] / float(config.resolution),
        #                                   coords[:, 2] / float(config.resolution),
        #                                   in_u_sphere=True, show=False)
        #         fig.savefig(os.path.join(config.log_dir, 'samples', f'{iter}_{b}_{layer}_pred.png'))
        #         plt.close(fig)
        #
        #         if layer == 0:
        #             fig = plot_3d_point_cloud(data_dict['xyzs'][b].numpy()[:, 0],
        #                                       data_dict['xyzs'][b].numpy()[:, 1],
        #                                       data_dict['xyzs'][b].numpy()[:, 2],
        #                                       in_u_sphere=True, show=False)
        #             fig.savefig(os.path.join(config.log_dir, 'samples', f'{iter}_{b}_real.png'))
        #             plt.close(fig)
        #
        #         n_vis += 1
        #         if n_vis > config.max_visualization * len(out_cls):
        #             break
        batch_coords, batch_feats = sout.decomposed_coordinates_and_features
        point_clouds_real = []
        point_clouds_pred = []
        for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
            if b < 4:
                point_clouds_real.append(data_dict['xyzs'][b])
                point_clouds_pred.append(torch.true_divide(coords, 128))
        render_four_point_clouds(point_clouds_real,
                                 os.path.join(config.log_dir, 'samples', f'{iter}_{batch_idx}_real.png'),
                                 SCREEN_CAMERA_4MODELNET2_FILE)
        render_four_point_clouds(point_clouds_pred,
                                 os.path.join(config.log_dir, 'samples', f'{iter}_{batch_idx}_pred.png'),
                                 SCREEN_CAMERA_4MODELNET2_FILE)

        n_vis += 1
        if n_vis > config.max_visualization:
            return


def export(net, dataloader, device, config, test_phase=True, ):
    net.eval()

    batch_idx = 0
    for data_dict in dataloader:
        batch_idx += 1

        sin = ME.SparseTensor(
            torch.ones(len(data_dict['coords']), 1),
            data_dict['coords'].int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
        ).to(device)

        # Generate target sparse tensor
        target_key = sin.coords_key

        out_cls, targets, sout, means, log_vars, zs = net(sin, target_key, keep=False if test_phase else True)
        zs = zs.decomposed_coordinates_and_features[1]
        for z, file_idx in zip(zs, data_dict['labels']):
            filename = dataloader.dataset.files[file_idx]
            building, component = filename.split("/")[-3], filename.split("/")[-2]
            filename = os.path.join(config.log_dir, config.encodings_dir, 'z_dim_all/as_is', building, f'{component}.npy')
            os.makedirs(os.path.join(config.log_dir, config.encodings_dir, 'z_dim_all/as_is', building), exist_ok=True)
            np.save(filename, z.cpu().numpy().reshape(-1))


if __name__ == '__main__':
    config = parser.parse_args()
    logger = set_logger_file(os.path.join(config.log_dir, 'logs.txt'), logger)

    logger.info(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VAE()
    net.to(device)

    logger.info(net)

    data_dict = {
        "dataset": config.dataset,
        "transforms": ["rotate"],
        "resolution": 128,
        "density": 40000,
        "collate_fn": "collate_pointcloud_fn",
        'train_split_file': config.train_split_file,
        'val_split_file': config.val_split_file
    }
    if config.train_split_file:
        train_dataloader = make_data_loader(
            'train',
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            repeat=True,
            config=config,
            data_dict=data_dict,
            logger=logger
        )
    if config.val_split_file:
        val_dataloader = make_data_loader(
            'val',
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            repeat=False,
            config=config,
            data_dict=data_dict,
            logger=logger)

    if config.train:
        train(net, train_dataloader, train_dataloader, device, config)
    else:
        logger.info(f'Loading weights from {config.weights}')
        checkpoint = torch.load(config.weights)
        net.load_state_dict(checkpoint['state_dict'])

        with torch.no_grad():
            if config.export:
                export(net, val_dataloader, device, config)
            else:
                visualize(net, val_dataloader, device, config)

# note: the input coordinates lie within box [0,1]
