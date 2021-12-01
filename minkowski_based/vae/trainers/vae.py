from importlib import import_module
from itertools import chain
from os.path import join
from datetime import datetime

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import MinkowskiEngine as ME

from trainers import get_lr
from losses.chamfer_loss import CustomChamferDistance
from utils import find_latest_epoch, SCREEN_CAMERA_4MODELNET2_FILE
from utils.visualize import plot_3d_point_cloud, plt
from utils.open3d_vis import render_four_point_clouds


class Trainer:

    def __init__(self, config, device):
        arch = import_module(f"models.{config['arch']}")
        self.device = device
        self.E = arch.Encoder().to(device)
        self.D = arch.Decoder().to(device)
        # assert config['trainer']['reconstruction_loss'].lower() == 'chamfer'
        # self.reconstruction_loss = CustomChamferDistance()
        assert config['trainer']['reconstruction_loss'].lower() == 'bce'
        self.reconstruction_loss = torch.nn.BCEWithLogitsLoss()

        ED_optim_class = getattr(optim, config['optimizer']['ED']['type'])
        self.ED_optim = ED_optim_class(chain(self.E.parameters(), self.D.parameters()),
                                       **config['optimizer']['ED']['hyperparams'])

        scheduler_class = getattr(lr_scheduler, config['scheduler']['type'])
        self.scheduler = scheduler_class(self.ED_optim,
                                         **config['scheduler']['hyperparams'])

    def run(self, results_dir, dataloader, log, config):

        weights_path = join(results_dir, 'weights')
        starting_epoch = find_latest_epoch(results_dir) + 1

        print(f"Will start/resume training with {len(dataloader.dataset)} training examples..")
        num_batches = len(dataloader.dataset) / dataloader.batch_size
        global_step = (starting_epoch - 1) * int(num_batches)

        if starting_epoch > 1:
            print(f"Loaded weights and starting from epoch {starting_epoch}")
            self.D.load_state_dict(torch.load(join(weights_path, f'{starting_epoch-1:05}_D.pth')))
            self.E.load_state_dict(torch.load(join(weights_path, f'{starting_epoch-1:05}_E.pth')))
            self.ED_optim.load_state_dict(torch.load(join(weights_path, f'{starting_epoch-1:05}_EDo.pth')))

        writer = SummaryWriter(log_dir=results_dir)

        for epoch in range(starting_epoch, config['max_epochs'] + 1):
            start_epoch_time = datetime.now()

            last_sin, out_cls, i, total_loss, global_step = self.train_one_epoch(config, dataloader,
                                                                                 epoch, global_step,
                                                                                 log, start_epoch_time,
                                                                                 writer)

            if epoch % config['stat_frequency'] == 0:
                log.debug(
                    f'[{epoch}/{config["max_epochs"]}] '
                    f'Loss: {total_loss / i:.4f} '
                    f'Time: {datetime.now() - start_epoch_time}'
                )

            self.evaluate_one_epoch(config, epoch, results_dir, weights_path, dataloader)

    def evaluate_one_epoch(self, config, epoch, results_dir, weights_path, dataloader):

        # self.E.eval()
        # self.D.eval()
        #
        # with torch.no_grad():
        #     pass

        if epoch % config['stat_frequency'] == 0:
            self.save_intermediate_results(epoch, results_dir, dataloader)

        if epoch % config['save_frequency'] == 0:
            torch.save(self.E.state_dict(), join(weights_path, f'{epoch:05}_E.pth'))
            torch.save(self.D.state_dict(), join(weights_path, f'{epoch:05}_D.pth'))
            torch.save(self.ED_optim.state_dict(), join(weights_path, f'{epoch:05}_EDo.pth'))

    def train_one_epoch(self, config, dataloader, epoch, global_step, log, start_epoch_time, writer):

        self.E.train()
        self.D.train()

        epoch_loss = []

        for i, batch_data in enumerate(dataloader, 1):

            self.ED_optim.zero_grad()
            self.E.zero_grad()  # redundant if all E.params are in ED_optim
            self.D.zero_grad()  # redundant if all D.params are in ED_optim

            log.debug('-' * 20)
            global_step += 1

            batch_coords = batch_data['coords']
            batch_xyzs = batch_data['xyzs']
            batch_labels = batch_data['labels']

            sin = ME.SparseTensor(
                torch.ones(len(batch_coords), 1),
                batch_coords.int(),
                allow_duplicate_coords=True,
            ).to(self.device)

            target_key = sin.coords_key

            means, log_vars = self.E(sin)
            z = means + torch.exp(0.5 * log_vars.F) * torch.randn_like(log_vars.F)  # TRAIN TIME
            out_cls, targets, sout = self.D(z, target_key)

            num_layers, BCE = len(out_cls), 0
            losses = []
            for layer_out_cls, layer_target in zip(out_cls, targets):
                curr_loss = self.reconstruction_loss(layer_out_cls.F.squeeze(),
                                                     layer_target.type(layer_out_cls.F.dtype).to(self.device))
                losses.append(curr_loss.item())
                BCE += curr_loss / num_layers

            KLD = -0.5 * torch.mean(torch.mean(1 + log_vars.F - means.F.pow(2) - log_vars.F.exp(), 1))

            batch_loss = BCE + KLD

            batch_loss.backward()
            self.ED_optim.step()

            epoch_loss += [batch_loss.item()]

            if epoch % config['stat_frequency'] == 0:
                log.debug(f'[{epoch}: ({i})] '
                          f'Loss: {batch_loss.item():.4f} '
                          f'Time: {datetime.now() - start_epoch_time}')
                writer.add_scalar('loss', batch_loss.item(), global_step)
                writer.add_scalar('lr', get_lr(self.ED_optim), global_step)
        return sin, out_cls, i, sum(epoch_loss)/len(epoch_loss), global_step

    def save_intermediate_results(self, epoch, results_dir, dataloader):

        self.E.eval()
        self.D.eval()

        n_vis = 0

        batch_idx = 0
        for data_dict in dataloader:
            batch_idx += 1

            sin = ME.SparseTensor(
                torch.ones(len(data_dict['coords']), 1),
                data_dict['coords'].int(),
                allow_duplicate_coords=True,  # for classification, it doesn't matter
            ).to(self.device)

            # Generate target sparse tensor
            target_key = sin.coords_key

            means, log_vars = self.E(sin)
            z = means  # TEST TIME
            out_cls, targets, sout = self.D(z, target_key)

            batch_coords, batch_feats = sout.decomposed_coordinates_and_features
            point_clouds_real = []
            point_clouds_pred = []
            for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
                if b < 4:
                    point_clouds_real.append(data_dict['xyzs'][b])
                    point_clouds_pred.append(coords)
            render_four_point_clouds(point_clouds_real,
                                     join(results_dir, f'{epoch}_{batch_idx}_real.png'),
                                     SCREEN_CAMERA_4MODELNET2_FILE)
            render_four_point_clouds(point_clouds_pred,
                                     join(results_dir, f'{epoch}_{batch_idx}_pred.png'),
                                     SCREEN_CAMERA_4MODELNET2_FILE)

            n_vis += 1
            if n_vis > 10:
                return
