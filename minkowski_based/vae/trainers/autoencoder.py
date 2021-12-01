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
from utils.utils import find_latest_epoch
from utils.visualize import plot_3d_point_cloud, plt


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

            z = self.E(sin)
            out_cls, targets, sout = self.D(z, target_key)

            num_layers, BCE = len(out_cls), 0
            losses = []
            for layer_out_cls, layer_target in zip(out_cls, targets):
                curr_loss = self.reconstruction_loss(layer_out_cls.F.squeeze(),
                                                     layer_target.type(layer_out_cls.F.dtype).to(self.device))
                losses.append(curr_loss.item())
                BCE += curr_loss / num_layers

            batch_loss = BCE


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

        for i in range(8):

            coords, _, _ = dataloader.dataset.__getitem__(i)
            resolution = dataloader.dataset.resolution

            sin = ME.SparseTensor(
                torch.ones(len(coords), 1),
                torch.cat([torch.zeros(len(coords)).reshape((1, -1)), torch.from_numpy(coords).T]).T.int(),
                allow_duplicate_coords=True,
            ).to(self.device)

            target_key = sin.coords_key

            z = self.E(sin)
            out_cls, targets, sout = self.D(z, target_key)

            if len(out_cls) > 0:
                fig = plot_3d_point_cloud(coords[:, 0] / float(resolution),
                                          coords[:, 1] / float(resolution),
                                          coords[:, 2] / float(resolution),
                                          in_u_sphere=True, show=False,
                                          title=str(epoch))
                fig.savefig(join(results_dir, 'samples', f'{epoch:05}_{i}_real.png'))
                plt.close(fig)

                for layer, sout in enumerate(out_cls):
                    coords_out, feats_out = sout.decomposed_coordinates_and_features
                    if len(coords_out) > 0:
                        coords_out = coords_out[0]
                        feats_out = feats_out[0]
                        if len(coords_out) > 0:
                            fig = plot_3d_point_cloud(coords_out.cpu().numpy()[:, 0]/float(resolution),
                                                      coords_out.cpu().numpy()[:, 1]/float(resolution),
                                                      coords_out.cpu().numpy()[:, 2]/float(resolution),
                                                      in_u_sphere=True, show=False,
                                                      title=str(epoch))
                            fig.savefig(join(results_dir, 'samples', f'{epoch:05}_{i}_{layer}_reconstructed.png'))
                            plt.close(fig)
