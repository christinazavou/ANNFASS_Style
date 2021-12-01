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
from utils.utils import find_latest_epoch, dotdict
from utils.visualize import plot_3d_point_cloud, plt
import models.hrnet_ae as hrnet_ae


HRNET_AE_MODELS = [getattr(hrnet_ae, a) for a in dir(hrnet_ae) if 'Net' in a]


def load_model_class(name):
    mdict = {model.__name__: model for model in HRNET_AE_MODELS}
    if name not in mdict:
        print('Invalid model index. Options are:')
        # Display a list of valid dataset names
        for model in HRNET_AE_MODELS:
            print('\t* {}'.format(model.__name__))
        raise ValueError(f'Dataset {name} not defined')
    DatasetClass = mdict[name]
    return DatasetClass


class Trainer:

    def __init__(self, config, device):
        model_class = load_model_class(config['model']['name'])
        self.model = model_class(3, 3, dotdict(config['model']), 3).to(device)
        self.device = device
        assert config['trainer']['reconstruction_loss'].lower() == 'chamfer'
        self.reconstruction_loss = CustomChamferDistance()

        ED_optim_class = getattr(optim, config['optimizer']['ED']['type'])
        self.ED_optim = ED_optim_class(self.model.parameters(),
                                       **config['optimizer']['ED']['hyperparams'])

        scheduler_class = getattr(lr_scheduler, config['scheduler']['type'])
        self.scheduler = scheduler_class(self.ED_optim,
                                         **config['scheduler']['hyperparams'])

    def run(self, results_dir, dataloader, log, config):

        weights_path = join(results_dir, 'weights')
        starting_epoch = find_latest_epoch(results_dir) + 1

        if starting_epoch > 1:
            print(f"Loaded wweights and starting from epoch {starting_epoch}")
            self.model.load_state_dict(torch.load(join(weights_path, f'{starting_epoch-1:05}_ED.pth')))
            self.ED_optim.load_state_dict(torch.load(join(weights_path, f'{starting_epoch-1:05}_EDo.pth')))

        writer = SummaryWriter(log_dir=results_dir)
        global_step = 0

        for epoch in range(starting_epoch, config['max_epochs'] + 1):
            start_epoch_time = datetime.now()

            last_sin, last_sout, i, total_loss = self.train_one_epoch(config, dataloader,
                                                                      epoch, global_step,
                                                                      log, start_epoch_time,
                                                                      writer)

            if epoch % config['stat_frequency'] == 0:
                log.debug(
                    f'[{epoch}/{config["max_epochs"]}] '
                    f'Loss: {total_loss / i:.4f} '
                    f'Time: {datetime.now() - start_epoch_time}'
                )

            self.evaluate_one_epoch(last_sin, last_sout, i, config, epoch, results_dir, weights_path)

    def evaluate_one_epoch(self, sin, sout, i, config, epoch, results_dir, weights_path):

        # self.E.eval()
        # self.D.eval()
        #
        # with torch.no_grad():
        #     pass

        if epoch % config['stat_frequency'] == 0:
            self.save_intermediate_results(sin, sout, i, epoch, results_dir)

        if epoch % config['save_frequency'] == 0:
            torch.save(self.model.state_dict(), join(weights_path, f'{epoch:05}_ED.pth'))
            torch.save(self.ED_optim.state_dict(), join(weights_path, f'{epoch:05}_EDo.pth'))

    def train_one_epoch(self, config, dataloader, epoch, global_step, log, start_epoch_time, writer):

        self.model.train()

        total_loss = 0.0
        for i, batch_data in enumerate(dataloader, 1):
            log.debug('-' * 20)
            global_step += 1

            sin = ME.SparseTensor(
                batch_data['feats'].float(),
                batch_data['coords'].int(),
                allow_duplicate_coords=True,
            ).to(self.device)

            target = batch_data['feats']

            sout = self.model(sin)

            sout_coords, sout_feats = sout.decomposed_coordinates_and_features

            loss = self.reconstruction_loss(sout.F.squeeze(),
                                            target.to(self.device))

            self.ED_optim.zero_grad()
            self.model.zero_grad()

            loss.backward()
            total_loss += loss.item()
            self.ED_optim.step()

            if epoch % config['stat_frequency'] == 0:
                log.debug(f'[{epoch}: ({i})] '
                          f'Loss: {loss.item():.4f} '
                          f'Time: {datetime.now() - start_epoch_time}')
                writer.add_scalar('loss', loss.item(), global_step)
                writer.add_scalar('lr', get_lr(self.ED_optim), global_step)
        return sin, sout, i, total_loss

    def save_intermediate_results(self, sin, sout, i, epoch, results_dir):
        if len(sout) > 0:
            batch_coords_in, batch_feats_in = sin.decomposed_coordinates_and_features
            batch_coords_out, batch_feats_out = sout.decomposed_coordinates_and_features
            for b, (coords_in, coords_out) in enumerate(zip(batch_coords_in, batch_coords_out)):
                if len(coords_out) > 0:

                    fig = plot_3d_point_cloud(coords_in.cpu().numpy()[:, 0]/100.,
                                              coords_in.cpu().numpy()[:, 1]/100.,
                                              coords_in.cpu().numpy()[:, 2]/100.,
                                              in_u_sphere=True, show=False,
                                              title=str(epoch))
                    fig.savefig(join(results_dir, 'samples', f'{epoch:05}_{i}_{b}_real.png'))
                    plt.close(fig)

                    fig = plot_3d_point_cloud(coords_out.cpu().numpy()[:, 0]/100.,
                                              coords_out.cpu().numpy()[:, 1]/100.,
                                              coords_out.cpu().numpy()[:, 2]/100.,
                                              in_u_sphere=True, show=False,
                                              title=str(epoch))
                    fig.savefig(join(results_dir, 'samples', f'{epoch:05}_{i}_{b}_reconstructed.png'))
                    plt.close(fig)
