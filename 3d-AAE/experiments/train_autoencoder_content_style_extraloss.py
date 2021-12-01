import argparse
import json
import logging
import random
from datetime import datetime
from importlib import import_module
from itertools import chain
from os.path import join, exists

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn as nn

from losses.champfer_loss import ChamferLoss
from losses.chamfer_loss import CustomChamferDistance
from utils.pcutil import plot_3d_point_cloud
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging

cudnn.benchmark = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main(config):
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    results_dir = prepare_results_dir(config)
    starting_epoch = find_latest_epoch(results_dir) + 1

    if not exists(join(results_dir, 'config.json')):
        with open(join(results_dir, 'config.json'), mode='w') as f:
            json.dump(config, f)

    setup_logging(results_dir)
    log = logging.getLogger(__name__)

    device = cuda_setup(config['cuda'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    weights_path = join(results_dir, 'weights')

    #
    # Dataset
    #
    from datasets import load_dataset_class
    dset_class = load_dataset_class(config['dataset'])
    train_dataset = dset_class(config['data_dir'], **config["train_dataset"])
    # val_dataset = dset_class(root_dir=config['data_dir'], classes=config['classes'], split='valid')

    log.debug("Selected {} classes. Loaded {} samples.".format(
        'all' if not config["train_dataset"]['classes'] else ','.join(config["train_dataset"]['classes']),
        len(train_dataset)))

    points_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                   shuffle=config["train_dataset"]['shuffle'],
                                   num_workers=config['num_workers'],
                                   drop_last=True, pin_memory=True)

    #
    # Models
    #
    arch = import_module(f"models.{config['arch']}")
    G = arch.Generator(config['model']['G']).to(device)
    CE = arch.Encoder(config['model']['CE']).to(device)
    SE = arch.Encoder(config['model']['SE']).to(device)

    G.apply(weights_init)
    CE.apply(weights_init)
    SE.apply(weights_init)

    if config['reconstruction_loss'].lower() == 'chamfer':
        reconstruction_loss = CustomChamferDistance()
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')

    style_loss_criterion = nn.MSELoss()

    #
    # Optimizers
    #
    EG_optim = getattr(optim, config['optimizer']['EG']['type'])
    EG_optim = EG_optim(chain(CE.parameters(), G.parameters(), SE.parameters()),
                        **config['optimizer']['EG']['hyperparams'])

    if starting_epoch > 1:
        G.load_state_dict(torch.load(join(weights_path, f'{starting_epoch-1:05}_G.pth')))
        CE.load_state_dict(torch.load(join(weights_path, f'{starting_epoch-1:05}_CE.pth')))
        SE.load_state_dict(torch.load(join(weights_path, f'{starting_epoch-1:05}_SE.pth')))

        EG_optim.load_state_dict(torch.load(join(weights_path, f'{starting_epoch-1:05}_EGo.pth')))

    writer = SummaryWriter(log_dir=results_dir)
    global_step = 0

    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()

        G.train()
        CE.train()
        SE.train()

        total_loss = 0.0
        for i, point_data in enumerate(points_dataloader, 1):
            log.debug('-' * 20)
            global_step += 1

            CX, DCX, SX, _ = point_data
            CX = CX.to(device)
            DCX = DCX.to(device)
            SX = SX.to(device)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if CX.size(-1) == 3:
                CX.transpose_(CX.dim() - 2, CX.dim() - 1)

            if SX.size(-1) == 3:
                SX.transpose_(SX.dim() - 2, SX.dim() - 1)

            z_c = CE(CX)  # [batch_size, z_dim]
            z_s = SE(SX)  # [batch_size, z_dim]
            z = torch.cat([z_c, z_s], 1)
            X_rec = G(z)

            X_rec_style_enc = SE(X_rec)

            content_loss = torch.mean(
                config['reconstruction_coef'] *
                reconstruction_loss(DCX + 0.5,
                                    X_rec.permute(0, 2, 1) + 0.5))

            style_loss = style_loss_criterion(z_s, X_rec_style_enc)
            loss = content_loss + style_loss

            EG_optim.zero_grad()
            CE.zero_grad()
            SE.zero_grad()
            G.zero_grad()

            loss.backward()
            total_loss += loss.item()
            EG_optim.step()

            if epoch % config['stat_frequency'] == 0:
                log.debug(f'[{epoch}: ({i})] '
                          f'Loss: {loss.item():.4f} '
                          f'Time: {datetime.now() - start_epoch_time}')
                writer.add_scalar('loss', loss.item(), global_step)
                writer.add_scalar('content_loss', content_loss.item(), global_step)
                writer.add_scalar('style_loss', style_loss.item(), global_step)
                writer.add_scalar('lr', get_lr(EG_optim), global_step)

        if epoch % config['stat_frequency'] == 0:
            log.debug(
                f'[{epoch}/{config["max_epochs"]}] '
                f'Loss: {total_loss / i:.4f} '
                f'Time: {datetime.now() - start_epoch_time}'
            )

        #
        # Save intermediate results
        #
        G.eval()
        CE.eval()
        SE.eval()
        with torch.no_grad():
            z_c = CE(CX)
            z_s = SE(SX)
            z = torch.cat([z_c, z_s], 1)
            X_rec = G(z).data.cpu().numpy()

        if epoch % config['stat_frequency'] == 0:

            for k in range(4):
                fig = plot_3d_point_cloud(CX[k][0].cpu().numpy(), CX[k][1].cpu().numpy(), CX[k][2].cpu().numpy(),
                                          in_u_sphere=True, show=False,
                                          title=str(epoch))
                fig.savefig(
                    join(results_dir, 'samples', f'{epoch:05}_{k}_content.png'))
                plt.close(fig)

            for k in range(4):
                fig = plot_3d_point_cloud(SX[k][0].cpu().numpy(), SX[k][1].cpu().numpy(), SX[k][2].cpu().numpy(),
                                          in_u_sphere=True, show=False,
                                          title=str(epoch))
                fig.savefig(
                    join(results_dir, 'samples', f'{epoch:05}_{k}_style.png'))
                plt.close(fig)

            for k in range(4):
                fig = plot_3d_point_cloud(DCX[k,:,0].cpu().numpy(), DCX[k,:,1].cpu().numpy(), DCX[k,:,2].cpu().numpy(),
                                          in_u_sphere=True, show=False,
                                          title=str(epoch))
                fig.savefig(
                    join(results_dir, 'samples', f'{epoch:05}_{k}_detailed_content.png'))
                plt.close(fig)

            for k in range(4):
                fig = plot_3d_point_cloud(X_rec[k][0], X_rec[k][1], X_rec[k][2],
                                          in_u_sphere=True, show=False,
                                          title=str(epoch))
                fig.savefig(join(results_dir, 'samples',
                                 f'{epoch:05}_{k}_reconstructed.png'))
                plt.close(fig)

        if epoch % config['save_frequency'] == 0:
            torch.save(G.state_dict(), join(weights_path, f'{epoch:05}_G.pth'))
            torch.save(CE.state_dict(), join(weights_path, f'{epoch:05}_CE.pth'))
            torch.save(SE.state_dict(), join(weights_path, f'{epoch:05}_SE.pth'))

            torch.save(EG_optim.state_dict(), join(weights_path, f'{epoch:05}_EGo.pth'))


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    main(config)
