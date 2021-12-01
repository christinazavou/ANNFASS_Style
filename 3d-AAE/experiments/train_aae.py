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
from torch.autograd import grad
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.pcutil import plot_3d_point_cloud
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging

cudnn.benchmark = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


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

    # device = cuda_setup(config['cuda'], config['gpu'])
    device = cuda_setup(config['cuda'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    weights_path = join(results_dir, 'weights')

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        train_dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'])
        val_dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'], split='valid')
    elif dataset_name == 'annfasscomponent':
        from datasets.annfasscomponent import BuildingComponentDataset
        train_dataset = BuildingComponentDataset(root_dir=config['data_dir'],
                                                 classes=config['classes'], n_points=config['n_points'])
        val_dataset = BuildingComponentDataset(root_dir=config['data_dir'], split='val',
                                               classes=config['classes'], n_points=config['n_points'])
    elif dataset_name == 'buildingcomponentdataset2':
        from datasets.buildingcomponent import BuildingComponentDataset2
        train_dataset = BuildingComponentDataset2(txt_file=config['train_txt'],
                                                  n_points=config['n_points'],
                                                  data_root=config['train_data_root'])
        val_dataset = BuildingComponentDataset2(txt_file=config['val_txt'],
                                                n_points=config['n_points'],
                                                data_root=config['val_data_root'])
    elif dataset_name == 'buildingcomponentdataset2withcolor':
        from datasets.buildingcomponent import BuildingComponentDataset2WithColor
        train_dataset = BuildingComponentDataset2WithColor(txt_file=config['train_txt'],
                                                           n_points=config['n_points'],
                                                           data_root=config['train_data_root'])
        val_dataset = BuildingComponentDataset2WithColor(txt_file=config['val_txt'],
                                                         n_points=config['n_points'],
                                                         data_root=config['val_data_root'])
    else:
        raise ValueError(f'Invalid dataset name. Got: `{dataset_name}`')

    points_train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                         shuffle=True,
                                         num_workers=config['num_workers'],
                                         drop_last=True, pin_memory=True)
    points_val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'],
                                       shuffle=False,
                                       num_workers=config['num_workers'],
                                       drop_last=True, pin_memory=True)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    #
    # Models
    #
    arch = import_module(f"models.{config['arch']}")
    G = arch.Generator(config).to(device)
    E = arch.Encoder(config).to(device)
    D = arch.Discriminator(config).to(device)

    G.apply(weights_init)
    E.apply(weights_init)
    D.apply(weights_init)

    if config['reconstruction_loss'].lower() == 'chamfer':
        from losses.champfer_loss import ChamferLoss
        reconstruction_loss = ChamferLoss().to(device)
    elif config['reconstruction_loss'].lower() == 'earth_mover':
        from losses.earth_mover_distance import EMD
        reconstruction_loss = EMD().to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')
    #
    # Float Tensors
    #
    fixed_noise = torch.FloatTensor(config['batch_size'], config['z_size'], 1)
    fixed_noise.normal_(mean=config['normal_mu'], std=config['normal_std'])  # so that in every epoch we use the same to evaluate ... is this really needed?
    noise = torch.FloatTensor(config['batch_size'], config['z_size'])

    fixed_noise = fixed_noise.to(device)
    noise = noise.to(device)

    #
    # Optimizers
    #
    EG_optim = getattr(optim, config['optimizer']['EG']['type'])
    EG_optim = EG_optim(chain(E.parameters(), G.parameters()), **config['optimizer']['EG']['hyperparams'])

    D_optim = getattr(optim, config['optimizer']['D']['type'])
    D_optim = D_optim(D.parameters(), **config['optimizer']['D']['hyperparams'])

    if starting_epoch > 1:
        G.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_G.pth')))
        E.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_E.pth')))
        D.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_D.pth')))

        D_optim.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_Do.pth')))

        EG_optim.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_EGo.pth')))

    writer = SummaryWriter(log_dir=results_dir)
    global_step = 0

    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()

        G.train()
        E.train()
        D.train()

        total_loss_d = 0.0
        total_loss_eg = 0.0
        for i, point_data in enumerate(points_train_dataloader, 1):
            log.debug('-' * 20)
            global_step += 1

            X, _ = point_data
            X = X.to(device)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)

            codes, _, _ = E(X)  # [batch, z_size], _, _
            noise.normal_(mean=config['normal_mu'], std=config['normal_std'])  # [batch, z_size]
            synth_logit = D(codes)  # [batch, 1]
            real_logit = D(noise)  # [batch, 1]
            loss_d = torch.mean(synth_logit) - torch.mean(real_logit)  # real pred should be equal or bigger than synth pred

            alpha = torch.rand(config['batch_size'], 1).to(device)
            differences = codes - noise
            interpolates = noise + alpha * differences
            disc_interpolates = D(interpolates)

            gradients = grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(disc_interpolates).to(device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
            gradient_penalty = ((slopes - 1) ** 2).mean()
            loss_gp = config['gp_lambda'] * gradient_penalty
            ###
            loss_d += loss_gp

            D_optim.zero_grad()
            D.zero_grad()

            loss_d.backward(retain_graph=True)
            total_loss_d += loss_d.item()
            D_optim.step()

            # EG part of training
            X_rec = G(codes)

            loss_e = torch.mean(
                config['reconstruction_coef'] *
                reconstruction_loss(X.permute(0, 2, 1) + 0.5,
                                    X_rec.permute(0, 2, 1) + 0.5))

            synth_logit = D(codes)  # we updated the weights of discriminator and we pass again through encoder and we will now update weights of encoder and generator (if we didnt update encoder already why to calculate synth_logits again? WE MUST HAVE UPDATED THEM BECAUSE IT GIVES DIFFERENT LOGITS)

            loss_g = -torch.mean(synth_logit)  # ??

            loss_eg = loss_e + loss_g
            EG_optim.zero_grad()
            E.zero_grad()
            G.zero_grad()

            loss_eg.backward()
            total_loss_eg += loss_eg.item()
            EG_optim.step()

            if epoch % config['stat_frequency'] == 0:
                log.debug(f'[{epoch}: ({i})] '
                          f'Loss_D: {loss_d.item():.4f} '
                          f'(GP: {loss_gp.item(): .4f}) '
                          f'Loss_EG: {loss_eg.item():.4f} '
                          f'(REC: {loss_e.item(): .4f}) '
                          f'Time: {datetime.now() - start_epoch_time}')
                writer.add_scalar('loss_D', loss_d.item(), global_step)
                writer.add_scalar('loss_EG', loss_eg.item(), global_step)
                writer.add_scalar('REC', loss_e.item(), global_step)
                writer.add_scalar('lr_D', get_lr(D_optim), global_step)
                writer.add_scalar('lr_EG', get_lr(EG_optim), global_step)

        if epoch % config['stat_frequency'] == 0:
            log.debug(
                f'[{epoch}/{config["max_epochs"]}] '
                f'Loss_D: {total_loss_d / i:.4f} '
                f'Loss_EG: {total_loss_eg / i:.4f} '
                f'Time: {datetime.now() - start_epoch_time}'
            )

        #
        # Save intermediate results
        #
        G.eval()
        E.eval()
        D.eval()
        with torch.no_grad():
            fake = G(fixed_noise).data.cpu().numpy()
            codes, _, _ = E(X)
            X_rec = G(codes).data.cpu().numpy()

        if epoch % config['stat_frequency'] == 0:

            for k in range(5):
                fig = plot_3d_point_cloud(X[k][0].cpu().numpy(), X[k][1].cpu().numpy(), X[k][2].cpu().numpy(),
                                          in_u_sphere=True, show=False,
                                          title=str(epoch))
                fig.savefig(
                    join(results_dir, 'samples', f'{epoch:05}_{k}_real.png'))
                plt.close(fig)

            for k in range(5):
                fig = plot_3d_point_cloud(fake[k][0], fake[k][1], fake[k][2],
                                          in_u_sphere=True, show=False,
                                          title=str(epoch))
                fig.savefig(
                    join(results_dir, 'samples', f'{epoch:05}_{k}_fixed.png'))
                plt.close(fig)

            for k in range(5):
                fig = plot_3d_point_cloud(X_rec[k][0],
                                          X_rec[k][1],
                                          X_rec[k][2],
                                          in_u_sphere=True, show=False,
                                          title=str(epoch))
                fig.savefig(join(results_dir, 'samples',
                                 f'{epoch:05}_{k}_reconstructed.png'))
                plt.close(fig)

        if epoch % config['save_frequency'] == 0:
            torch.save(G.state_dict(), join(weights_path, f'{epoch:05}_G.pth'))
            torch.save(D.state_dict(), join(weights_path, f'{epoch:05}_D.pth'))
            torch.save(E.state_dict(), join(weights_path, f'{epoch:05}_E.pth'))

            torch.save(EG_optim.state_dict(),
                       join(weights_path, f'{epoch:05}_EGo.pth'))

            torch.save(D_optim.state_dict(),
                       join(weights_path, f'{epoch:05}_Do.pth'))


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
