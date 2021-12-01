import argparse
import json
import logging
import os.path
import random
from importlib import import_module
from os.path import join

import numpy as np
import torch
from torch.distributions import Beta, Bernoulli
from torch.utils.data import DataLoader

from datasets.shapenet import ShapeNetDataset
from utils.util import find_latest_epoch, cuda_setup, setup_logging


def main(eval_config):
    # Load hyperparameters as they were during training
    train_results_path = join(eval_config['results_root'], eval_config['arch'],
                              eval_config['experiment_name'])
    with open(join(train_results_path, 'config.json')) as f:
        train_config = json.load(f)

    random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])

    log = logging.getLogger(__name__)

    weights_path = join(train_results_path, 'weights')
    if "epoch" not in eval_config or eval_config['epoch'] == 0:
        epoch = find_latest_epoch(weights_path)
    else:
        epoch = eval_config['epoch']
    log.debug(f'Starting from epoch: {epoch}')

    encodings_path = join(train_results_path, f'encodings{eval_config["encodings_suffix"]}', f'{epoch:05}_z_e')
    os.makedirs(encodings_path, exist_ok=True)

    device = cuda_setup(eval_config['cuda'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    #
    # Dataset
    #
    from datasets import load_dataset_class
    dset_class = load_dataset_class(eval_config['dataset'])
    test_dataset = dset_class(eval_config['data_dir'], **eval_config["test_dataset"])
    if "train_dataset" in eval_config:
        train_dataset = dset_class(eval_config['data_dir'], **eval_config["train_dataset"])

    #
    # Models
    #
    arch = import_module(f"models.{eval_config['arch']}")
    E = arch.Encoder(train_config).to(device)

    #
    # Load saved state
    #
    E.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_E.pth')))

    E.eval()

    if "train_dataset" in eval_config:
        train_test_sets = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    else:
        train_test_sets = test_dataset

    data_loader = DataLoader(train_test_sets, batch_size=eval_config['batch_size'],
                             shuffle=False, num_workers=eval_config['num_workers'],
                             drop_last=False, pin_memory=True)

    with torch.no_grad():

        for X_batch, X_batch_files in data_loader:
            X_batch = X_batch.to(device)

            z_e_batch = E(X_batch.transpose(1, 2))
            if isinstance(z_e_batch, tuple):
                z_e_batch = z_e_batch[0]

            for z_e, X_file in zip(z_e_batch, X_batch_files):
                filename = os.path.basename(X_file)
                building = filename.split("_style_mesh_")[0]
                component = filename.split("_style_mesh_")[1].replace("_detail", "").replace(".ply", "")
                os.makedirs(os.path.join(encodings_path, building), exist_ok=True)
                np.save(join(encodings_path, building, f"{component}.npy"), z_e.cpu().numpy())


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='File path for evaluation config')
    args = parser.parse_args()

    evaluation_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            evaluation_config = json.load(f)
    assert evaluation_config is not None

    main(evaluation_config)
