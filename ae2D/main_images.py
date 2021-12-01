import argparse
import json
import os

import yaml

from ae2D.models.autoencoder import ConvAutoencoder
from ae2D.models.solver import Solver
from ae2D.utils.datasets import StylisticImagesTestDataLoader, StylisticImagesTrainDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--logs", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--train_txt", type=str)
parser.add_argument("--val_txt", type=str)
parser.add_argument("--test_csv",  type=str)
parser.add_argument("--encodings_dir", type=str)
parser.add_argument("--mode", help="train or encode", default="encode", type=str)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--z_dim", default=16, type=int)
parser.add_argument("--num_channels", default=3, type=int)
parser.add_argument("--batch_norm", default=False, type=str)
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--config_yml", type=str)
args = parser.parse_args()


def str2bool(x):
    return x in ['True', '1']


if args.config_yml:
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    args.__dict__.update(config)
    assert args.train_txt and args.val_txt and args.mode == 'train' or args.test_csv and args.mode == 'encode'

os.makedirs(args.logs, exist_ok=True)
with open(os.path.join(args.logs, "config.yml"), "w") as fout:
    json.dump(args.__dict__, fout, indent=4)

model = ConvAutoencoder(in_channels=args.num_channels, z_dim=args.z_dim, batch_norm=str2bool(args.batch_norm))

if args.mode == 'train':
    train_loader = StylisticImagesTrainDataLoader(args.data_dir, args.train_txt, args.batch_size, args.num_workers)
    val_loader = StylisticImagesTrainDataLoader(args.data_dir, args.val_txt, args.batch_size, args.num_workers)
    Solver(model, args.epochs, args.logs)\
        .run(train_loader, val_loader)
else:
    test_loader = StylisticImagesTestDataLoader(csv_file=args.test_csv, num_workers=args.num_workers)
    Solver(model, args.epochs, args.logs)\
        .generate_encodings(test_loader, args.encodings_dir)
