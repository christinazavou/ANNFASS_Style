import os
import time

import numpy as np
import argparse
from importlib import import_module
import yaml


def str2bool(x):
    return True if x in ['1', 1, 'True', 'true', 'Yes', 'y', 'yes'] else False


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=20, type=int, help="Epoch to train [20]")
parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int, help="Iteration to train. Either epoch or iteration need to be zero [0]")

parser.add_argument("--g_steps", type=int, dest="g_steps", default=1, help="generator steps per training step")
parser.add_argument("--d_steps", type=int, dest="d_steps", default=1, help="discriminator steps per training step")
parser.add_argument("--r_steps", type=int, dest="r_steps", default=4, help="reconstruction steps per training step")

# parser.add_argument("--patch_num", type=int, dest="patch_num", default=4, help="patch_num")
# parser.add_argument("--patch_size", type=int, dest="patch_size", default=32, help="patch_size")
# parser.add_argument("--patch_stride", type=int, dest="patch_stride", default=16, help="patch_stride")
parser.add_argument("--patch_factor", type=int, dest="patch_factor", default=2, help="patch_factor")
parser.add_argument("--stride_factor", type=int, dest="stride_factor", default=1, help="stride_factor")
parser.add_argument("--num_triplets", type=int, dest="num_triplets", default=128, help="num_triplets")
parser.add_argument("--num_pairs", type=int, dest="num_pairs", default=64, help="num_pairs")

# parser.add_argument("--cycle_factor", type=float, dest="cycle_factor", default=10, help="cycle latent loss factor")
# parser.add_argument("--cycle_loss", type=str, dest="cycle_loss", default='L1', help="cycle loss type")
# parser.add_argument("--recon_loss", type=str, dest="recon_loss", default='MSE', help="reconstruction loss type")
# parser.add_argument("--nt_xent_factor", type=float, dest="nt_xent_factor", default=1., help="nt_xent_factor")
# parser.add_argument("--tau", type=float, dest="tau", default=0.5, help="tau")
# parser.add_argument("--style_batch", type=int, dest="style_batch", default=4, help="style_batch")
parser.add_argument("--save_iter", type=int, dest="save_iter", default=1000, help="Save every N iterations")
parser.add_argument("--log_iter", type=int, dest="log_iter", default=500, help="Log every N iterations")
parser.add_argument("--model_module", type=str, dest="model_module", default='runners_open3d.mymodelAE', help="The name of file")
parser.add_argument("--lr", type=float, dest="lr", default=0.0001, help="learning rate")
parser.add_argument("--se_lr", type=float, dest="se_lr", default=0.0001, help="se_lr")
parser.add_argument("--style_dim", type=int, dest="style_dim", default=8, help="style dimensionality")
parser.add_argument("--disc_dim", type=int, dest="disc_dim", default=32, help="discriminator channels in 1st conv")
parser.add_argument("--gen_dim", type=int, dest="gen_dim", default=32, help="generator channels in 1st conv")
parser.add_argument("--margin", type=float, dest="margin", default=0.1, help="margin")
parser.add_argument("--kernel", type=int, dest="kernel", default=5, help="kernel in style encoder")
parser.add_argument("--dilation", type=str2bool, dest="dilation", default=True, help="dilation in style encoder")
parser.add_argument("--pooling", type=str, dest="pooling", default='max', help="pooling method (avg/max) in style encoder")
parser.add_argument("--batch_size", type=int, dest="batch_size", default=1, help="batch_size")

parser.add_argument("--clamp_num", type=float, dest="clamp_num", default=0.01, help="clamp_num")
parser.add_argument("--optim", type=str, dest="optim", default='Adam', help="optimizer [Adam, SGD]")
parser.add_argument("--with_norm", action="store_true", dest="with_norm", default=False, help="True for normalizing style code [False]")
parser.add_argument("--norm_type", type=str, dest="norm_type", default='unit_length', help="norm_type [unit_length, group_norm]")
parser.add_argument("--weight_init", action="store_true", default=False, dest="init_weights", )
parser.add_argument("--weight_decay", type=float, default=0, dest="weight_decay", help="0 or 1e-4")
parser.add_argument("--use_wc", action="store_true", default=False, dest="use_wc", )
parser.add_argument("--group_norm", action="store_true", default=False, dest="group_norm", help="Whether to use group norm layers")

parser.add_argument("--any_share_type", dest="any_share_type", default=3, type=int)

parser.add_argument("--alpha", action="store", dest="alpha", default=0.5, type=float, help="Parameter alpha [0.5]")
parser.add_argument("--beta", action="store", dest="beta", default=10.0, type=float, help="Parameter beta [10.0]")
parser.add_argument("--gamma", action="store", dest="gamma", default=1.0, type=float, help="Parameter gamma [1.0]")
parser.add_argument("--delta", action="store", dest="delta", default=1.0, type=float, help="Parameter delta [1.0]")
parser.add_argument("--adain_alpha", action="store", dest="adain_alpha", default=1.0, type=float, help="Adain alpha [1.0]")

# parser.add_argument("--triplet_splits",  type=str, dest="triplet_splits", )
# parser.add_argument("--buildings_dir", type=str, dest="buildings_dir", help="The name of file")
parser.add_argument("--data_filename", type=str, dest="filename", default='model.binvox', help="The name of file")
parser.add_argument("--style_filename", type=str, dest="filename", default='model.binvox', help="The name of file")
parser.add_argument("--val_filename", type=str, dest="val_filename", default='model.binvox', help="The name of file")
parser.add_argument("--datapath", action="store", dest="datapath", help="The name of content dataset")
parser.add_argument("--stylepath", default=None, type=str, help="The name of style dataset")
parser.add_argument("--valpath", default=None, type=str, help="The name of val dataset")
parser.add_argument("--data_cache_dir", dest="data_cache_dir", default=None, type=str, help="Dir for cached dataset")
parser.add_argument("--style_cache_dir", dest="style_cache_dir", default=None, type=str, help="Dir for cached dataset")
parser.add_argument("--val_cache_dir", dest="val_cache_dir", default=None, type=str, help="Dir for cached dataset")
parser.add_argument("--data_dir", action="store", dest="data_dir", help="Root directory of dataset")
parser.add_argument("--style_dir", action="store", dest="style_dir", help="Root directory of dataset")
parser.add_argument("--val_dir", action="store", dest="val_dir", help="Root directory of dataset")

parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples [samples]")
parser.add_argument("--export_dir", action="store", dest="export_dir", default="./encodings/", help="Directory name to save the style_encodings [samples]")
parser.add_argument("--test_fig_3_dir", action="store", dest="test_fig_3_dir", default="./test_fig_3_dir/", help="Directory name to save figure 3 results [samples]")
parser.add_argument("--style_codes_dir", action="store", dest="style_codes_dir", default="./style_codes_dir/", help="Directory name to save style_codes [samples]")

parser.add_argument("--input_size", action="store", dest="input_size", default=64, type=int, help="Input voxel size [64]")
parser.add_argument("--output_size", action="store", dest="output_size", default=256, type=int, help="Output voxel size [256]")
parser.add_argument("--asymmetry", action="store_true", dest="asymmetry", default=False, help="True for training on asymmetric shapes [False]")

parser.add_argument("--layer",  dest="layer", default="all", type=str)
parser.add_argument("--style_encoder",  dest="style_encoder", default="", type=str, help="type of shared encoder")

parser.add_argument("--finetune", action="store_true", dest="finetune", default=False, help="True for fine-tuning discriminator with triplet loss [False]")
parser.add_argument("--pct_suffix",  dest="pct_suffix", default="", type=str, help="few shot pct")

parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for trainin [False]")
parser.add_argument("--style_codes", action="store_true", dest="style_codes", default=False, help="True for rough style_codes [False]")
parser.add_argument("--test_fig_3", action="store_true", dest="test_fig_3", default=False, help="True for rough test_fig_3 [False]")
parser.add_argument("--style_indices", dest="style_indices", help="separated by comma and space")
parser.add_argument("--export", action="store_true", dest="export", default=False, help="True for encoding export [False]")
parser.add_argument("--debug", action="store_true", dest="debug", default=False, help="True for debugging training [False]")
parser.add_argument("--visualize_styles", action="store_true", default=False, dest="visualize_styles", )
parser.add_argument("--visualize_contents", action="store_true", default=False, dest="visualize_contents", )
parser.add_argument("--visualize_validation", action="store_true", default=False, dest="visualize_validation", )

parser.add_argument("--output_for_eval_dir", dest="output_for_eval_dir", default="output_for_eval")
parser.add_argument("--output_for_FID_dir", dest="output_for_FID_dir", default="output_for_FID")
parser.add_argument("--unique_patches_dir", dest="unique_patches_dir", default="unique_patches")
parser.add_argument("--eval_output_dir", dest="eval_output_dir", default="eval_output")
parser.add_argument("--cls_dir", dest="cls_dir", default="./")
parser.add_argument("--prepvox", action="store_true", dest="prepvox", default=False, help="True for preparing voxels for evaluating IOU, LP and Div [False]")
parser.add_argument("--prepvoxstyle", action="store_true", dest="prepvoxstyle", default=False, help="True for preparing voxels for evaluating IOU, LP and Div [False]")
parser.add_argument("--evalvox", action="store_true", dest="evalvox", default=False, help="True for evaluating IOU, LP and Div [False]")

parser.add_argument("--prepimg", action="store_true", dest="prepimg", default=False, help="True for preparing rendered views for evaluating Cls_score [False]")
parser.add_argument("--prepimgreal", action="store_true", dest="prepimgreal", default=False, help="True for preparing rendered views of all content shapes (as real) for evaluating Cls_score [False]")
parser.add_argument("--evalimg", action="store_true", dest="evalimg", default=False, help="True for evaluating Cls_score [False]")

parser.add_argument("--prepFID", action="store_true", dest="prepFID", default=False, help="True for preparing voxels for evaluating FID [False]")
parser.add_argument("--prepFIDmodel", action="store_true", dest="prepFIDmodel", default=False, help="True for training a classifier for evaluating FID [False]")
parser.add_argument("--prepFIDreal", action="store_true", dest="prepFIDreal", default=False, help="True for computing the mean and sigma vectors (real) for evaluating FID [False]")
parser.add_argument("--evalFID", action="store_true", dest="evalFID", default=False, help="True for evaluating FID [False]")

parser.add_argument("--ui", action="store_true", dest="ui", default=False, help="launch a UI for latent space exploration [False]")

parser.add_argument("--gpu", dest="gpu", type=int, default="0", help="to use which GPU [0]")

parser.add_argument("--config_yml", dest="config_yml", default=None, help="use this yml file instead")

FLAGS = parser.parse_args()


if FLAGS.config_yml:
    config = yaml.load(open(FLAGS.config_yml), Loader=yaml.FullLoader)
    FLAGS.__dict__.update(config)

print(f"FLAGS:\n\n{FLAGS}\n\n")

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu


model_module = import_module(FLAGS.model_module)

if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

im_ae = model_module.IM_AE(FLAGS)

if FLAGS.train:
    im_ae.train(FLAGS)
else:

    if FLAGS.export:
        im_ae.export(FLAGS)
    elif FLAGS.style_codes:
        im_ae.test_style_codes(FLAGS)
    elif FLAGS.test_fig_3:
        im_ae.test_fig_3(FLAGS)
    elif FLAGS.finetune:
        im_ae.fine_tune_discr_with_rank(FLAGS)
    else:

        if FLAGS.prepvox:
            im_ae.prepare_voxel_for_eval(FLAGS)

        if FLAGS.prepvoxstyle:
            from evaluation import dp2evalAE
            im_ae.prepare_voxel_style(FLAGS)
            dp2evalAE.precompute_unique_patches_per_style(im_ae.device, im_ae.style_set, FLAGS)

        if FLAGS.evalvox:
            from evaluation import dp2evalAE
            dp2evalAE.eval_IOU(im_ae.device, im_ae.style_set, im_ae.dset, FLAGS)
            dp2evalAE.eval_LP_Div_IOU(im_ae.device, im_ae.style_set, im_ae.dset, FLAGS)
            dp2evalAE.eval_LP_Div_Fscore(im_ae.device, im_ae.style_set, im_ae.dset, FLAGS)


        if FLAGS.prepFID:
            im_ae.prepare_voxel_for_FID(FLAGS)

        if FLAGS.prepFIDreal:
            from evaluation import dp2evalFID
            dp2evalFID.compute_FID_for_real(im_ae.device, im_ae.style_set, im_ae.dset, FLAGS)

        if FLAGS.evalFID:
            from evaluation import dp2evalFID
            dp2evalFID.eval_FID(im_ae.device, FLAGS)
