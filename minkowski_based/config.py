import argparse


def str2opt(arg):
  assert arg in ['SGD', 'Adam']
  return arg


def str2scheduler(arg):
  assert arg in ['StepLR', 'PolyLR', 'ExpLR', 'SquaredLR', 'ReduceLROnPlateau']
  return arg


def str2export_feat_mode(arg):
  assert arg in ['max', 'avg', 'sum']
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


def str2list(l):
  return [int(i) for i in l.split(',')]


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


arg_lists = []
parser = argparse.ArgumentParser()

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='ResUNet14', help='Model name')
net_arg.add_argument('--conv1_kernel_size', type=int, default=5, help='First layer conv kernel size')
net_arg.add_argument('--weights', type=str, default='None', help='Saved weights to load')
net_arg.add_argument('--ps_weights', type=str, default='None', help='Saved weights from ps model to load in style model')
net_arg.add_argument(
    '--weights_for_inner_model',
    type=str2bool,
    default=False,
    help='Weights for model inside a wrapper')
net_arg.add_argument(
    '--dilations', type=str2list, default='1,1,1,1', help='Dilations used for ResNet or DenseNet')

# Wrappers
net_arg.add_argument('--wrapper_type', default='None', type=str, help='Wrapper on the network')
net_arg.add_argument(
    '--wrapper_region_type',
    default=1,
    type=int,
    help='Wrapper connection types 0: hypercube, 1: hypercross, (default: 1)')
net_arg.add_argument('--wrapper_kernel_size', default=3, type=int, help='Wrapper kernel size')
net_arg.add_argument(
    '--wrapper_lr',
    default=1e-1,
    type=float,
    help='Used for freezing or using small lr for the base model, freeze if negative')

# Meanfield arguments
net_arg.add_argument(
    '--meanfield_iterations', type=int, default=10, help='Number of meanfield iterations')
net_arg.add_argument('--crf_spatial_sigma', default=1, type=int, help='Trilateral spatial sigma')
net_arg.add_argument(
    '--crf_chromatic_sigma', default=12, type=int, help='Trilateral chromatic sigma')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--lr', type=float, default=1e-2)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--param_histogram_freq', type=int, default=100)
opt_arg.add_argument('--save_param_histogram', type=str2bool, default=False)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.02)

# Scheduler
opt_arg.add_argument('--scheduler', type=str2scheduler, default='StepLR')
opt_arg.add_argument('--max_iter', type=int, default=6e4)
opt_arg.add_argument('--step_size', type=int, default=2e4)
opt_arg.add_argument('--step_gamma', type=float, default=0.1)
opt_arg.add_argument('--poly_power', type=float, default=0.9)
opt_arg.add_argument('--exp_gamma', type=float, default=0.95)
opt_arg.add_argument('--exp_step_size', type=float, default=445)
opt_arg.add_argument('--cooldown', type=int, default=10)
opt_arg.add_argument('--r_factor', type=float, default=0.1)

# Directories
dir_arg = add_argument_group('Directories')
dir_arg.add_argument('--log_dir', type=str, default='outputs/default')
dir_arg.add_argument('--data_dir', type=str, default='data')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='ScannetVoxelization2cmDataset')
data_arg.add_argument('--temporal_dilation', type=int, default=30)
data_arg.add_argument('--temporal_numseq', type=int, default=3)
data_arg.add_argument('--point_lim', type=int, default=-1)
data_arg.add_argument('--pre_point_lim', type=int, default=-1)
data_arg.add_argument('--batch_size', type=int, default=16)
data_arg.add_argument('--val_batch_size', type=int, default=1)
data_arg.add_argument('--test_batch_size', type=int, default=1)
data_arg.add_argument('--cache_data', type=str2bool, default=False)
data_arg.add_argument(
    '--num_workers', type=int, default=0, help='num workers for train/test dataloader')
data_arg.add_argument('--num_val_workers', type=int, default=0, help='num workers for val dataloader')
data_arg.add_argument('--ignore_label', type=int, default=255)
data_arg.add_argument('--return_transformation', type=str2bool, default=False)
data_arg.add_argument('--ignore_duplicate_class', type=str2bool, default=False)
data_arg.add_argument('--prefetch_data', type=str2bool, default=False)
data_arg.add_argument('--partial_crop', type=float, default=0.)
data_arg.add_argument('--train_limit_numpoints', type=int, default=0)

# Point Cloud Dataset

data_arg.add_argument(
    '--synthia_path',
    type=str,
    default='/home/chrischoy/datasets/Synthia/Synthia4D',
    help='Point Cloud dataset root dir')
# For temporal sequences
data_arg.add_argument(
    '--synthia_camera_path', type=str, default='/home/chrischoy/datasets/Synthia/%s/CameraParams/')
data_arg.add_argument('--synthia_camera_intrinsic_file', type=str, default='intrinsics.txt')
data_arg.add_argument(
    '--synthia_camera_extrinsics_file', type=str, default='Stereo_Right/Omni_F/%s.txt')
data_arg.add_argument('--temporal_rand_dilation', type=str2bool, default=False)
data_arg.add_argument('--temporal_rand_numseq', type=str2bool, default=False)

data_arg.add_argument(
    '--scannet_path',
    type=str,
    default='/home/chrischoy/datasets/scannet/scannet_preprocessed',
    help='Scannet online voxelization dataset root dir')

data_arg.add_argument(
    '--stanford3d_path',
    type=str,
    default='/home/chrischoy/datasets/Stanford3D',
    help='Stanford precropped dataset root dir')

data_arg.add_argument(
    '--buildnet_path',
    type=str,
    default='',
    help='Buildnet online voxelization dataset root dir')

data_arg.add_argument(
    '--partnet_path',
    type=str,
    default='',
    help='Partnet online voxelization dataset root dir')
data_arg.add_argument(
    '--partnet_category',
    type=str,
    default='',
    help='Specify Partnet category')

data_arg.add_argument(
    '--stylenet_path',
    type=str,
    default='',
    help='Stylenet online voxelization dataset root dir')

data_arg.add_argument(
    '--vaebcecomponent_path',
    type=str,
    default='',
    help='vaebcecomponent_path')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--stat_freq', type=int, default=40, help='print frequency')
train_arg.add_argument('--test_stat_freq', type=int, default=100, help='print frequency')
train_arg.add_argument('--save_freq', type=int, default=1000, help='save frequency')
train_arg.add_argument('--val_freq', type=int, default=1000, help='validation frequency')
train_arg.add_argument(
    '--empty_cache_freq', type=int, default=1, help='Clear pytorch cache frequency')
train_arg.add_argument('--train_phase', type=str, default='train', help='Dataset for training')
train_arg.add_argument('--val_phase', type=str, default='val', help='Dataset for validation')
train_arg.add_argument(
    '--overwrite_weights', type=str2bool, default=True, help='Overwrite checkpoint during training')
train_arg.add_argument(
    '--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
train_arg.add_argument(
    '--resume_optimizer',
    default=True,
    type=str2bool,
    help='Use checkpoint optimizer states when resume training')
train_arg.add_argument('--eval_upsample', type=str2bool, default=False)
train_arg.add_argument(
    '--lenient_weight_loading',
    type=str2bool,
    default=False,
    help='Weights with the same size will be loaded')
train_arg.add_argument('--weighted_cross_entropy', type=str2bool, default=False, help='Use weighted cross entropy')
train_arg.add_argument('--chamfer_loss', type=str2bool, default=False, help='Use chamfer reconstruction loss')
train_arg.add_argument('--loss_factor', type=float, default=10., help='Chamfer factor loss')
train_arg.add_argument('--exp_loss', type=str2bool, default=False, help='Chamfer loss exp factor')
train_arg.add_argument('--class_balanced_loss', type=str2bool, default=False, help='Use class-balanced loss')
train_arg.add_argument('--class_balanced_beta', type=float, default=0.999999, help='Class-balanced loss')
train_arg.add_argument('--inv_freq_class_weight', type=str2bool, default=False, help='Use inverse frequency class weights')
train_arg.add_argument('--class_weights', type=str, default="", help='Path of class weights to be loaded')
train_arg.add_argument('--focal_loss', type=str2bool, default=False, help='Use focal loss')
train_arg.add_argument('--weighted_focal_loss', type=str2bool, default=False, help='Use weighted focal loss')
train_arg.add_argument('--input_feat', type=str, default='rgb', help='Input features')
train_arg.add_argument('--normalize_coords', type=str2bool, default=False, help='Normalize coordinates')
train_arg.add_argument('--normalize_method', type=str, default="sphere", help='Methot do normalize coordinates')
train_arg.add_argument('--normalize_y', type=str2bool, default=False, help='Whether to normalize reconstructed xyz')


# Data augmentation
data_aug_arg = add_argument_group('DataAugmentation')
data_aug_arg.add_argument(
    '--use_feat_aug', type=str2bool, default=True, help='Simple feat augmentation')
data_aug_arg.add_argument(
    '--data_aug_color_trans_ratio', type=float, default=0.10, help='Color translation range')
data_aug_arg.add_argument(
    '--data_aug_color_jitter_std', type=float, default=0.05, help='STD of color jitter')
data_aug_arg.add_argument('--normalize_color', type=str2bool, default=False)
data_aug_arg.add_argument('--shift', type=str2bool, default=False)
data_aug_arg.add_argument('--jitter', type=str2bool, default=False)
data_aug_arg.add_argument('--scale', type=str2bool, default=False)
data_aug_arg.add_argument('--rot_aug', type=str2bool, default=False)
data_aug_arg.add_argument('--random_rotation', type=str2bool, default=False)
data_aug_arg.add_argument('--color_offset', type=float, default=0.5)
data_aug_arg.add_argument('--data_aug_scale_min', type=float, default=0.8)
data_aug_arg.add_argument('--data_aug_scale_max', type=float, default=1.2)
data_aug_arg.add_argument(
    '--data_aug_hue_max', type=float, default=0.5, help='Hue translation range. [0, 1]')
data_aug_arg.add_argument(
    '--data_aug_saturation_max',
    type=float,
    default=0.20,
    help='Saturation translation range, [0, 1]')
data_aug_arg.add_argument('--distort_partnet', type=str2bool, default=False)

# Test
test_arg = add_argument_group('Test')
test_arg.add_argument('--visualize', type=str2bool, default=False)
test_arg.add_argument('--test_temporal_average', type=str2bool, default=False)
test_arg.add_argument('--visualize_path', type=str, default='outputs/visualize')
test_arg.add_argument('--test_config', type=str, default='')
test_arg.add_argument('--save_prediction', type=str2bool, default=False)
test_arg.add_argument('--save_pred_dir', type=str, default='outputs/pred')
test_arg.add_argument('--test_phase', type=str, default='test', help='Dataset for test')
test_arg.add_argument('--idw', type=str2bool, default=False, help='Use IDW for tranferring voxel predictions to points')
test_arg.add_argument(
    '--evaluate_original_pointcloud',
    type=str2bool,
    default=False,
    help='Test on the original pointcloud space during network evaluation using voxel projection.')
test_arg.add_argument(
    '--test_original_pointcloud',
    type=str2bool,
    default=False,
    help='Test on the original pointcloud space as given by the dataset using kd-tree.')
test_arg.add_argument(
    '--test_buildnet',
    type=str2bool,
    default=False,
    help='Test on BuildNet dataset')
test_arg.add_argument(
    '--test_stylenet',
    type=str2bool,
    default=False,
    help='Test on Stylenet dataset')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_cuda', type=str2bool, default=True)
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--multi_gpu', type=str2bool, default=False)
misc_arg.add_argument('--max_ngpu', type=int, default=8)
misc_arg.add_argument('--seed', type=int, default=123)
misc_arg.add_argument('--export_feat', type=str2bool, default=False, help='Export features')
misc_arg.add_argument('--export_feat_mode', type=str2export_feat_mode, default='sum', help='Export features mode')


def get_config():
  config = parser.parse_args()
  if 'AE' in config.model and not "VAE" in config.model:
    assert config.input_feat == 'coords' and config.chamfer_loss
  return config  # Training settings
