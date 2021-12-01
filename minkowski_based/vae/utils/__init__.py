import logging
import os
import re
from os import listdir, makedirs
from os.path import join, exists, dirname, realpath
from shutil import rmtree
from time import sleep

import torch

this_dir = dirname(realpath(__file__))
SCREEN_CAMERA_POSITION_FILE = join(this_dir, "ScreenCameraPosition.json")
SCREEN_CAMERA_UNIT_CUBE_FILE = join(this_dir, "ScreenCameraUnitCube.json")
SCREEN_CAMERA_4UNIT_CUBES_FILE = join(this_dir, "ScreenCamera4UnitCubes.json")
SCREEN_CAMERA_4CUBES_FILE = join(this_dir, "ScreenCamera4cubes.json")
SCREEN_CAMERA_4MODELNET1_FILE = join(this_dir, "ScreenCamera4modelnet1.json")
SCREEN_CAMERA_4MODELNET2_FILE = join(this_dir, "ScreenCamera4modelnet2.json")


def setup_logging(log_dir):
    makedirs(log_dir, exist_ok=True)

    logpath = join(log_dir, 'log.txt')
    filemode = 'a' if exists(logpath) else 'w'

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=logpath,
                        filemode=filemode)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def set_logger_file(log_file, logger):
    makedirs(dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, 'a')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    for hdlr in logger.handlers[:]:  # remove the existing file handlers
        if isinstance(hdlr, logging.FileHandler):
            logger.removeHandler(hdlr)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger


def prepare_results_dir(config):
    output_dir = join(config['results_root'],
                      config['data']['dataset'],
                      config['arch'],
                      config['experiment_name'])
    if config['clean_results_dir']:
        if exists(output_dir):
            print('Attention! Cleaning results directory in 10 seconds!')
            sleep(10)
        rmtree(output_dir, ignore_errors=True)
    makedirs(output_dir, exist_ok=True)
    makedirs(join(output_dir, 'weights'), exist_ok=True)
    makedirs(join(output_dir, 'samples'), exist_ok=True)
    makedirs(join(output_dir, 'encodings'), exist_ok=True)
    makedirs(join(output_dir, 'results'), exist_ok=True)
    return output_dir


def find_latest_epoch(dirpath):
    # Files with weights are in format ddddd_{D,E,G}.pth
    epoch_regex = re.compile(r'^(?P<n_epoch>\d+)_[DEG]\.pth$')
    epochs_completed = []
    if exists(join(dirpath, 'weights')):
        dirpath = join(dirpath, 'weights')
    for f in listdir(dirpath):
        m = epoch_regex.match(f)
        if m:
            epochs_completed.append(int(m.group('n_epoch')))
    return max(epochs_completed) if epochs_completed else 0


# def cuda_setup(cuda=False, gpu_idx=0):
def cuda_setup(cuda=False):
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        # torch.cuda.set_device(gpu_idx)
    else:
        device = torch.device('cpu')
    return device


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
