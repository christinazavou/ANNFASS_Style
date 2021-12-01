import logging
import os

import pandas as pd
import open3d as o3d
import torch

from datasets.dataset_utils import load_ply_mesh, load_off, get_voxelized_data, load_obj
from datasets.dataset_utils import PointCloud
import numpy as np


class ComponentMeshDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, phase, transform=None, logger=None, config=None):
        self.phase = phase
        self.files = []
        self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.resolution = config['resolution']
        self.last_cache_percent = 0

        if logger is None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger

        self.f_path = os.path.join(root_dir, f"{phase}.txt")
        df = pd.read_csv(self.f_path, sep=';', header=None, names=['file', 'building', 'component'])
        self.files = list(df['file'].values)
        assert len(self.files) > 0, "No file loaded"
        self.logger.info(
            f"Loading the subset {phase} from {self.f_path} with {len(self.files)} files"
        )
        self.density = config['density']
        self.resample = True
        self.normalize = True

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mesh_file = self.files[idx]
        if idx in self.cache:
            xyz = self.cache[idx]
        else:
            if mesh_file.endswith(".ply"):
                xyz = load_ply_mesh(mesh_file, self.density, resample=self.resample, normalize=self.normalize)
            elif mesh_file.endswith(".off"):
                xyz = load_off(mesh_file, self.density, normalize=self.normalize)
            else:
                raise Exception(f"how to read {mesh_file}")
            self.cache[idx] = xyz
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                self.logger.info(
                    f"Cached {self.phase}: {len(self.cache)} / {len(self)}: {cache_percent}%"
                )
                self.last_cache_percent = cache_percent

        if len(xyz) < 300:
            self.logger.info(
                f"Skipping {mesh_file}: does not have sufficient CAD sampling density after resampling: {len(xyz)}."
            )
            return None

        res = get_voxelized_data(xyz, self.resolution)
        return res[0], res[1], idx


class ComponentSamplesDataset(ComponentMeshDataset):

    def __init__(self, root_dir, phase, transform=None, logger=None, config=None):
        super().__init__(root_dir, phase, transform=transform, logger=logger, config=config)
        self.resample = False
        self.normalize = False


class ComponentObjDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, phase, transform=None, logger=None, config=None, ):
        self.phase = phase
        self.files = []
        self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.resolution = config['resolution']
        self.last_cache_percent = 0

        if logger is None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger

        if phase == 'train':
            assert 'train_split_file' in config, "Missing train_split_file"
            split_file = config['train_split_file']
        elif phase == 'val':
            assert 'val_split_file' in config, "Missing val_split_file"
            split_file = config['val_split_file']
        else:
            assert phase == 'test'
            assert 'test_split_file' in config, "Missing test_split_file"
            split_file = config['test_split_file']

        self.f_path = split_file
        with open(self.f_path, "r") as fin:
            self.files = fin.readlines()
            self.files = [os.path.join(root_dir, f.strip()) for f in self.files]
            self.files = [f for f in self.files if os.path.exists(f)]
        assert len(self.files) > 0, "No file loaded"
        self.logger.info(
            f"Loading the subset {self.phase} from {self.f_path} with {len(self.files)} files"
        )
        self.density = config['density']
        self.resample = True
        self.normalize = True

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.files)

    def _get_point_cloud_from_file(self, mesh_file):
        xyz = load_obj(mesh_file, self.density, normalize=self.normalize)
        return xyz

    def __getitem__(self, idx):
        mesh_file = self.files[idx]
        if idx in self.cache:
            xyz = self.cache[idx]
        else:
            if mesh_file.endswith(".obj"):
                xyz = self._get_point_cloud_from_file(mesh_file)
            else:
                raise Exception(f"how to read {mesh_file}")
            self.cache[idx] = xyz
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                self.logger.info(
                    f"Cached {self.phase}: {len(self.cache)} / {len(self)}: {cache_percent}%"
                )
                self.last_cache_percent = cache_percent

        if len(xyz) < 300:
            self.logger.info(
                f"Skipping {mesh_file}: does not have sufficient CAD sampling density after resampling: {len(xyz)}."
            )
            return None

        res = get_voxelized_data(xyz, self.resolution)
        return res[0], res[1], idx
