import os
import logging

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.plyfile import load_ply


class BuildingComponentDataset(Dataset):
    STYLES = []
    def __init__(self, root_dir, classes=[],
                 transform=None, split='train', n_points=2048, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the point clouds.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        if not classes:
            classes = self.STYLES

        self.phase = split
        self.files = []
        self.cache = {}
        self.transform = transform
        self.last_cache_percent = 0
        self.n_points = n_points

        for key, value in kwargs.items():
            self.__dict__[key] = value

        self.init()

    def init(self):
        self.fpath = os.path.join(self.root_dir, self.split+".txt")
        df = pd.read_csv(self.fpath, sep=';', header=None, names=['file', 'building', 'component'])
        self.files = list(df['file'].values)
        assert len(self.files) > 0, "No file loaded"
        logging.info(
            f"Loading the subset {self.split} from {self.fpath} with {len(self.files)} files"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        ply_file = self.files[idx]
        if idx in self.cache:
            xyz = self.cache[idx]
        else:
            xyz = load_ply(ply_file)
            self.cache[idx] = xyz
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                logging.info(
                    f"Cached {self.phase}: {len(self.cache)} / {len(self)}: {cache_percent}%"
                )
                self.last_cache_percent = cache_percent

        if len(xyz) < self.n_points:
            logging.info(
                f"Skipping {ply_file}: does not have sufficient sampling density: {len(xyz)}."
            )
            return None, None

        if self.transform:
            xyz = self.transform(xyz)

        if len(xyz) > self.n_points:
            xyz = xyz[np.random.randint(xyz.shape[0], size=self.n_points), :]

        return xyz, ply_file


class AnnfassComponentDataset(BuildingComponentDataset):
    STYLES = [
        "Unknown",
        "Colonial",
        "Neo_classicism",
        "Modernist",
        "Ottoman",
        "Gothic",
        "Byzantine",
        "Venetian"
    ]


class BuildnetComponentDataset(BuildingComponentDataset):
    STYLES = [
    ]


class BuildingComponentDataset2(BuildingComponentDataset):

    def __init__(self, transform=None, n_points=2048, **kwargs):
        super(BuildingComponentDataset2, self).__init__("", [], transform, "", n_points, **kwargs)
        assert 'data_root' in kwargs
        assert 'txt_file' in kwargs

    def init(self):
        with open(self.txt_file, "r") as fin:
            files = fin.readlines()
            files = [f.rstrip() for f in files]
            files = [os.path.join(self.data_root, f) for f in files if os.path.exists(os.path.join(self.data_root, f))]

        self.files = files
        assert len(self.files) > 0, "No file loaded"
        logging.info(
            f"Loading the subset {self.split} from {self.txt_file} with {len(self.files)} files"
        )


class BuildingComponentDataset2WithColor(BuildingComponentDataset2):

    def __init__(self, transform=None, n_points=2048, **kwargs):
        super(BuildingComponentDataset2WithColor, self).__init__(transform, n_points, **kwargs)

    def __getitem__(self, idx):

        ply_file = self.files[idx]
        if idx in self.cache:
            xyz, rgb = self.cache[idx]
        else:
            xyz, rgb = load_ply(ply_file, with_color=True)
            # normalize color. The ply i know it's normalized.
            if 1 < rgb.max() < 256 and 0 <= rgb.min():
                rgb = rgb / 255.
            self.cache[idx] = xyz, rgb
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                logging.info(
                    f"Cached {self.phase}: {len(self.cache)} / {len(self)}: {cache_percent}%"
                )
                self.last_cache_percent = cache_percent

        if len(xyz) < self.n_points:
            logging.info(
                f"Skipping {ply_file}: does not have sufficient sampling density: {len(xyz)}."
            )
            return None, None

        if self.transform:
            xyz = self.transform(xyz)

        if len(xyz) > self.n_points:
            xyz = xyz[np.random.randint(xyz.shape[0], size=self.n_points), :]
            rgb = rgb[np.random.randint(rgb.shape[0], size=self.n_points), :]

        features = np.hstack([xyz, rgb])
        return features, ply_file


class BuildingComponentRawDataset(BuildingComponentDataset):
    STYLES = []

    def init(self):
        self.fpath = self.root_dir
        self.files = []
        for root, dirs, files in os.walk(self.fpath):
            for file in files:
                if file.endswith(".ply"):
                    self.files.append(os.path.join(root, file))
        assert len(self.files) > 0, "No file loaded"
        logging.info(
            f"Loading the subset {self.split} from {self.fpath} with {len(self.files)} files"
        )
