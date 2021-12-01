import os
import logging
import random

import pandas as pd
import open3d as o3d
from torch.utils.data import Dataset

from utils.plyfile import load_ply


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


class ContentStyleComponentDataset(Dataset):

    def __init__(self, root_dir, transform=None, split='train', content_pts=1024, style_pts=8192, **kwargs):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.content_files = []
        self.cache = {}
        self.data_objects = []
        self.last_cache_percent = 0
        self.content_pts = content_pts
        self.style_pts = style_pts

        f_content_path = os.path.join(root_dir, "content", f"{split}.txt")
        f_style_path = os.path.join(root_dir, "style", f"{split}.txt")
        content_df = pd.read_csv(f_content_path, sep=";", header=None, names=['file', 'building', 'component'])
        style_df = pd.read_csv(f_style_path, sep=";", header=None, names=['file', 'building', 'component'])

        assert len(content_df) > 0, "No content file loaded"
        assert len(style_df) > 0, "No style file loaded"
        logging.info(
            f"Loading the subset {split} from {f_content_path} with {len(content_df)} files "
            f"and {f_style_path} with {len(style_df)} files."
        )

        self.file_pairs = []
        for building in content_df.building.unique():
            content_building_df = content_df[content_df['building'] == building]
            style_building_df = style_df[style_df['building'] == building]

            for component in content_building_df.component.unique():
                content_file = content_building_df[content_building_df['component'] == component]
                assert len(content_file) == 1
                content_file = content_file.file.values[0]

                content_detailed_file = style_building_df[style_building_df['component'] == component]
                assert len(content_detailed_file) == 1
                content_detailed_file = content_detailed_file.file.values[0]

                other_style_building_df = style_building_df[style_building_df['component'] != component]
                if len(other_style_building_df) == 0:
                    continue
                for random_pair in range(min(len(other_style_building_df), 5)):
                    rnd_idx = random.sample(list(other_style_building_df.index), 1)[0]
                    style_file = other_style_building_df.loc[rnd_idx].file

                    self.file_pairs.append((content_file, content_detailed_file, style_file))

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        content_file, content_detailed_file, style_file = self.file_pairs[idx]
        if idx in self.cache:
            content_xyz, content_detailed_xyz, style_xyz = self.cache[idx]
        else:
            if content_file.endswith(".ply"):
                content_xyz = load_ply(content_file)
                content_detailed_xyz = load_ply(content_detailed_file)
                style_xyz = load_ply(style_file)
            else:
                raise Exception(f"how to read {content_file}")
            self.cache[idx] = (content_xyz, content_detailed_xyz, style_xyz)
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                logging.info(
                    f"Cached {self.split}: {len(self.cache)} / {len(self)}: {cache_percent}%"
                )
                self.last_cache_percent = cache_percent

        if len(content_xyz) < self.content_pts or len(style_xyz) < self.style_pts:
            logging.info(
                f"Skipping {content_file}, {style_xyz}: do not have sufficient sampling density: {len(content_xyz)}, {len(style_xyz)}."
            )
            return None, None, None, (content_file, content_detailed_file, style_file)

        if self.transform:
            content_xyz = self.transform(content_xyz)
            content_detailed_xyz = self.transform(content_detailed_xyz)
            style_xyz = self.transform(style_xyz)

        return content_xyz, content_detailed_xyz, style_xyz, (content_file, content_detailed_file, style_file)
