import os
import logging
import pickle
import random
import sys

import numpy as np
import yaml

from dataset2 import FlexDataset
from utils import get_vox_from_binvox_1over2,get_vox_from_binvox
from scipy.ndimage.filters import gaussian_filter


class RotFilesDataset(FlexDataset):

    def __init__(self, data_dir, filepath, config, log, **kwargs):
        self.current_file_idx = 0
        super(RotFilesDataset, self).__init__(data_dir, filepath, config, log, **kwargs)

    def init(self):
        files_to_keys = {}
        self.keys_to_files = {}
        self.files = {}
        with open(self.filepath, "r") as fin:
            files = fin.readlines()
        for file in files:
            try:
                model_name, rotation = file.rstrip().split("_rot")
            except:
                model_name, rotation = file.rstrip(), "0"
            file = os.path.join(self.data_dir, file.strip(), self.filename)
            if os.path.exists(file):
                if not model_name in files_to_keys:
                    files_to_keys[model_name] = len(files_to_keys)
                    self.keys_to_files[len(self.keys_to_files)] = model_name
                self.files.setdefault(model_name, [])
                self.files[model_name].append(file)
        assert len(self.files) > 0, "No file loaded"
        print(f"Loading {self.filepath} with {len(self.files)} models")
        self.log.debug(f"Loading {self.filepath} with {len(self.files)} models")

        # shuffle them in each split. files are sorted in all splits.
        self.files_per_idx = {}
        for model_name in self.files:
            binvox_files = self.files[model_name]
            random.shuffle(binvox_files)
            for idx in range(len(binvox_files)):
                file = binvox_files[idx]
                self.files_per_idx.setdefault(idx, [])
                self.files_per_idx[idx] += [file]

    def __len__(self):
        lengths = set()
        for idx in self.files_per_idx:
            length = len(self.files_per_idx[idx])
            lengths.add(length)
        assert len(lengths) == 1, "Invalid Data: Not all files contain same amount of rotations :("
        return lengths.pop()

    def change_file_idx(self, file_idx=None):
        indices = [i for i in self.files_per_idx]
        if file_idx is None:
            random.shuffle(indices)
            file_idx = indices[0]
        else:
            total = len(indices)
            file_idx = file_idx % total
        if file_idx != self.current_file_idx:
            self.current_file_idx = file_idx
            self.cache = {}
            self.last_cache_percent = 0
            self.load()

    def __getitem__(self, idx):

        len_self = len(self)

        binvox_file = self.files_per_idx[self.current_file_idx][idx]
        if idx in self.cache:
            data_dict = self.cache[idx]
        else:
            print(f"reading {binvox_file}")
            self.log.debug(f"reading {binvox_file}")
            if self.output_size == 128:
                tmp_raw = get_vox_from_binvox_1over2(binvox_file).astype(np.uint8)
            elif self.output_size == 256:
                tmp_raw = get_vox_from_binvox(binvox_file).astype(np.uint8)
            xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)

            voxel_style = gaussian_filter(tmp.astype(np.float32), sigma=1)  # TODO: shouldn't be saved
            tmp_Dmask_style = self.get_style_voxel_Dmask(tmp)
            tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)

            data_dict = {
                # 'raw': tmp_raw,
                # 'cropped': tmp,
                'pos': (xmin, xmax, ymin, ymax, zmin, zmax),
                'voxel_style': voxel_style,
                'Dmask_style': tmp_Dmask_style,
                'input': tmp_input,
                'Dmask': tmp_Dmask,
                'mask': tmp_mask,
                # 'fname': binvox_file
            }
            self.cache[idx] = data_dict

            cache_percent = int((len(self.cache) / len_self) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                print(f"Cached : {len(self.cache)} / {len_self}: {cache_percent}%")
                self.log.debug(f"Cached : {len(self.cache)} / {len_self}: {cache_percent}%")
                self.last_cache_percent = cache_percent

            if cache_percent == 100 and self.cache_dir:
                self.save()

        return data_dict

    def get_without_cache(self, idx, rot_idx=None):

        binvox_file = self.files_per_idx[self.current_file_idx][idx]

        print(f"reading {binvox_file}")
        self.log.debug(f"reading {binvox_file}")
        if self.output_size == 128:
            tmp_raw = get_vox_from_binvox_1over2(binvox_file).astype(np.uint8)
        elif self.output_size == 256:
            tmp_raw = get_vox_from_binvox(binvox_file).astype(np.uint8)
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
        tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)

        voxel_style = gaussian_filter(tmp.astype(np.float32), sigma=1)
        tmp_Dmask_style = self.get_style_voxel_Dmask(tmp)
        tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)

        data_dict = {
            # 'raw': tmp_raw,
            # 'cropped': tmp,
            'pos': (xmin, xmax, ymin, ymax, zmin, zmax),
            'voxel_style': voxel_style,
            'Dmask_style': tmp_Dmask_style,
            'input': tmp_input,
            'Dmask': tmp_Dmask,
            'mask': tmp_mask,
            # 'fname': binvox_file
        }

        return data_dict

    def get_more(self, idx, rot_idx=None):
        binvox_file = self.files_per_idx[self.current_file_idx][idx]
        if self.output_size == 128:
            tmp_raw = get_vox_from_binvox_1over2(binvox_file).astype(np.uint8)
        elif self.output_size == 256:
            tmp_raw = get_vox_from_binvox(binvox_file).astype(np.uint8)
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
        tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)
        return tmp, None

    def get_model_name(self, i):
        mesh_file = self.files_per_idx[self.current_file_idx][i]
        model_name = mesh_file.replace(self.data_dir+"/", "")
        model_name = model_name.replace("/"+self.filename, "")
        return model_name

    def load(self):
        if not self.cache_dir:
            return False
        files_file = os.path.join(self.cache_dir, "data_files.pkl")
        cache_file = os.path.join(self.cache_dir, f"data_cache_{self.current_file_idx}.pkl")
        last_cache_percent_file = os.path.join(self.cache_dir, f"data_last_cache_percent_{self.current_file_idx}.pkl")
        if os.path.exists(cache_file) and os.path.exists(files_file) and os.path.exists(last_cache_percent_file):
            with open(files_file, "rb") as fin:
                self.files_per_idx = pickle.load(fin)
            with open(cache_file, "rb") as fin:
                self.cache = pickle.load(fin)
            with open(last_cache_percent_file, "rb") as fin:
                self.last_cache_percent = pickle.load(fin)
            print("Loaded pickled data!")
            self.log.debug("Loaded pickled data!")
            return True
        return False

    def save(self):
        if not self.cache_dir:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        files_file = os.path.join(self.cache_dir, "data_files.pkl")
        cache_file = os.path.join(self.cache_dir, f"data_cache_{self.current_file_idx}.pkl")
        last_cache_percent_file = os.path.join(self.cache_dir, f"data_last_cache_percent_{self.current_file_idx}.pkl")
        with open(files_file, "wb") as fout:
            pickle.dump(self.files_per_idx, fout)
        with open(cache_file, "wb") as fout:
            pickle.dump(self.cache, fout)
        with open(last_cache_percent_file, "wb") as fout:
            pickle.dump(self.last_cache_percent, fout)
        print("Saved pickled data!")
        self.log.debug("Saved pickled data!")


class RotRndNoiseFilesDataset(RotFilesDataset):

    def __init__(self, data_dir, filepath, config, log, **kwargs):
        self.current_file_idx = 0
        super(RotRndNoiseFilesDataset, self).__init__(data_dir, filepath, config, log, **kwargs)

    def __getitem__(self, idx):

        len_self = len(self)

        binvox_file = self.files_per_idx[self.current_file_idx][idx]
        if idx in self.cache:
            data_dict = self.cache[idx]
        else:
            print(f"reading {binvox_file}")
            self.log.debug(f"reading {binvox_file}")
            if self.output_size == 128:
                tmp_raw = get_vox_from_binvox_1over2(binvox_file).astype(np.uint8)
            elif self.output_size == 256:
                tmp_raw = get_vox_from_binvox(binvox_file).astype(np.uint8)
            xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)

            voxel_style = tmp.astype(np.float32)
            tmp_Dmask_style = self.get_style_voxel_Dmask(tmp)
            tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)

            data_dict = {
                'pos': (xmin, xmax, ymin, ymax, zmin, zmax),
                'voxel_style': voxel_style,
                'Dmask_style': tmp_Dmask_style,
                'input': tmp_input,
                'Dmask': tmp_Dmask,
                'mask': tmp_mask,
            }
            self.cache[idx] = data_dict

            cache_percent = int((len(self.cache) / len_self) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                print(f"Cached : {len(self.cache)} / {len_self}: {cache_percent}%")
                self.log.debug(f"Cached : {len(self.cache)} / {len_self}: {cache_percent}%")
                self.last_cache_percent = cache_percent

            if cache_percent == 100 and self.cache_dir:
                self.save()

        data_dict['voxel_style'] = gaussian_filter(data_dict['voxel_style'], sigma=1)

        return data_dict


class RotSameFilesDataset(RotFilesDataset):

    def __init__(self, data_dir, filepath, config, log, **kwargs):
        self.current_file_idx = 0
        super(RotSameFilesDataset, self).__init__(data_dir, filepath, config, log, **kwargs)

    def init(self):
        self.modelnames = []  # without rotation
        with open(self.filepath, "r") as fin:
            files = fin.readlines()
        for file in files:
            model_name = file.rstrip()
            file = os.path.join(self.data_dir, model_name, self.filename)
            if os.path.exists(file):
                self.modelnames.append(model_name)
                for rot in range(36, 360, 36):
                    file = os.path.join(self.data_dir, model_name + f"_rot{rot}", self.filename)
                    assert os.path.exists(file)
        assert len(self.modelnames) > 0, "No file loaded"
        print(f"Loading {self.filepath} with {len(self.modelnames)} models")
        self.log.debug(f"Loading {self.filepath} with {len(self.modelnames)} models")

        # idx is about one split ... and contains various rotations ...
        # shuffle them in each split.
        rotations = list(range(0, 360, 36))
        indices = list(range(len(rotations)))

        files_per_idx = {idx: [] for idx in range(len(rotations))}  # stores filepath, rot
        for model_name in self.modelnames:
            random.shuffle(indices)
            for idx, rot in zip(indices, rotations):
                if rot == 0:
                    binvox_file = os.path.join(self.data_dir, model_name, self.filename)
                else:
                    binvox_file = os.path.join(self.data_dir, model_name + f"_rot{rot}", self.filename)
                files_per_idx[idx].append((binvox_file, rot))

        self.file_indices_per_rot_per_idx = {idx: {rot: [] for rot in rotations} for idx in indices}

        self.files_per_idx = {}
        for idx, files in files_per_idx.items():
            random.shuffle(files)
            self.files_per_idx[idx] = files
            for rot in rotations:
                rot_files_indices = [i for i, (f, r) in enumerate(files) if r == rot]
                self.file_indices_per_rot_per_idx[idx][rot] = rot_files_indices

    def get_file_indices_with_same_rot(self, idx):
        binvox_file, rot = self.files_per_idx[self.current_file_idx][idx]
        return self.get_file_indices_with_rot(rot)

    def get_file_indices_with_rot(self, rot):
        return self.file_indices_per_rot_per_idx[self.current_file_idx][rot]

    def __getitem__(self, idx):

        len_self = len(self)

        binvox_file, rot = self.files_per_idx[self.current_file_idx][idx]
        if idx in self.cache:
            data_dict = self.cache[idx]
        else:
            print(f"reading {binvox_file}")
            self.log.debug(f"reading {binvox_file}")
            if self.output_size == 128:
                tmp_raw = get_vox_from_binvox_1over2(binvox_file).astype(np.uint8)
            elif self.output_size == 256:
                tmp_raw = get_vox_from_binvox(binvox_file).astype(np.uint8)
            xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)

            voxel_style = gaussian_filter(tmp.astype(np.float32), sigma=1)  # TODO: shouldn't be saved
            tmp_Dmask_style = self.get_style_voxel_Dmask(tmp)
            tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)

            data_dict = {
                # 'raw': tmp_raw,
                # 'cropped': tmp,
                'pos': (xmin, xmax, ymin, ymax, zmin, zmax),
                'voxel_style': voxel_style,
                'Dmask_style': tmp_Dmask_style,
                'input': tmp_input,
                'Dmask': tmp_Dmask,
                'mask': tmp_mask,
                # 'fname': binvox_file
            }
            self.cache[idx] = data_dict

            cache_percent = int((len(self.cache) / len_self) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                print(f"Cached : {len(self.cache)} / {len_self}: {cache_percent}%")
                self.log.debug(f"Cached : {len(self.cache)} / {len_self}: {cache_percent}%")
                self.last_cache_percent = cache_percent

            if cache_percent == 100 and self.cache_dir:
                self.save()

        return data_dict

    def get_without_cache(self, idx, rot_idx=None):
        binvox_file, rot = self.files_per_idx[self.current_file_idx][idx]
        print(f"reading {binvox_file}")
        self.log.debug(f"reading {binvox_file}")
        if self.output_size == 128:
            tmp_raw = get_vox_from_binvox_1over2(binvox_file).astype(np.uint8)
        elif self.output_size == 256:
            tmp_raw = get_vox_from_binvox(binvox_file).astype(np.uint8)
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
        tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)

        voxel_style = gaussian_filter(tmp.astype(np.float32), sigma=1)  # TODO: shouldn't be saved
        tmp_Dmask_style = self.get_style_voxel_Dmask(tmp)
        tmp_input, tmp_Dmask, tmp_mask = self.get_voxel_input_Dmask_mask(tmp)

        data_dict = {
            # 'raw': tmp_raw,
            # 'cropped': tmp,
            'pos': (xmin, xmax, ymin, ymax, zmin, zmax),
            'voxel_style': voxel_style,
            'Dmask_style': tmp_Dmask_style,
            'input': tmp_input,
            'Dmask': tmp_Dmask,
            'mask': tmp_mask,
            # 'fname': binvox_file
        }
        return data_dict

    def get_more(self, idx, rot_idx=None):
        binvox_file, rot = self.files_per_idx[self.current_file_idx][idx]
        if self.output_size == 128:
            tmp_raw = get_vox_from_binvox_1over2(binvox_file).astype(np.uint8)
        elif self.output_size == 256:
            tmp_raw = get_vox_from_binvox(binvox_file).astype(np.uint8)
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
        tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)
        return tmp, None

    def get_model_name(self, i):
        mesh_file, rot = self.files_per_idx[self.current_file_idx][i]
        model_name = mesh_file.replace(self.data_dir+"/", "")
        model_name = model_name.replace("/"+self.filename, "")
        return model_name

    def load(self):
        if not self.cache_dir:
            return False
        files_file = os.path.join(self.cache_dir, "data_files.pkl")
        file_indices_per_rot_per_idx_file = os.path.join(self.cache_dir, "data_file_indices_per_rot_per_idx.pkl")
        cache_file = os.path.join(self.cache_dir, f"data_cache_{self.current_file_idx}.pkl")
        last_cache_percent_file = os.path.join(self.cache_dir, f"data_last_cache_percent_{self.current_file_idx}.pkl")
        if os.path.exists(cache_file) and os.path.exists(files_file) and os.path.exists(last_cache_percent_file):
            with open(files_file, "rb") as fin:
                self.files_per_idx = pickle.load(fin)
            with open(file_indices_per_rot_per_idx_file, "rb") as fin:
                self.file_indices_per_rot_per_idx = pickle.load(fin)
            with open(cache_file, "rb") as fin:
                self.cache = pickle.load(fin)
            with open(last_cache_percent_file, "rb") as fin:
                self.last_cache_percent = pickle.load(fin)
            print("Loaded pickled data!")
            self.log.debug("Loaded pickled data!")
            return True
        return False

    def save(self):
        if not self.cache_dir:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        files_file = os.path.join(self.cache_dir, "data_files.pkl")
        file_indices_per_rot_per_idx_file = os.path.join(self.cache_dir, "data_file_indices_per_rot_per_idx.pkl")
        cache_file = os.path.join(self.cache_dir, f"data_cache_{self.current_file_idx}.pkl")
        last_cache_percent_file = os.path.join(self.cache_dir, f"data_last_cache_percent_{self.current_file_idx}.pkl")
        with open(files_file, "wb") as fout:
            pickle.dump(self.files_per_idx, fout)
        with open(file_indices_per_rot_per_idx_file, "wb") as fout:
            pickle.dump(self.file_indices_per_rot_per_idx, fout)
        with open(cache_file, "wb") as fout:
            pickle.dump(self.cache, fout)
        with open(last_cache_percent_file, "wb") as fout:
            pickle.dump(self.last_cache_percent, fout)
        print("Saved pickled data!")
        self.log.debug("Saved pickled data!")


if __name__ == '__main__':

    import argparse
    from utils.io_helper import setup_logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", type=str, help="Dir containing binvox data")
    parser.add_argument("--file_path", dest="file_path", type=str, help="Split file")
    parser.add_argument("--asymmetry", action="store_true", dest="asymmetry", default=False, help="True for asymmetric shapes [False]")
    parser.add_argument("--cache_dir", dest="cache_dir", default=None, type=str, help="Dir for cached dataset")
    parser.add_argument("--log_dir", dest="log_dir", type=str, help="Dir for log file")
    parser.add_argument("--filename", type=str, dest="filename", default='model_depth_fusion.binvox', help="The name of file")
    parser.add_argument("--gpu", type=int, dest="gpu", default=0, help="The gpu")
    parser.add_argument("--input_size", type=int, dest="input_size", default=16)
    parser.add_argument("--output_size", type=int, dest="output_size", default=128)
    parser.add_argument("--config_yml", dest="config_yml", default=None, help="use this yml file instead")
    FLAGS = parser.parse_args()

    if FLAGS.config_yml:
        config = yaml.load(open(FLAGS.config_yml), Loader=yaml.FullLoader)
        FLAGS.__dict__.update(config)

    print(f"FLAGS:\n\n{FLAGS}\n\n")

    assert os.path.exists(FLAGS.data_dir) and os.path.exists(FLAGS.file_path)

    setup_logging(FLAGS.log_dir)
    LOGGER = logging.getLogger(__name__)

    # dset = RotFilesDataset(FLAGS.data_dir, FLAGS.file_path, FLAGS, LOGGER, filename=FLAGS.filename)
    dset = RotSameFilesDataset(FLAGS.data_dir, FLAGS.file_path, FLAGS, LOGGER, filename=FLAGS.filename)
    dset_len = len(dset)
    for repeat_idx in range(20):
        dset.change_file_idx(repeat_idx)
        for iter_idx in range(dset_len):
            dd = dset.__getitem__(iter_idx)
        print(f"repeat_idx is {repeat_idx}, size of cache is {sys.getsizeof(dset.cache)}")
