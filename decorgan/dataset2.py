import os
import logging
import torch
import random
import numpy as np
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from utils import get_vox_from_binvox_1over2, get_voxel_bbox, get_torch_device, get_vox_from_binvox, crop_voxel
from scipy.ndimage.filters import gaussian_filter
from utils.io_helper import ObjMeshBasic, parse_simple_obj_file
import pickle


class BuildingDataset(object):

    def __init__(self, data_dir):
        assert os.path.exists(data_dir), "Can't generate BuildingDataset..."
        self.data_dir = data_dir

    def load_mesh(self, building):
        mesh_file = os.path.join(self.data_dir, "normalizedObj", building, f"{building}.obj")
        obj = ObjMeshBasic(mesh_file)
        return obj


class BasicDataset(Dataset):

    def _set_basic_config(self, data_dir, filepath, config, log, **kwargs):
        self.data_dir = data_dir

        self.filepath = filepath
        self.log = log or logging.getLogger(self.__class__.__name__)

        self.files = []
        self.cache = {}
        self.last_cache_percent = 0

        self.real_size = 256
        self.mask_margin = 8

        self.asymmetry = config.asymmetry
        self.cache_dir = config.cache_dir or None
        if self.cache_dir:
            self.cache_dir = os.path.join(self.cache_dir, os.path.basename(self.filepath).split(".")[0])

        self.filename = "model_depth_fusion.binvox" if 'filename' not in kwargs else kwargs['filename']

        self.device = get_torch_device(config, self.log)

    def _set_specific_config(self, config):
        self.input_size = 16
        self.output_size = 128
        self.upsample_rate = 8

    def __init__(self, data_dir, filepath, config, log, **kwargs):
        self._set_basic_config(data_dir, filepath, config, log, **kwargs)
        self._set_specific_config(config)
        if not self.load():
            self.init()

    def load(self):
        if not self.cache_dir:
            return False
        cache_file = os.path.join(self.cache_dir, "data_cache.pkl")
        files_file = os.path.join(self.cache_dir, "data_files.pkl")
        last_cache_percent_file = os.path.join(self.cache_dir, "data_last_cache_percent.pkl")
        if os.path.exists(cache_file) and os.path.exists(files_file) and os.path.exists(last_cache_percent_file):
            with open(cache_file, "rb") as fin:
                self.cache = pickle.load(fin)
            with open(files_file, "rb") as fin:
                self.files = pickle.load(fin)
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
        cache_file = os.path.join(self.cache_dir, "data_cache.pkl")
        files_file = os.path.join(self.cache_dir, "data_files.pkl")
        last_cache_percent_file = os.path.join(self.cache_dir, "data_last_cache_percent.pkl")
        with open(cache_file, "wb") as fout:
            pickle.dump(self.cache, fout)
        with open(files_file, "wb") as fout:
            pickle.dump(self.files, fout)
        with open(last_cache_percent_file, "wb") as fout:
            pickle.dump(self.last_cache_percent, fout)
        print("Saved pickled data!")
        self.log.debug("Saved pickled data!")

    def init(self):
        with open(self.filepath, "r") as fin:
            self.files = fin.readlines()
            self.files = [os.path.join(self.data_dir, f.strip(), self.filename) for f in self.files]
            self.files = [f for f in self.files if os.path.exists(f)]
        assert len(self.files) > 0, "No file loaded"
        print(f"Loading {self.filepath} with {len(self.files)} files")
        self.log.debug(f"Loading {self.filepath} with {len(self.files)} files")

    def get_reference_file(self, i):
        file_dir = os.path.dirname(self.files[i])
        mesh_file = os.path.join(file_dir, "model.obj")
        return mesh_file

    def get_input_file(self, i):
        file_dir = os.path.dirname(self.files[i])
        mesh_file = os.path.join(file_dir, self.filename)
        return mesh_file

    def get_model_name(self, i):
        mesh_file = self.files[i]
        model_name = mesh_file.replace(self.data_dir+"/", "")
        model_name = model_name.replace("/"+self.filename, "")
        return model_name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        binvox_file = self.files[idx]
        if idx in self.cache:
            data_dict = self.cache[idx]
        else:
            print(f"reading {binvox_file}")
            self.log.debug(f"reading {binvox_file}")
            tmp_raw = get_vox_from_binvox_1over2(binvox_file).astype(np.uint8)
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
            self.cache[idx] = data_dict

            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                print(f"Cached : {len(self.cache)} / {len(self)}: {cache_percent}%")
                self.log.debug(f"Cached : {len(self.cache)} / {len(self)}: {cache_percent}%")
                self.last_cache_percent = cache_percent

            if cache_percent == 100 and self.cache_dir:
                self.save()

        return data_dict

    def get_without_cache(self, idx):
        binvox_file = self.files[idx]
        print(f"reading {binvox_file}")
        self.log.debug(f"reading {binvox_file}")
        tmp_raw = get_vox_from_binvox_1over2(binvox_file).astype(np.uint8)
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

    def get_more(self, idx):
        binvox_file = self.files[idx]
        tmp_raw = get_vox_from_binvox_1over2(binvox_file).astype(np.uint8)
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
        tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)
        return tmp, None

    def crop_voxel(self, vox, xmin, xmax, ymin, ymax, zmin, zmax):
        return crop_voxel(vox, xmin, xmax, ymin, ymax, zmin, zmax,
                          self.upsample_rate, self.mask_margin, self.asymmetry)

    def get_voxel_bbox(self, vox, debug=False):
        return get_voxel_bbox(vox, self.device, self.upsample_rate, self.asymmetry, debug=debug)

    def get_style_voxel_Dmask(self, vox):
        if self.upsample_rate == 4:
            # 256 -crop- 244 -maxpoolk6s2- 120
            crop_margin = 6
            kernel_size = 6
        elif self.upsample_rate == 8:
            # 256 -crop- 252 -maxpoolk14s2- 120
            crop_margin = 2
            kernel_size = 14
        vox_tensor = torch.from_numpy(
            vox[crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]).to(
            self.device).unsqueeze(0).unsqueeze(0).float()
        smallmask_tensor = F.max_pool3d(vox_tensor, kernel_size=kernel_size, stride=2, padding=0)
        smallmask = smallmask_tensor.detach().cpu().numpy()[0, 0]
        smallmask = np.round(smallmask).astype(np.uint8)
        return smallmask

    def get_style_voxel_Dmask_tensor(self, vox):
        if self.upsample_rate == 4:
            # 256 -crop- 244 -maxpoolk6s2- 120
            crop_margin = 6
            kernel_size = 6
        elif self.upsample_rate == 8:
            # 256 -crop- 252 -maxpoolk14s2- 120
            crop_margin = 2
            kernel_size = 14
        vox_tensor = torch.from_numpy(
            vox[crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]).to(
            self.device).unsqueeze(0).unsqueeze(0).float()
        smallmask_tensor = F.max_pool3d(vox_tensor, kernel_size=kernel_size, stride=2, padding=0)
        smallmask = torch.round(smallmask_tensor).type(torch.uint8)
        return smallmask

    def get_patches_content_voxel_Dmask_tensor(self, vox):
        if self.upsample_rate == 4:
            crop_margin = 2
        elif self.upsample_rate == 8:
            crop_margin = 1
        vox_tensor = vox.unsqueeze(1).float()
        # input
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate, padding=0)
        # Dmask
        smallmask_tensor = smallmaskx_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin,
                                             crop_margin:-crop_margin]
        smallmask_tensor = F.interpolate(smallmask_tensor, scale_factor=self.upsample_rate // 2, mode='nearest')
        smallmask_tensor = torch.round(smallmask_tensor).type(torch.uint8)
        return smallmask_tensor

    def get_patches_style_voxel_Dmask_tensor(self, vox):
        # 256 -crop- 252 -maxpoolk14s2- 120
        crop_margin = 2
        kernel_size = 14
        vox_tensor = vox[:, crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]
        vox_tensor = vox_tensor.unsqueeze(1).float()
        smallmask_tensor = F.max_pool3d(vox_tensor, kernel_size=kernel_size, stride=2, padding=0)
        smallmask = torch.round(smallmask_tensor).type(torch.uint8)
        return smallmask

    def get_voxel_input_Dmask_mask(self, vox):
        if self.upsample_rate == 4:
            # 256 -maxpoolk4s4- 64 -crop- 60 -upsample- 120
            # output: 64, 120, 64
            crop_margin = 2
        elif self.upsample_rate == 8:
            # 256 -maxpoolk8s8- 32 -crop- 30 -upsample- 120
            # output: 32, 120, 64
            crop_margin = 1
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        # input
        smallmaskx_tensor = F.max_pool3d(vox_tensor,
                                         kernel_size=self.upsample_rate, stride=self.upsample_rate, padding=0)
        # Dmask
        smallmask_tensor = smallmaskx_tensor[:, :,
                                             crop_margin:-crop_margin,
                                             crop_margin:-crop_margin,
                                             crop_margin:-crop_margin]
        smallmask_tensor = F.interpolate(smallmask_tensor, scale_factor=self.upsample_rate // 2, mode='nearest')
        # mask
        # expand 1
        if self.upsample_rate == 4:
            mask_tensor = smallmaskx_tensor
        elif self.upsample_rate == 8:
            mask_tensor = F.interpolate(smallmaskx_tensor, scale_factor=2, mode='nearest')
        mask_tensor = F.max_pool3d(mask_tensor, kernel_size=3, stride=1, padding=1)
        # to numpy
        smallmaskx = smallmaskx_tensor.detach().cpu().numpy()[0, 0]
        smallmask = smallmask_tensor.detach().cpu().numpy()[0, 0]
        mask = mask_tensor.detach().cpu().numpy()[0, 0]
        smallmaskx = np.round(smallmaskx).astype(np.uint8)
        smallmask = np.round(smallmask).astype(np.uint8)
        mask = np.round(mask).astype(np.uint8)
        return smallmaskx, smallmask, mask

    def get_voxel_input_Dmask_mask_tensor(self, vox):
        if self.upsample_rate == 4:
            crop_margin = 2
        elif self.upsample_rate == 8:
            crop_margin = 1
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        # input
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate,
                                         padding=0)
        # Dmask
        smallmask_tensor = smallmaskx_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin,
                           crop_margin:-crop_margin]
        smallmask_tensor = F.interpolate(smallmask_tensor, scale_factor=self.upsample_rate // 2, mode='nearest')
        # mask
        # expand 1
        if self.upsample_rate == 4:
            mask_tensor = smallmaskx_tensor
        elif self.upsample_rate == 8:
            mask_tensor = F.interpolate(smallmaskx_tensor, scale_factor=2, mode='nearest')
        mask_tensor = F.max_pool3d(mask_tensor, kernel_size=3, stride=1, padding=1)

        smallmaskx_tensor = torch.round(smallmaskx_tensor).type(torch.uint8)
        smallmask_tensor = torch.round(smallmask_tensor).type(torch.uint8)
        mask_tensor = torch.round(mask_tensor).type(torch.uint8)
        return smallmaskx_tensor, smallmask_tensor, mask_tensor


class FlexDataset(BasicDataset):

    def _set_specific_config(self, config):
        self.input_size = config.input_size
        self.output_size = config.output_size

        if self.input_size == 64 and self.output_size == 256:
            self.upsample_rate = 4
        elif self.input_size == 32 and self.output_size == 128:
            self.upsample_rate = 4
        elif self.input_size == 32 and self.output_size == 256:
            self.upsample_rate = 8
        elif self.input_size == 16 and self.output_size == 128:
            self.upsample_rate = 8
        else:
            print("ERROR: invalid input/output size!")
            exit(-1)

    def __init__(self, data_dir, filepath, config, log, **kwargs):
        super(FlexDataset, self).__init__(data_dir, filepath, config, log, **kwargs)

    def __getitem__(self, idx):

        binvox_file = self.files[idx]
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
            self.cache[idx] = data_dict

            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                print(f"Cached : {len(self.cache)} / {len(self)}: {cache_percent}%")
                self.log.debug(f"Cached : {len(self.cache)} / {len(self)}: {cache_percent}%")
                self.last_cache_percent = cache_percent

            if cache_percent == 100 and self.cache_dir:
                self.save()

        return data_dict

    def get_without_cache(self, idx):
        binvox_file = self.files[idx]
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

    def get_more(self, idx):
        binvox_file = self.files[idx]
        if self.output_size == 128:
            tmp_raw = get_vox_from_binvox_1over2(binvox_file).astype(np.uint8)
        elif self.output_size == 256:
            tmp_raw = get_vox_from_binvox(binvox_file).astype(np.uint8)
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
        tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)
        return tmp, None


class FlexRndNoiseDataset(FlexDataset):

    def __init__(self, data_dir, filepath, config, log, **kwargs):
        super(FlexRndNoiseDataset, self).__init__(data_dir, filepath, config, log, **kwargs)

    def __getitem__(self, idx):

        binvox_file = self.files[idx]
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

            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                print(f"Cached : {len(self.cache)} / {len(self)}: {cache_percent}%")
                self.log.debug(f"Cached : {len(self.cache)} / {len(self)}: {cache_percent}%")
                self.last_cache_percent = cache_percent

            if cache_percent == 100 and self.cache_dir:
                self.save()

        data_dict['voxel_style'] = gaussian_filter(data_dict['voxel_style'], sigma=1)

        return data_dict


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

    # dset = BasicDataset(FLAGS.data_dir, FLAGS.file_path, FLAGS, LOGGER, filename=FLAGS.filename)
    dset = FlexDataset(FLAGS.data_dir, FLAGS.file_path, FLAGS, LOGGER, filename=FLAGS.filename)
    dset_len = len(dset)
    for shape_idx in range(dset_len):
        dd = dset.__getitem__(shape_idx)

