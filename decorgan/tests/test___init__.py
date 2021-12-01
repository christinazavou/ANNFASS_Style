from unittest import TestCase

import numpy as np
import torch

from common_utils.utils import get_points_from_voxel
from utils import get_voxel_bbox


class Test(TestCase):
    def test_get_points_from_voxel(self):
        vox = np.zeros((20, 20, 20))
        v = get_points_from_voxel(vox)
        print(v)

    def test_get_voxel_bbox(self):
        vox = np.random.randint(0, 2, (512, 512, 512))
        v = get_voxel_bbox(vox, torch.device('cuda'), 8, True)
        print(v)
        vox = np.packbits(vox, axis=None)
        vox = np.unpackbits(vox, axis=None).reshape([512, 512, 512]).astype(np.uint8)
        v = get_voxel_bbox(vox, torch.device('cuda'), 8, True)
        print(v)
