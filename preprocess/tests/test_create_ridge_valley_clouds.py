import os
from unittest import TestCase

import numpy as np

from preprocess.point_cloud_generation.create_ridge_valley_clouds import get_ridge_or_valley_vertices, \
    get_avg_min_distance_of_sampled_points


class Test(TestCase):
    def test_sample_edges(self):
        root_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_march"
        obj_dir = os.path.join(root_dir, "normalizedObj")
        for root, dirs, files in os.walk(obj_dir):
            for building in dirs:
                if building == "textures":
                    continue
                ridge_txt = os.path.join(root, building, "{}.obj.ridge.txt".format(building))
                pts_file = root_dir + "/samplePoints/point_cloud_10000K/{}.pts".format(building)
                outfile = os.path.join(obj_dir, building, "{}.obj.ridge.ext.txt".format(building))
                if os.path.exists(ridge_txt) and os.path.exists(pts_file):
                    sampled_xyz = np.loadtxt(pts_file)
                    avg_min_distance = get_avg_min_distance_of_sampled_points(sampled_xyz)
                    ridge_xyz = get_ridge_or_valley_vertices(ridge_txt, avg_min_distance)
                    np.savetxt(outfile, ridge_xyz)
                    return
