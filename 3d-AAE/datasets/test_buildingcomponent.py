from unittest import TestCase

import numpy as np

from buildingcomponent import *

import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class TestAnnfassComponentDataset(TestCase):
    def test_me(self):
        # inp_path = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/"
        # # inp_path += "style_detection/logs/annfass_content_style_splits/style"
        # inp_path += "style_detection/logs/buildnet_content_style_splits/style"
        # dset = BuildingComponentDataset(inp_path, split='train', n_points=2048)
        # for i in range(5):
        #     xyz, other = dset.__getitem__(i)
        #     print(xyz, other)
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(xyz)
        #     o3d.visualization.draw_geometries([pcd],)
        inp_path = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/samplePoints/stylePly_cut10.0K_pgc_style4096"
        dset = BuildingComponentRawDataset(inp_path, split='train', n_points=2048)
        for i in range(5):
            xyz, other = dset.__getitem__(i)
            print(xyz, other)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            o3d.visualization.draw_geometries([pcd],)


class TestBuildingComponent2Dataset(TestCase):
    def test_load_ply_with_color(self):
        xyz, rgb = load_ply("/media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/samplePoints_refinedTextures/groups_june17_colorPly_cut10.0K_pgc/COMMERCIALcastle_mesh0904/COMMERCIALcastle_mesh0904_group2_688_673_window__unknown.ply",
                          with_color=True)
        print(xyz, rgb)
        if 1 < rgb.max() < 256 and 0 <= rgb.min():
            rgb = rgb / 255.
        print(xyz, rgb)
        features = np.hstack([xyz, rgb])
        print(features.shape)

    def test_me(self):
        txt_file = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/style_detection/logs_final/unlabeled_data/setA_train_val_test/withcolor/BUILDNET_Buildings_groups_june17_colorPly_cut10.0K_pgc/unique/columndomedoorwindowtower/split_train_val_test/all.txt"
        dset = BuildingComponentDataset2WithColor(n_points=2048,
                                                 data_root="/media/graphicslab/BigData1/zavou/ANNFASS_DATA",
                                                 txt_file=txt_file)
        for i in range(5):
            xyz, other = dset.__getitem__(i)
            print(xyz, other)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            o3d.visualization.draw_geometries([pcd],)
