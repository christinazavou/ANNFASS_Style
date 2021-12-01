from unittest import TestCase
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d

from datasets.Component import ComponentSamplesDataset, ComponentMeshDataset, ComponentObjDataset
from datasets.dataset_utils import collate_pointcloud_fn, collate_pointcloud_with_features_fn, parse_simple_obj_file, \
    TriangleMesh, resample_mesh
from utils.open3d_vis import render_geometries, render_four_point_clouds
from utils import *


class TestComponentSamplesDataset(TestCase):
    def test_me(self):

        data_dict = {
            "dataset": "ComponentSamplesDataset",
            "data_dir": "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/annfass_content_style_splits_forsvm/style",
            "shuffle": True,
            "transforms": ["rotate"],
            "num_workers": 0,
            "n_points": 2048,
            "resolution": 128,
            "density": 30000,
            "collate_fn": "collate_pointcloud_fn",
            "batch_size": 4
        }
        dset = ComponentSamplesDataset(data_dict['data_dir'], 'train', config=dotdict(data_dict))
        dataloader = DataLoader(dset,
                                batch_size=data_dict['batch_size'],
                                shuffle=data_dict['shuffle'],
                                num_workers=data_dict['num_workers'],
                                drop_last=True,
                                pin_memory=True,
                                collate_fn=collate_pointcloud_fn)
        for i, batch_data in enumerate(dataloader, 1):
            print(batch_data)
            return


class TestComponentMeshDatasetModelNet(TestCase):
    def test_me(self):

        data_dict = {
            "data_dir": "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/minkoski_pytorch/examples/data_and_logs/ModelNet40multiple",
            "shuffle": True,
            "transforms": ["rotate"],
            "num_workers": 0,
            "resolution": 128,
            "density": 30000,
            "collate_fn": "collate_pointcloud_fn",
            "batch_size": 4
        }
        dset = ComponentMeshDataset(data_dict['data_dir'], 'train', config=dotdict(data_dict))
        dataloader = DataLoader(dset,
                                batch_size=data_dict['batch_size'],
                                shuffle=data_dict['shuffle'],
                                num_workers=data_dict['num_workers'],
                                drop_last=True,
                                pin_memory=True,
                                collate_fn=collate_pointcloud_fn)

        for i, batch_data in enumerate(dataloader, 1):
            print(batch_data)
            # pcd1 = PointCloud(batch_data['xyzs'][0])
            # render_geometries([pcd1], f"out{i}.png", SCREEN_CAMERA_UNIT_CUBE_FILE)
            # return
            render_four_point_clouds([batch_data['xyzs'][0], batch_data['xyzs'][1],
                                      batch_data['xyzs'][2], batch_data['xyzs'][3]],
                                     # save_img=f"out{i}.png",
                                     camera_loc=SCREEN_CAMERA_4MODELNET2_FILE
                                     )
            if i == 4:
                break


class TestComponentObjDataset(TestCase):
    def test_me(self):

        data_dict = {
            "data_dir": "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components",
            "train_split_file": "../example.txt",
            "shuffle": True,
            "transforms": ["rotate"],
            "num_workers": 0,
            "resolution": 128,
            "density": 30000,
            "collate_fn": "collate_pointcloud_fn",
            "batch_size": 4
        }
        dset = ComponentObjDataset(data_dict['data_dir'], 'train', config=dotdict(data_dict))
        dataloader = DataLoader(dset,
                                batch_size=data_dict['batch_size'],
                                shuffle=data_dict['shuffle'],
                                num_workers=data_dict['num_workers'],
                                drop_last=True,
                                pin_memory=True,
                                collate_fn=collate_pointcloud_fn)
        for i, batch_data in enumerate(dataloader, 1):
            print(batch_data)
            # pcd1 = PointCloud(batch_data['xyzs'][0])
            # render_geometries([pcd1], f"out{i}.png", SCREEN_CAMERA_UNIT_CUBE_FILE)
            # return
            render_four_point_clouds([batch_data['xyzs'][0], batch_data['xyzs'][1],
                                      batch_data['xyzs'][2], batch_data['xyzs'][3]],
                                     # save_img=f"out{i}.png",
                                     camera_loc=SCREEN_CAMERA_4MODELNET2_FILE
                                     )
            if i == 4:
                break


class TestComponentSamplesDataset_collate_pointcloud_with_features_fn(TestCase):
    def test_me(self):

        data_dict = {
            "data_dir": "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/minkoski_pytorch/examples/data_and_logs/ModelNet40multiple",
            "shuffle": True,
            "transforms": ["rotate"],
            "num_workers": 0,
            "n_points": 2048,
            "resolution": 128,
            "density": 30000,
            "collate_fn": "collate_pointcloud_fn",
            "batch_size": 4
        }
        dset = ComponentMeshDataset(data_dict['data_dir'], 'train', config=dotdict(data_dict))
        dataloader = DataLoader(dset,
                                # batch_size=data_dict['batch_size'],
                                batch_size=1,
                                shuffle=data_dict['shuffle'],
                                num_workers=data_dict['num_workers'],
                                drop_last=True,
                                pin_memory=True,
                                # collate_fn=collate_pointcloud_with_features_fn)
                                collate_fn=collate_pointcloud_fn)
        dataloader = iter(dataloader)
        for i in range(3):
            d = next(dataloader)
            print(d)


class TestResampleDensity(TestCase):
    def test_me(self):
        obj_file = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components/02_Panagia_Chrysaliniotissa/style_mesh_group1_Roof__Byzantine_02_Mesh.010/model.obj"
        vertices, faces = parse_simple_obj_file(obj_file)
        vertices = np.array(vertices)
        faces = np.array(faces)
        mesh = TriangleMesh(vertices, faces)
        vmax = vertices.max(0, keepdims=True)
        vmin = vertices.min(0, keepdims=True)
        mesh.vertices = o3d.utility.Vector3dVector((vertices - vmin) / (vmax - vmin).max())
        print(vertices.shape)
        xyz = resample_mesh(mesh, density=40000)
        print(xyz.shape)
        xyz = resample_mesh(mesh, density=4000)
        print(xyz.shape)
        xyz = resample_mesh(mesh, density=400)
        print(xyz.shape)
