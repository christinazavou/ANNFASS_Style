import time
import os

import torch
import torch.nn.functional as F
import numpy as np
import sys

sys.path.extend(os.path.dirname(os.path.abspath(__file__)))
from common_utils import binvox_rw_faster as binvox_rw
import cutils
import mcubes
import open3d as o3d
import random

from common_utils.utils import normalize_vertices
from utils import CameraJsonPosition, recover_voxel, get_voxel_bbox
from utils.open3d_render import render_geometries
from utils.open3d_utils import TriangleMesh
# from utils.patch_gen import get_random_paired_patches, get_random_patches
from utils.patch_gen_2 import get_random_paired_patches, get_random_patches, get_random_paired_non_cube_patches, \
    get_random_non_cube_patches
from utils.patch_gen_2new import get_random_paired_non_cube_patches as rpncp
from utils.patch_gen_2new import get_random_non_cube_patches as rncp
from utils.patch_vis import visualize_paired_patches_pytorch, render_paired_patches_pytorch


def apply_random_crop_on_images(x, target_size, scale_range, num_crops=1, return_rect=False):
    # target_size = size of patch in one dimension

    # build grid
    B = x.size(0) * num_crops
    flip = torch.round(torch.rand(B, 1, 1, 1, device=x.device)) * 2 - 1.0  # [B, 1, 1, 1] with values in {-1,1}
    unit_grid_x = torch.linspace(-1.0, 1.0, target_size, device=x.device)[np.newaxis, np.newaxis, :, np.newaxis]\
        .repeat(B, target_size, 1, 1)  # [B, target_size, target_size, 1] with each element in dimension 1 and 2 lying in equal positions within [-1,1] specifying patch positions x and y
    unit_grid_y = unit_grid_x.transpose(1, 2)  # same as above (with y and x in dimensions 1 and 2)
    unit_grid = torch.cat([unit_grid_x * flip, unit_grid_y], dim=3)


    #crops = []
    x = x.unsqueeze(1).expand(-1, num_crops, -1, -1, -1).flatten(0, 1)  # [B, channels, width, height]
    #for i in range(num_crops):
    scale = torch.rand(B, 1, 1, 2, device=x.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    offset = (torch.rand(B, 1, 1, 2, device=x.device) * 2 - 1) * (1 - scale)
    sampling_grid = unit_grid * scale + offset
    crop = F.grid_sample(x, sampling_grid, align_corners=False)
    #crops.append(crop)
    #crop = torch.stack(crops, dim=1)
    crop = crop.view(B // num_crops, num_crops, crop.size(1), crop.size(2), crop.size(3))

    return crop


def check_style_patches(style_shape, thr=0.):

    buffer_size = 256*256*16 #change the buffer size if the input voxel is large
    patch_size = 12
    padding_size = 0

    all_meshes = []

    style_shape = style_shape.astype(np.uint8)
    vertices, triangles = mcubes.marching_cubes(style_shape, thr)
    m_coa = TriangleMesh(vertices, triangles)
    m_coa.compute_vertex_normals()
    m_coa.paint_uniform_color(np.array([236, 228, 230])/255.)
    all_meshes.append(m_coa)

    style_shape = np.ascontiguousarray(style_shape[:,:,padding_size:])

    style_shape_tensor = torch.from_numpy(style_shape).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0).float()
    style_shape_edge_tensor = F.max_pool3d(-style_shape_tensor, kernel_size = 3, stride = 1, padding = 1) + style_shape_tensor
    style_shape_dilated_tensor = F.max_pool3d(style_shape_edge_tensor, kernel_size = 3, stride = 1, padding = 1)
    style_shape_edge = style_shape_edge_tensor.detach().cpu().numpy()[0,0]
    style_shape_edge = np.round(style_shape_edge).astype(np.uint8)
    style_shape_dilated = style_shape_dilated_tensor.detach().cpu().numpy()[0,0]
    style_shape_dilated = np.round(style_shape_dilated).astype(np.uint8)

    patches = np.zeros([buffer_size,patch_size,patch_size,patch_size], np.uint8)
    patches_edge = np.zeros([buffer_size,patch_size,patch_size,patch_size], np.uint8)
    patches_dilated = np.zeros([buffer_size,patch_size,patch_size,patch_size], np.uint8)
    patch_num = cutils.get_patches_edge_dilated(style_shape,style_shape_edge,style_shape_dilated,patches,patches_edge,patches_dilated,patch_size)

    print(f"found {patch_num}")
    for p in range(10):
        patch = patches[p]
        vertices, triangles = mcubes.marching_cubes(patch, thr)
        # vertices = normalize_vertices(vertices)
        m_coa = TriangleMesh(vertices, triangles)
        m_coa.compute_vertex_normals()
        m_coa.paint_uniform_color(np.random.random((3, 1)))
        # render_geometries([m_coa], camera_json=CameraJsonPosition, out_img=False, out_file=False)
        all_meshes.append(m_coa)
    o3d.visualization.draw_geometries(all_meshes)


def get_useful_random_patches(img, patch_size, patch_num):
    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]
    margin_size = (patch_size-2)//2

    i_j_k = np.zeros((patch_num, 3))
    patches = np.zeros((patch_num, patch_size, patch_size, patch_size))

    for p in range(patch_num):
        found = False
        while not found:
            i = np.random.randint(margin_size, dimx - margin_size - 1)
            j = np.random.randint(margin_size, dimy - margin_size - 1)
            k = np.random.randint(margin_size, dimz - margin_size - 1)

            if (img[i,j,k]==0 or img[i,j,k+1]==0 or img[i,j+1,k]==0 or
                img[i,j+1,k+1]==0 or img[i+1,j,k]==0 or img[i+1,j,k+1]==0 or
                img[i+1,j+1,k]==0 or img[i+1,j+1,k+1]==0) and \
                    (img[i,j,k]!=0 or img[i,j,k+1]!=0 or img[i,j+1,k]!=0 or
                     img[i,j+1,k+1]!=0 or img[i+1,j,k]!=0 or img[i+1,j,k+1]!=0 or
                     img[i+1,j+1,k]!=0 or img[i+1,j+1,k+1]!=0):
                patches[p] = img[i-margin_size:i+margin_size+2,
                                 j-margin_size:j+margin_size+2,
                                 k-margin_size:k+margin_size+2]
                i_j_k[p, 0] = i
                i_j_k[p, 1] = j
                i_j_k[p, 2] = k

                found = True

    return patches, i_j_k


def check_random_style_patches(style_shape, patch_size=32, patch_num=16, thr=0.):

    padding_size = 0
    centers = np.zeros((patch_num, 3)).astype(np.int32)
    all_meshes = []

    vertices, triangles = mcubes.marching_cubes(style_shape, thr)
    avg_v = np.mean(vertices, 0)
    avg_v = np.array([avg_v[0]*2, 0, 0])
    vertices -= avg_v
    m_coa = TriangleMesh(vertices, triangles)
    m_coa.compute_vertex_normals()
    m_coa.paint_uniform_color(np.array([236, 228, 230])/255.)
    all_meshes.append(m_coa)

    s_time = time.time()
    patches = np.zeros([patch_num,patch_size,patch_size,patch_size], np.float32)
    centers = cutils.get_useful_random_patches(style_shape,patches,patch_size,patch_num,centers)
    print(f"with cutils: {time.time() - s_time}")
    patch_num = len(centers)

    s_time = time.time()
    patches_, centers_ = get_useful_random_patches(style_shape,patch_size,patch_num)
    print(f"with python: {time.time() - s_time}")

    patches_masks = get_patches_style_Dmask_tensor(patches)
    patches_masks = get_patches_content_voxel_Dmask_tensor(patches)

    print(f"found {patch_num}")
    for p in range(patch_num):
        patch = patches[p]
        color = np.random.random((3, 1))

        vertices, triangles = mcubes.marching_cubes(patch, thr)
        vertices += centers.base[p]
        m_coa = TriangleMesh(vertices, triangles)
        m_coa.compute_vertex_normals()
        m_coa.paint_uniform_color(color)
        all_meshes.append(m_coa)

        patch_Dmask = get_style_voxel_Dmask_tensor(patch)
        patch_Dmask = F.interpolate(patch_Dmask, scale_factor=8 // 2, mode='nearest')
        patch_Dmask = patch_Dmask.detach().cpu().numpy()[0, 0]
        vertices, triangles = mcubes.marching_cubes(patch_Dmask, thr)
        vertices += centers.base[p]
        vertices += avg_v
        m_coa = TriangleMesh(vertices, triangles)
        m_coa.compute_vertex_normals()
        m_coa.paint_uniform_color(color)
        all_meshes.append(m_coa)

    o3d.visualization.draw_geometries(all_meshes)


def check_random_patches_pytorch(style_shape, thr=0.):

    all_meshes = []

    style_shape = style_shape.astype(np.uint8)
    vertices, triangles = mcubes.marching_cubes(style_shape, thr)
    m_coa = TriangleMesh(vertices, triangles)
    m_coa.compute_vertex_normals()
    m_coa.paint_uniform_color(np.array([236, 228, 230])/255.)
    all_meshes.append(m_coa)

    style_shape = torch.from_numpy(style_shape).to(torch.device('cuda'))
    patches = get_random_patches(style_shape).detach().cpu().numpy()

    for p in range(len(patches)):
        patch = patches[p]
        vertices, triangles = mcubes.marching_cubes(patch, thr)
        # vertices = normalize_vertices(vertices)
        m_coa = TriangleMesh(vertices, triangles)
        m_coa.compute_vertex_normals()
        m_coa.paint_uniform_color(np.random.random((3, 1)))
        # render_geometries([m_coa], camera_json=CameraJsonPosition, out_img=False, out_file=False)
        all_meshes.append(m_coa)
    o3d.visualization.draw_geometries(all_meshes)


def check_random_non_cube_patches_pytorch(style_shape, thr=0.):

    all_meshes = []

    style_shape = style_shape.astype(np.uint8)
    vertices, triangles = mcubes.marching_cubes(style_shape, thr)
    m_coa = TriangleMesh(vertices, triangles)
    m_coa.compute_vertex_normals()
    m_coa.paint_uniform_color(np.array([236, 228, 230])/255.)
    all_meshes.append(m_coa)

    style_shape = torch.from_numpy(style_shape).to(torch.device('cuda'))
    patches = get_random_non_cube_patches(style_shape).detach().cpu().numpy()

    for p in range(len(patches)):
        patch = patches[p]
        vertices, triangles = mcubes.marching_cubes(patch, thr)
        # vertices = normalize_vertices(vertices)
        m_coa = TriangleMesh(vertices, triangles)
        m_coa.compute_vertex_normals()
        m_coa.paint_uniform_color(np.random.random((3, 1)))
        # render_geometries([m_coa], camera_json=CameraJsonPosition, out_img=False, out_file=False)
        all_meshes.append(m_coa)
    o3d.visualization.draw_geometries(all_meshes)


def check_random_paired_patches(generated_shape, content_shape, patch_size=32, patch_num=16, thr=0.):

    padding_size = 0
    centers = np.zeros((patch_num, 3)).astype(np.int32)
    all_meshes = []

    vertices, triangles = mcubes.marching_cubes(generated_shape, thr)
    avg_v = np.mean(vertices, 0)
    avg_v = np.array([avg_v[0], 0, 0])
    vertices -= avg_v*4
    m_coa = TriangleMesh(vertices, triangles)
    m_coa.compute_vertex_normals()
    m_coa.paint_uniform_color(np.array([236, 228, 230])/255.)
    all_meshes.append(m_coa)

    vertices, triangles = mcubes.marching_cubes(content_shape, thr)
    vertices -= avg_v*2
    m_coa = TriangleMesh(vertices, triangles)
    m_coa.compute_vertex_normals()
    m_coa.paint_uniform_color(np.array([236, 228, 230])/255.)
    all_meshes.append(m_coa)

    patches_content = np.zeros([patch_num,patch_size,patch_size,patch_size], np.uint8)
    patches_fake = np.zeros([patch_num,patch_size,patch_size,patch_size], np.float32)
    centers = cutils.get_useful_random_paired_patches(content_shape, generated_shape, patches_content, patches_fake, patch_size, patch_num, centers)
    patch_num = len(centers)

    patches_content_masks = get_patches_content_voxel_Dmask_tensor(patches_content)
    patches_content_masks = F.interpolate(patches_content_masks, scale_factor=8 // 2, mode='nearest')
    patches_content_masks = patches_content_masks.detach().cpu().numpy()

    for p in range(patch_num):
        patch_fake = patches_fake[p]
        patch_content = patches_content[p]
        patch_content_mask = patches_content_masks[p, 0]
        color = np.random.random((3, 1))

        vertices, triangles = mcubes.marching_cubes(patch_content, thr)
        vertices += centers.base[p]
        m_coa = TriangleMesh(vertices, triangles)
        m_coa.compute_vertex_normals()
        m_coa.paint_uniform_color(color)
        all_meshes.append(m_coa)

        vertices, triangles = mcubes.marching_cubes(patch_content_mask, thr)
        vertices += centers.base[p]
        vertices += avg_v*2
        m_coa = TriangleMesh(vertices, triangles)
        m_coa.compute_vertex_normals()
        m_coa.paint_uniform_color(color)
        all_meshes.append(m_coa)

        vertices, triangles = mcubes.marching_cubes(patch_fake, thr)
        vertices += centers.base[p]
        vertices += avg_v*4
        m_coa = TriangleMesh(vertices, triangles)
        m_coa.compute_vertex_normals()
        m_coa.paint_uniform_color(color)
        all_meshes.append(m_coa)

    o3d.visualization.draw_geometries(all_meshes)


def check_patches_given_centers(generated_shape, content_shape, patch_size=32, patch_num=16, thr=0.):
    centers = np.zeros((patch_num, 3)).astype(np.int32)
    patches_content = np.zeros([patch_num,patch_size,patch_size,patch_size], np.float32)
    centers = cutils.get_useful_random_patches(content_shape.astype(np.float32), patches_content,
                                               patch_size, patch_num, centers)

    centers = np.asarray(centers)
    margin_size = (patch_size-2)//2
    # can't index patches :(


def check_open3d_paired_patches_pytorch(generated_shape, content_shape, patch_size=32, patch_num=16, stride=8, thr=0.):

    content_patches, generated_patches, xyz_patches = get_random_paired_patches(
        content_shape, generated_shape, patch_size, patch_num, stride)

    visualize_paired_patches_pytorch(generated_shape, content_shape, generated_patches, content_patches, xyz_patches, thr)


def check_open3d_paired_non_cube_patches_pytorch(generated_shape, content_shape,
                                                 patch_factor=4, patch_num=4, stride_factor=2, thr=0.):

    content_patches, generated_patches, xyz_patches = get_random_paired_non_cube_patches(
        content_shape, generated_shape, patch_factor, patch_num, stride_factor)

    visualize_paired_patches_pytorch(generated_shape, content_shape, generated_patches, content_patches, xyz_patches, thr)


def check_open3d_pncp_pytorch(generated_shape, content_shape, patch_factor=4, patch_num=4, stride_factor=2, thr=0.):

    content_patches, generated_patches, xyz_patches = rpncp(
        content_shape, generated_shape, patch_factor, patch_num, stride_factor)
    if content_patches.shape[0] == 0:
        return

    input_fake_patches = F.max_pool3d(content_patches.float(), kernel_size=8, stride=8, padding=0)
    input_generated_patches = F.max_pool3d(generated_patches, kernel_size=8, stride=8, padding=0)
    input_xyz_patches = xyz_patches / 8
    input_generated_shape = F.max_pool3d(generated_shape.unsqueeze(0), kernel_size=8, stride=8, padding=0)[0]
    input_content_shape = F.max_pool3d(content_shape.unsqueeze(0).float(), kernel_size=8, stride=8, padding=0)[0]

    visualize_paired_patches_pytorch(input_generated_shape, input_content_shape,
                                     input_generated_patches, input_fake_patches, input_xyz_patches, thr)


def check_render_paired_patches_pytorch(generated_shape, content_shape, pos_content,
                                        patch_size=32, patch_num=16, stride=8, thr=0.):

    content_patches, generated_patches, xyz_patches = get_random_paired_patches(
        content_shape, generated_shape, patch_size, patch_num, stride)

    render_paired_patches_pytorch(generated_shape, content_shape,
                                  generated_patches, content_patches, xyz_patches,
                                  pos_content, thr=0.4)


def get_patches_style_Dmask_tensor(vox):
    # 256 -crop- 252 -maxpoolk14s2- 120
    crop_margin = 2
    kernel_size = 14
    vox_tensor = torch.from_numpy(
        vox[:, crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]).to(
        torch.device('cuda')).unsqueeze(1).float()
    smallmask_tensor = F.max_pool3d(vox_tensor, kernel_size=kernel_size, stride=2, padding=0)
    smallmask = torch.round(smallmask_tensor).type(torch.uint8)
    return smallmask


def get_style_voxel_Dmask_tensor(vox):
    # 256 -crop- 252 -maxpoolk14s2- 120
    crop_margin = 2
    kernel_size = 14
    vox_tensor = torch.from_numpy(
        vox[crop_margin:-crop_margin, crop_margin:-crop_margin, crop_margin:-crop_margin]).to(
        torch.device('cuda')).unsqueeze(0).unsqueeze(0).float()
    smallmask_tensor = F.max_pool3d(vox_tensor, kernel_size=kernel_size, stride=2, padding=0)
    smallmask = torch.round(smallmask_tensor).type(torch.uint8)
    return smallmask


def get_patches_content_voxel_Dmask_tensor(vox):
    device = torch.device('cuda')
    upsample_rate = 8
    crop_margin = 1
    vox_tensor = torch.from_numpy(vox).to(device).unsqueeze(1).float()
    # input
    smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=upsample_rate, stride=upsample_rate, padding=0)
    # Dmask
    smallmask_tensor = smallmaskx_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin,
                       crop_margin:-crop_margin]
    smallmask_tensor = F.interpolate(smallmask_tensor, scale_factor=upsample_rate // 2, mode='nearest')
    smallmask_tensor = torch.round(smallmask_tensor).type(torch.uint8)
    return smallmask_tensor


def check_style_and_Dmask(vox):
    all_meshes = []
    vertices, triangles = mcubes.marching_cubes(vox, 0.4)
    vertices = normalize_vertices(vertices)
    vertices -= np.array([0.5, 0, 0])
    m_coa = TriangleMesh(vertices, triangles)
    m_coa.compute_vertex_normals()
    m_coa.paint_uniform_color(np.array([236, 228, 230])/255.)
    all_meshes.append(m_coa)
    Dmask_tensor = get_style_voxel_Dmask_tensor(vox)
    vertices, triangles = mcubes.marching_cubes(Dmask_tensor.detach().cpu().numpy()[0, 0], 0.4)
    vertices = normalize_vertices(vertices)
    vertices += np.array([0.5, 0, 0])
    m_coa = TriangleMesh(vertices, triangles)
    m_coa.compute_vertex_normals()
    all_meshes.append(m_coa)
    o3d.visualization.draw_geometries(all_meshes)


if __name__ == '__main__':
    # image = np.random.random((2, 3, 256, 256))  # batch, channels, width, height
    # image = torch.from_numpy(image)
    # res = apply_random_crop_on_images(image, target_size=8, scale_range=(1/8, 1/4), num_crops=10)
    # voxel_model_file = open("/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/data/03001627/"
    #                         "1a6f615e8b1b5ae4dbbc9440457e303e/model_depth_fusion.binvox", 'rb')
    # style_shape_data = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=False).data
    # check_style_patches(style_shape_data)
    # check_random_style_patches(style_shape_data)
    # check_random_patches_pytorch(style_shape_data, thr=0.)  # note: this is not cropped, therefore many empty patches
    # could be retrieved

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__


    data_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/data/03001627"
    fpath = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/chair_fig3_content_style.txt"
    config = {"asymmetry": True, "gpu": 0}
    config = dotdict(config)
    from dataset3 import BasicDataset
    dset = BasicDataset(data_dir, fpath, config, None, filename='model_depth_fusion.binvox')
    for di in range(len(dset)):
        data_dict = dset.__getitem__(di)
        # tmp = data_dict['cropped']
        # tmp = data_dict['voxel_style']
        # check_style_and_Dmask(tmp)
        # check_random_style_patches(tmp, patch_size=64, patch_num=4, thr=0.4)
        # check_style_patches(tmp)

        tmp_content = data_dict['cropped']
        pos_content = data_dict['pos']
        # check_random_patches_pytorch(tmp_content, thr=0.)
        # check_random_non_cube_patches_pytorch(tmp_content, thr=0.)
        # continue
        from scipy.ndimage.filters import gaussian_filter
        tmp_generated = gaussian_filter(tmp_content.astype(np.float32), sigma=1)
        # check_random_paired_patches(tmp_generated, tmp_content)
        # check_patches_given_centers(tmp_generated, tmp_content)
        # check_open3d_paired_patches_pytorch(torch.from_numpy(tmp_generated).cuda(),
        #                                     torch.from_numpy(tmp_content).cuda(),
        #                                     patch_size=32, patch_num=4, stride=32)

        # check_open3d_paired_non_cube_patches_pytorch(torch.from_numpy(tmp_generated).cuda(),
        #                                              torch.from_numpy(tmp_content).cuda(),
        #                                              # patch_factor=4, patch_num=16, stride_factor=2)
        #                                              # patch_factor=2, patch_num=8, stride_factor=2)
        #                                              # patch_factor=1, patch_num=1, stride_factor=1)
        #                                              # patch_factor=3, patch_num=32, stride_factor=1)
        #                                              patch_factor=8, patch_num=32, stride_factor=1)

        check_open3d_pncp_pytorch(torch.from_numpy(tmp_generated).cuda(),
                                  torch.from_numpy(tmp_content).cuda(),
                                  patch_factor=2, patch_num=8, stride_factor=2)
                                  # patch_factor=4, patch_num=16, stride_factor=2)

        # check_render_paired_patches_pytorch(torch.from_numpy(tmp_generated).cuda(),
        #                                     torch.from_numpy(tmp_content).cuda(),
        #                                     pos_content,
        #                                     patch_size=32, patch_num=8, stride=16)

