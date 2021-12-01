import torch
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import open3d as o3d
from PIL import Image

from utils.open3d_render import render_geometries
from utils.open3d_utils import TriangleMesh
from utils import voxel_renderer, recover_voxel, CameraJsonPosition
from common_utils.utils import normalize_vertices

def visualize_paired_patches_pytorch(generated_shape, content_shape,
                                     generated_patches, content_patches, xyz_patches, thr=0.):

    all_meshes = []

    # generated_shape_with_bbox = torch.clone(generated_shape)
    # generated_shape_with_bbox[0, :, :] = 1
    # generated_shape_with_bbox[:, 0, :] = 1
    # generated_shape_with_bbox[:, :, 0] = 1
    # generated_shape_with_bbox[generated_shape_with_bbox.shape[0]-1, :, :] = 1
    # generated_shape_with_bbox[:, generated_shape_with_bbox.shape[1]-1, :] = 1
    # generated_shape_with_bbox[:, :, generated_shape_with_bbox.shape[2]-1] = 1
    # vertices, triangles = mcubes.marching_cubes(generated_shape_with_bbox.detach().cpu().numpy(), thr)
    vertices, triangles = mcubes.marching_cubes(generated_shape.detach().cpu().numpy(), thr)
    max_x = np.max(vertices, 0)[0]
    vertices -= np.array([max_x*2, 0, 0])
    m_coa = TriangleMesh(vertices, triangles)
    m_coa.compute_vertex_normals()
    m_coa.paint_uniform_color(np.array([236, 228, 230])/255.)
    all_meshes.append(m_coa)

    # content_shape_with_bbox = torch.clone(content_shape)
    # content_shape_with_bbox[0, :, :] = 1
    # content_shape_with_bbox[:, 0, :] = 1
    # content_shape_with_bbox[:, :, 0] = 1
    # content_shape_with_bbox[content_shape_with_bbox.shape[0]-1, :, :] = 1
    # content_shape_with_bbox[:, content_shape_with_bbox.shape[1]-1, :] = 1
    # content_shape_with_bbox[:, :, content_shape_with_bbox.shape[2]-1] = 1
    # vertices, triangles = mcubes.marching_cubes(content_shape_with_bbox.detach().cpu().numpy(), thr)
    vertices, triangles = mcubes.marching_cubes(content_shape.detach().cpu().numpy(), thr)
    m_coa = TriangleMesh(vertices, triangles)
    m_coa.compute_vertex_normals()
    m_coa.paint_uniform_color(np.array([236, 228, 230])/255.)
    all_meshes.append(m_coa)

    for p in range(len(content_patches)):
        patch_content = content_patches[p]
        patch_fake = generated_patches[p]
        patch_xyz_min = xyz_patches[p, 0].min((0, 1, 2))
        color = np.random.random((3, 1))

        vertices, triangles = mcubes.marching_cubes(patch_content.detach().cpu().numpy(), thr)
        vertices += patch_xyz_min
        vertices += np.array([max_x, 0, 0])
        m_coa = TriangleMesh(vertices, triangles)
        m_coa.compute_vertex_normals()
        m_coa.paint_uniform_color(color)
        all_meshes.append(m_coa)

        vertices, triangles = mcubes.marching_cubes(patch_fake.detach().cpu().numpy(), thr)
        vertices += patch_xyz_min
        vertices -= np.array([max_x, 0, 0])
        m_coa = TriangleMesh(vertices, triangles)
        m_coa.compute_vertex_normals()
        m_coa.paint_uniform_color(color)
        all_meshes.append(m_coa)

    o3d.visualization.draw_geometries(all_meshes)


voxel_renderer = voxel_renderer(256)


def visualize_generated_with_patches_pytorch(generated_shape, generated_patches, xyz_patches, vr=voxel_renderer, thr=0.):

    all_meshes = []

    vertices, triangles = mcubes.marching_cubes(generated_shape.detach().cpu().numpy(), thr)
    if len(vertices) == 0:
        return np.ones((vr.render_IO_vox_size, vr.render_IO_vox_size*2)).astype(np.uint8) * 255
    max_x = np.max(vertices, 0)[0]
    vertices -= np.array([max_x*2, 0, 0])
    vertices = normalize_vertices(vertices)
    vertices = vertices-np.array([0.7, 0, 0])
    m_coa = TriangleMesh(vertices, triangles)
    m_coa.compute_vertex_normals()
    m_coa.paint_uniform_color(np.array([236, 228, 230])/255.)
    all_meshes.append(m_coa)

    for p in range(len(generated_patches)):
        patch_fake = generated_patches[p]
        patch_xyz_min = xyz_patches[p, 0].min((0, 1, 2))
        color = np.random.random((3, 1))

        vertices, triangles = mcubes.marching_cubes(patch_fake.detach().cpu().numpy(), thr)
        vertices += patch_xyz_min
        vertices -= np.array([max_x, 0, 0])
        vertices = normalize_vertices(vertices)
        vertices = vertices + np.array([0.3, 0, 0])
        m_coa = TriangleMesh(vertices, triangles)
        m_coa.compute_vertex_normals()
        m_coa.paint_uniform_color(color)
        all_meshes.append(m_coa)

    # o3d.visualization.draw_geometries(all_meshes)
    img_gen = render_geometries(all_meshes, camera_json=CameraJsonPosition, out_img=True)
    img_gen = Image.fromarray(np.uint8(np.asarray(img_gen) * 255))
    return img_gen


def render_paired_patches_pytorch(generated_shape, content_shape,
                                  generated_patches, content_patches, xyz_patches,
                                  pos_content, thr=0.):

    patch_size = generated_patches.shape[1]
    content_patches_combined = torch.zeros_like(content_shape)
    generated_patches_combined = torch.zeros_like(generated_shape)
    for p in range(len(content_patches)):
        patch_content = content_patches[p]
        patch_fake = generated_patches[p]
        patch_xyz_min = xyz_patches[p, 0].min((0, 1, 2))
        content_patches_combined[patch_xyz_min[0]:patch_xyz_min[0]+patch_size,
                                 patch_xyz_min[1]:patch_xyz_min[1]+patch_size,
                                 patch_xyz_min[2]:patch_xyz_min[2]+patch_size] = patch_content
        generated_patches_combined[patch_xyz_min[0]:patch_xyz_min[0]+patch_size,
                                   patch_xyz_min[1]:patch_xyz_min[1]+patch_size,
                                   patch_xyz_min[2]:patch_xyz_min[2]+patch_size] = patch_fake

    generated_shape = generated_shape.detach().cpu().numpy()
    content_shape = content_shape.detach().cpu().numpy()
    generated_patches_combined = generated_patches_combined.detach().cpu().numpy()
    content_patches_combined = content_patches_combined.detach().cpu().numpy()

    recovered_content = recover_voxel(content_shape, *pos_content, 256, 8, 8, True)
    recovered_generated = recover_voxel(generated_shape, *pos_content, 256, 8, 8, True)
    recovered_content_patches = recover_voxel(content_patches_combined, *pos_content, 256, 8, 8, True)
    recovered_generated_patches = recover_voxel(generated_patches_combined, *pos_content, 256, 8, 8, True)

    _render_paired_patches_pytorch(recovered_generated, recovered_content,
                                   recovered_generated_patches, recovered_content_patches,
                                   len(generated_patches), len(content_patches), patch_size, thr)


def render_generated_with_patches_pytorch(generated_shape, generated_patches, xyz_patches,
                                          pos_content, thr=0., vert=True):

    patch_shape = np.array(generated_patches[0].shape)
    generated_patches_combined = torch.zeros_like(generated_shape)
    for p in range(len(generated_patches)):
        patch_fake = generated_patches[p]
        patch_xyz_min = xyz_patches[p, 0].min((0, 1, 2))
        generated_patches_combined[patch_xyz_min[0]:patch_xyz_min[0]+patch_shape[0],
                                   patch_xyz_min[1]:patch_xyz_min[1]+patch_shape[1],
                                   patch_xyz_min[2]:patch_xyz_min[2]+patch_shape[2]] = patch_fake

    generated_shape = generated_shape.detach().cpu().numpy()
    generated_patches_combined = generated_patches_combined.detach().cpu().numpy()

    recovered_generated = recover_voxel(generated_shape, *pos_content, 256, 8, 8, True)
    recovered_generated_patches = recover_voxel(generated_patches_combined, *pos_content, 256, 8, 8, True)

    return _render_generated_with_patches_pytorch(recovered_generated, recovered_generated_patches,
                                                  len(generated_patches), thr=thr, vert=vert)


def _render_paired_patches_pytorch(generated_vox: np.ndarray, content_vox: np.ndarray,
                                   generated_patches_vox: np.ndarray, content_patches_vox: np.ndarray,
                                   num_gen_patches, num_cont_patches, patch_size: int, thr: float, vr=voxel_renderer):

    assert generated_vox.shape[0] == generated_vox.shape[1] == generated_vox.shape[2] == vr.render_IO_vox_size
    assert content_vox.shape[0] == content_vox.shape[1] == content_vox.shape[2] == vr.render_IO_vox_size
    assert generated_patches_vox.shape[0] == generated_patches_vox.shape[1] == generated_patches_vox.shape[2] == vr.render_IO_vox_size
    assert content_patches_vox.shape[0] == content_patches_vox.shape[1] == content_patches_vox.shape[2] == vr.render_IO_vox_size

    gen_shape_img = vr.render_img(generated_vox, thr, 0)
    con_shape_img = vr.render_img(content_vox, thr, 0)
    gen_patches_shape_img = vr.render_img(generated_patches_vox, thr, 0)
    con_patches_shape_img = vr.render_img(content_patches_vox, thr, 0)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(con_shape_img), ax1.set_title(f"content shape")
    ax2.imshow(gen_shape_img), ax2.set_title(f"generated shape")
    ax3.imshow(con_patches_shape_img), ax3.set_title(f"{num_cont_patches} content patches (dim {patch_size})")
    ax4.imshow(gen_patches_shape_img), ax4.set_title(f"{num_gen_patches} generated patches (dim {patch_size})")

    plt.tight_layout()
    plt.show()


def _render_generated_with_patches_pytorch(generated_vox: np.ndarray, generated_patches_vox: np.ndarray,
                                           num_gen_patches, thr: float, vr=voxel_renderer, vert=True):

    assert generated_vox.shape[0] == generated_vox.shape[1] == generated_vox.shape[2] == vr.render_IO_vox_size
    assert generated_patches_vox.shape[0] == generated_patches_vox.shape[1] == generated_patches_vox.shape[2] == vr.render_IO_vox_size

    gen_shape_img = vr.render_img(generated_vox, thr, 0)
    gen_patches_shape_img = vr.render_img(generated_patches_vox, thr, 0)

    if vert:
        fig, axes = plt.subplots(2, 1)
        img_shape = (vr.render_IO_vox_size, vr.render_IO_vox_size*2)
    else:
        fig, axes = plt.subplots(1, 2)
        img_shape = (vr.render_IO_vox_size*2, vr.render_IO_vox_size)

    axes[0].imshow(gen_shape_img), axes[0].set_title(f"generated shape")
    axes[1].imshow(gen_patches_shape_img), axes[1].set_title(f"{num_gen_patches} generated patches")

    plt.tight_layout()

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_from_plot = Image.fromarray(image_from_plot)
    # PIL.Image and np.array have reversed dimensions
    image_from_plot = np.asarray(image_from_plot.resize(img_shape).convert('L'))
    plt.close()
    return image_from_plot
