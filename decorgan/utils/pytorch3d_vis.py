import os
import torch
import matplotlib.pyplot as plt

from pytorch3d.io import load_objs_as_meshes, load_obj

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    PointsRasterizationSettings,
    MeshRenderer,
    PointsRenderer,
    MeshRasterizer,
    PointsRasterizer,
    SoftPhongShader,
    AlphaCompositor,
    TexturesVertex
)
from utils.matplotlib_utils import image_grid, render_views
import numpy as np
from utils.io_helper import parse_xyz_rgb_ply


def get_mesh_from_file(obj_filename, device, load_textures=True):
    mesh = load_objs_as_meshes([obj_filename], device=device, load_textures=load_textures)
    mesh = Meshes(verts=mesh.verts_list(), faces=mesh.faces_list(),
                  textures=TexturesVertex(verts_features=[
                      torch.ones_like(torch.ones_like(mesh.verts_list()[0]))]))
    return mesh


def get_point_cloud_from_file(ply_filename, device):
    xyzs, rgbs = parse_xyz_rgb_ply(ply_filename)
    verts = torch.Tensor(xyzs).to(device)
    rgbs = rgbs / 255.
    rgb = torch.Tensor(rgbs).to(device)
    point_cloud = Pointclouds(points=[verts], features=[rgb])
    return point_cloud


def get_mesh_from_verts_and_triangles(verts, triangles, device):
    verts = torch.from_numpy(verts).type(torch.float32)
    triangles = torch.from_numpy(triangles).type(torch.int64)
    verts_rgb = torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh = Meshes(verts=[verts.to(device)],
                  faces=[triangles.to(device)],
                  textures=textures)
    return mesh


class CustomMeshRenderer():

    def __init__(self, device):

        self.device = device

        self.dist, self.elev, self.azim = 2.7, 0, 180
        R, T = look_at_view_transform(dist=self.dist, elev=self.elev, azim=self.azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=self.lights
            )
        )


class CustomMultiViewMeshRenderer(CustomMeshRenderer):

    def __init__(self, device):
        super().__init__(device)

    def __call__(self, obj_filename=None, mesh=None, verts=None, triangles=None,
                 batch_size=20, show=False, nrows=None, ncols=None):
        assert (not mesh is None) or (not obj_filename is None) or (not verts is None and not triangles is None)
        if obj_filename:
            mesh = get_mesh_from_file(obj_filename, self.device)
        elif not mesh:
            mesh = get_mesh_from_verts_and_triangles(verts, triangles, self.device)

        meshes = mesh.extend(batch_size)

        elev = torch.linspace(0, 180, batch_size)
        azim = torch.linspace(-180, 180, batch_size)

        R, T = look_at_view_transform(dist=1.5, elev=elev, azim=azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.lights.location = torch.tensor([[0.0, 0.0, -1.5]], device=self.device)
        images = self.renderer(meshes, cameras=self.cameras, lights=self.lights)

        if nrows is None:
            nrows = 2
        if ncols is None:
            ncols = int(batch_size/2)
        fig = image_grid(images.cpu().numpy(), rows=nrows, cols=ncols, rgb=True)
        if show:
            plt.show()

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image_from_plot


class CustomDefinedViewMeshRenderer(CustomMeshRenderer):

    def __init__(self, device, num_views=6, vertical=True):
        super(CustomDefinedViewMeshRenderer, self).__init__(device)
        assert num_views % 2 == 0
        self.num_views = num_views
        self.elev = torch.linspace(0, 180, self.num_views)
        self.azim = torch.linspace(-180, 180, self.num_views)
        self.vertical = vertical

    def __call__(self, obj_filename=None, mesh=None, verts=None, triangles=None,
                 view_idx=None, show=False):
        assert (not mesh is None) or (not obj_filename is None) or (not verts is None and not triangles is None)
        if obj_filename:
            mesh = get_mesh_from_file(obj_filename, self.device)
        elif not mesh:
            mesh = get_mesh_from_verts_and_triangles(verts, triangles, self.device)

        if view_idx is not None:
            elev, azim = self.elev[view_idx], self.azim[view_idx]
        else:
            mesh = mesh.extend(self.num_views)
            elev, azim = self.elev, self.azim

        R, T = look_at_view_transform(dist=1.5, elev=elev, azim=azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        images = self.renderer(mesh, cameras=self.cameras, lights=self.lights)

        if view_idx is not None:
            img = images[0, ..., :3].cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.grid("off")
            plt.axis("off")
            if show:
                plt.show()
        else:
            images = images[:, ..., :3].cpu().numpy()
            if self.vertical:
                img = render_views(images, show=show)
            else:
                fig = image_grid(images, rows=2, cols=self.num_views//2)
                if show:
                    plt.show()
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()
        return img


class CustomSingleViewMeshRenderer(CustomMeshRenderer):

    def __init__(self, device, dist=1.5, elev=30, azim=120):
        super(CustomSingleViewMeshRenderer, self).__init__(device)
        self.dist = dist
        self.elev = elev
        self.azim = azim

    def __call__(self, obj_filename=None, mesh=None, verts=None, triangles=None, show=False,
                 dist=None, elev=None, azim=None):
        assert (not mesh is None) or (not obj_filename is None) or (not verts is None and not triangles is None)
        if obj_filename:
            mesh = get_mesh_from_file(obj_filename, self.device)
        elif not mesh:
            mesh = get_mesh_from_verts_and_triangles(verts, triangles, self.device)

        R, T = look_at_view_transform(dist=dist or self.dist, elev=elev or self.elev, azim=azim or self.azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        images = self.renderer(mesh, cameras=self.cameras, lights=self.lights)

        plt.figure(figsize=(10, 10))
        plt.imshow(images[0, ..., :3].cpu().numpy())
        plt.grid("off")
        plt.axis("off")

        if show:
            plt.show()
        plt.close()
        return images[0, ..., :3].cpu().numpy()


def join_two_meshes(obj_file1, obj_file2, device):
    mesh1 = get_mesh_from_file("/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/buildnet_component_v2_orient_debug/RELIGIOUSchurch_mesh1270/style_mesh_group22_47_23_tower_steeple__unknown/model.obj", device)
    mesh2 = get_mesh_from_file("/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/buildnet_component_v2_orient_debug/RELIGIOUSchurch_mesh1270/style_mesh_group24_80_47_tower_steeple__unknown/model.obj", device)

    verts1, faces1 = mesh1.get_mesh_verts_faces(0)
    verts2, faces2 = mesh2.get_mesh_verts_faces(0)

    verts_rgb1 = torch.ones_like(verts1)
    verts_rgb2 = torch.ones_like(verts2)
    verts_rgb1[:, 1:] *= 0.  # red
    verts_rgb2[:, :2] *= 0.  # blue

    verts1 *= 0.25
    verts1[:, 0] += 0.8
    verts2[:, 0] -= 0.5
    verts = torch.cat([verts1, verts2])  #(3204, 3)

    #  Offset by the number of vertices in mesh1
    faces2 = faces2 + verts1.shape[0]
    faces = torch.cat([faces1, faces2])  # (6400, 3)

    verts_rgb = torch.cat([verts_rgb1, verts_rgb2])[None]  # (1, 204, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)
    return mesh


def join_voxel_and_bbox(vertices_vox, triangles_vox, vertices_bbox, triangles_bbox, device):

    mesh1 = get_mesh_from_verts_and_triangles(vertices_vox, triangles_vox, device)
    mesh2 = get_mesh_from_verts_and_triangles(vertices_bbox, triangles_bbox, device)
    verts1, faces1 = mesh1.get_mesh_verts_faces(0)
    verts2, faces2 = mesh2.get_mesh_verts_faces(0)

    verts_rgb1 = torch.ones_like(verts1)
    verts_rgb2 = torch.ones_like(verts2)
    verts_rgb1[:, 1:] *= 0.  # red
    verts_rgb2[:, :2] *= 0.  # blue

    verts = torch.cat([verts1, verts2])  #(3204, 3)

    #  Offset by the number of vertices in mesh1
    faces2 = faces2 + verts1.shape[0]
    faces = torch.cat([faces1, faces2])  # (6400, 3)

    verts_rgb = torch.cat([verts_rgb1, verts_rgb2])[None]  # (1, 204, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)
    return mesh


class CustomPointsRenderer():

    def __init__(self):

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            # torch.cuda.set_device(device)
        else:
            self.device = torch.device("cpu")

        self.dist, self.elev, self.azim = 2.7, 0, 180
        R, T = look_at_view_transform(dist=self.dist, elev=self.elev, azim=self.azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius=0.003,
            points_per_pixel=10
        )

        self.renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            compositor=AlphaCompositor(background_color=(1, 184 / 255, 51 / 255))
        )


class CustomDefinedViewPointsRenderer(CustomPointsRenderer):

    def __init__(self, num_views=6, vertical=True):
        super().__init__()
        assert num_views % 2 == 0
        self.num_views = num_views
        self.elev = torch.linspace(0, 180, self.num_views)
        self.azim = torch.linspace(-180, 180, self.num_views)
        self.vertical = vertical

    def __call__(self, ply_filename=None, point_cloud=None, verts=None, rgbs=None,
                 view_idx=None, show=False):
        assert (not point_cloud is None) or (not ply_filename is None) or (not verts is None and not rgbs is None)
        if ply_filename:
            point_cloud = get_point_cloud_from_file(ply_filename, self.device)
        elif not point_cloud:
            point_cloud = Pointclouds(points=[torch.Tensor(verts).to(self.device)],
                                      features=[torch.Tensor(rgbs).to(self.device)])

        if view_idx is not None:
            elev, azim = self.elev[view_idx], self.azim[view_idx]
        else:
            point_cloud = point_cloud.extend(self.num_views)
            elev, azim = self.elev, self.azim

        R, T = look_at_view_transform(dist=1.5, elev=elev, azim=azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        images = self.renderer(point_cloud, cameras=self.cameras)

        if view_idx is not None:
            img = images[0, ..., :3].cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.grid("off")
            plt.axis("off")
            if show:
                plt.show()
        else:
            images = images.cpu().numpy()
            if self.vertical:
                img = render_views(images, show=show)
            else:
                fig = image_grid(images, rows=2, cols=self.num_views//2)
                if show:
                    plt.show()
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()
        return img


if __name__ == '__main__':

    from io_helper import parse_simple_obj_file

    cdvmr = CustomDefinedViewMeshRenderer(4)
    # mesh = load_objs_as_meshes(["/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/group_debugeevee/RELIGIOUScathedral_mesh0085/grouped.obj"],
    mesh = load_objs_as_meshes(["/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/unifiedgroups17/RELIGIOUSchurch_mesh2460/style_mesh_group1_18_18_window__unknown/model.obj"],
                               cdvmr.device,
                               load_textures=True,
                               create_texture_atlas=True)
    cdvmr(mesh=mesh, show=True)

    # cdvpr = CustomDefinedViewPointsRenderer()
    # cdvpr(ply_filename="/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/samplePoints_refinedTextures/colorPly_1000K/COMMERCIALhotel_building_mesh0139.ply",
    #       show=True)

    # cmvr = CustomMultiViewMeshRenderer()
    # csvr = CustomSingleViewMeshRenderer()
    # directory = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/debug/RELIGIOUSchurch_mesh2430"
    # i = 0
    # for component_dir in os.listdir(directory):
    #     i += 1
    #     obj_file = os.path.join(directory, component_dir, "model.obj")
    #     cmvr(obj_file, 10, show=True)
    #     # csvr(obj_filename=obj_file, show=True)
    #     # v, f = parse_simple_obj_file(obj_file)
    #     # csvr(verts=v, triangles=f, show=True, dist=1.5, elev=30, azim=120)
    #     # csvr(verts=v, triangles=f, show=True, dist=1.5, elev=30, azim=60)  # no difference ... why?
    #     if i == 5:
    #         break

    # mesh_ = join_two_meshes(None, None, torch.device('cuda'))
    # CustomSingleViewRenderer()(mesh=mesh_, show=True)
    # img = CustomMultiViewRenderer()(mesh=mesh_, show=False, batch_size=3, ncols=1, nrows=3)

    # from matplotlib_utils import render_example
    # render_example([img, img, img], show=True, figsize=(10,10))

    # img1 = CustomDefinedViewRenderer(4)(mesh=mesh_, show=False, view_idx=0)
    # img2 = CustomDefinedViewRenderer(4)(mesh=mesh_, show=False, view_idx=1)
    # img3 = CustomDefinedViewRenderer(4)(mesh=mesh_, show=False, view_idx=2)
    # img4 = CustomDefinedViewRenderer(4)(mesh=mesh_, show=False, view_idx=3)
    # render_views(([img1, img2, img3, img4]), show=True)
    # img_all4 = CustomDefinedViewMeshRenderer(4)(mesh=mesh_, show=False)
    # plt.figure()
    # plt.imshow(img_all4)
    # plt.show()
    #
    # img_all4 = CustomDefinedViewMeshRenderer(4)(mesh=mesh_, show=True)
