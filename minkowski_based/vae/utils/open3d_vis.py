from typing import List, Optional
import copy

import open3d as o3d
from datasets.dataset_utils import PointCloud, TriangleMesh
import numpy as np
from utils import SCREEN_CAMERA_POSITION_FILE, SCREEN_CAMERA_UNIT_CUBE_FILE
from packaging import version


def load_and_update_view_point(vis, filename):
    param = o3d.io.read_pinhole_camera_parameters(filename)
    intrinsic = param.intrinsic.intrinsic_matrix
    extrinsic = param.extrinsic

    ctr = vis.get_view_control()
    param = convert_to_open3d_param(intrinsic, extrinsic)
    if version.parse(o3d.__version__) > version.parse("0.7.0.0"):
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    else:  # no use of given viewpoint ...
        ctr.convert_from_pinhole_camera_parameters(param)
    vis.update_renderer()


def convert_to_open3d_param(intrinsic, extrinsic):
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
    param.intrinsic.intrinsic_matrix = intrinsic
    param.extrinsic = extrinsic
    return param


def render_geometries(geometries: List[o3d.geometry.Geometry],
                      out_file: Optional[str] = None,
                      camera_json: Optional[str] = None):

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)
    vis.poll_events()
    vis.update_renderer()

    for geom in geometries:
        vis.add_geometry(geom)

    if camera_json:
        load_and_update_view_point(vis, camera_json)

    if out_file:
        vis.capture_screen_image(out_file, do_render=True)
    else:
        vis.run()  # this will open a window


def render_four_point_clouds(point_clouds, save_img=None, camera_loc=None):
    bbox1 = get_unit_bbox()
    bbox1.translate((1.2, 0, 0))
    pcd1 = PointCloud(point_clouds[0])
    pcd1.translate([1.2, 0, 0])
    if version.parse(o3d.__version__) > version.parse("0.7.0.0"):
        axes1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[1.2, 0, 0])

    bbox2 = get_unit_bbox()
    bbox2.translate((0, 0, 0))
    pcd2 = PointCloud(point_clouds[1])
    pcd2.translate([0, 0, 0])
    if version.parse(o3d.__version__) > version.parse("0.7.0.0"):
        axes2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    bbox3 = get_unit_bbox()
    bbox3.translate((1.2, 1.2, 0))
    pcd3 = PointCloud(point_clouds[2])
    pcd3.translate([1.2, 1.2, 0])
    if version.parse(o3d.__version__) > version.parse("0.7.0.0"):
        axes3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[1.2, 1.2, 0])

    bbox4 = get_unit_bbox()
    bbox4.translate((0, 1.2, 0))
    pcd4 = PointCloud(point_clouds[3])
    pcd4.translate([0, 1.2, 0])
    if version.parse(o3d.__version__) > version.parse("0.7.0.0"):
        axes4 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 1.2, 0])

    if version.parse(o3d.__version__) > version.parse("0.7.0.0"):
        render_geometries([pcd1, pcd2, pcd3, pcd4]+[axes1, axes2, axes3, axes4]+[bbox1, bbox2, bbox3, bbox4],
                          out_file=save_img, camera_json=camera_loc)
    else:
        render_geometries([pcd1, pcd2, pcd3, pcd4]+[bbox1, bbox2, bbox3, bbox4],
                          out_file=save_img, camera_json=camera_loc)


def render_four_meshes(meshes, save_img=None, camera_loc=None):
    meshes[0].translate([1.5, 0, 0])
    meshes[1].translate([0, 0, 0])
    meshes[2].translate([1.5, 1.5, 0])
    meshes[3].translate([0, 1.5, 0])
    mesh_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[-1, -1, -1])
    render_geometries(meshes+[mesh_axes], out_file=save_img, camera_json=camera_loc)


def get_unit_bbox():
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    if version.parse(o3d.__version__) <= version.parse("0.7.0.0"):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
    else:
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
    return line_set


if __name__ == '__main__':
    pcd = np.random.random((500, 3))
    pcd1 = np.random.random((500, 3))
    pcd2 = np.random.random((500, 3))
    pcd3 = np.random.random((500, 3))
    render_four_point_clouds([pcd, pcd1, pcd2, pcd3], "test_img.png")
    exit()
    pcd = PointCloud(pcd)

    # render_geometries([pcd], camera_json=SCREEN_CAMERA_POSITION_FILE)

    v = np.array([
        [1, -1, 1],
        [1, -1, -1],
        [1, 1, -1],
        [1, 1, 1],
        [-1, -1, 1],
        [-1, -1, -1],
        [-1, 1, -1],
        [-1, 1, 1]
    ])
    f = np.array([
        [4, 0, 3],
        [4, 3, 7],
        [0, 1, 2],
        [0, 2, 3],
        [1, 5, 6],
        [1, 6, 2],
        [5, 4, 7],
        [5, 7, 6],
        [7, 3, 2],
        [7, 2, 6],
        [0, 5, 1],
        [0, 4, 5]
    ])

    mesh = TriangleMesh(v, f)

    # render_geometries([mesh])
    # render_four_meshes([mesh, copy.copy(mesh), copy.copy(mesh), copy.copy(mesh)])

