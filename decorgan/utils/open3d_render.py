import open3d as o3d
from typing import List, Optional
import cv2


def convert_to_open3d_param(intrinsic, extrinsic):
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
    param.intrinsic.intrinsic_matrix = intrinsic
    param.extrinsic = extrinsic
    return param


def load_and_update_view_point(vis, filename):
    param = o3d.io.read_pinhole_camera_parameters(filename)
    intrinsic = param.intrinsic.intrinsic_matrix
    extrinsic = param.extrinsic

    ctr = vis.get_view_control()
    param = convert_to_open3d_param(intrinsic, extrinsic)
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    # ctr.convert_from_pinhole_camera_parameters(param)
    vis.update_renderer()


def render_geometries(geometries: List[o3d.geometry.Geometry],
                      out_file: Optional[str] = None,
                      camera_json: Optional[str] = None,
                      out_img: bool = False,
                      window_name: str = 'Open3D'):

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, window_name=window_name)
    vis.poll_events()
    vis.update_renderer()

    for geom in geometries:
        vis.add_geometry(geom)

    if camera_json:
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(camera_json)
        ctr.convert_from_pinhole_camera_parameters(param)
        # load_and_update_view_point(vis, camera_json)

    if out_file or out_img:
        # vis.capture_screen_image(out_file, do_render=True)
        buf = vis.capture_screen_float_buffer(do_render=True)
        if out_file:
            cv2.imwrite(out_file, buf)
        return buf
    else:
        vis.run()  # this will open a window


def save_view_point(geom, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(geom)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(geom, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(geom)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':

    import mcubes
    import numpy as np
    from utils import normalize_vertices, CameraJsonPosition
    from utils.open3d_utils import TriangleMesh
    import binvox_rw

    voxel_model_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/data/03001627/35d60ae4cb2e81979d9fad88e2f4c8ff/model_depth_fusion.binvox"
    voxel_model_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components/COMMERCIALhotel_building_mesh0295/style_mesh_group98_1005_935_door__unknown/model_filled.binvox"
    with open(voxel_model_file, "rb") as fin:
        voxel_model_512 = binvox_rw.read_as_3d_array(fin, fix_coords=True).data.astype(np.uint8)
    v, f = mcubes.marching_cubes(voxel_model_512, 0)
    v = normalize_vertices(v)
    # m = TriangleMesh(v, f)
    v1 = v-np.array([0.7, 0, 0])
    v2 = v+np.array([0.3, 0, 0])
    m1 = TriangleMesh(v1, f)
    m2 = TriangleMesh(v2, f)

    save_view_point(m1, "viewpoint.json")
    load_view_point(m1, "viewpoint.json")
    exit()

    # rg = render_geometries([m], out_img=True)
    # render_geometries([m], camera_json=CameraJsonPosition)
    # render_geometries([m])
    render_geometries([m1, m2], camera_json=CameraJsonPosition)
