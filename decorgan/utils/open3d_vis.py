import open3d
from utils.open3d_visualizer import VisOpen3D


def main():
    w = 1024
    h = 768

    # pcd = open3d.io.read_point_cloud("fragment.ply")
    obj_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/data/buildnet_component_refined/RELIGIOUSchurch_mesh2460/style_mesh_group2_28_28_window__unknown/model.obj"
    mesh = open3d.io.read_triangle_mesh(obj_file)

    # create window
    window_visible = True
    vis = VisOpen3D(width=w, height=h, visible=window_visible)

    # point cloud
    # vis.add_geometry(pcd)
    vis.add_geometry(mesh)

    # update view
    # vis.update_view_point(intrinsic, extrinsic)

    # save view point to file
    # vis.save_view_point("view_point.json")
    # vis.load_view_point("view_point.json")
    vis.load_view_point("ScreenCameraLocation.json")


    # capture images
    depth = vis.capture_depth_float_buffer(show=True)
    image = vis.capture_screen_float_buffer(show=True)

    # save to file
    vis.capture_screen_image("capture_screen_image.png")
    vis.capture_depth_image("capture_depth_image.png")


    # draw camera
    if window_visible:
        vis.load_view_point("ScreenCameraLocation.json")
        intrinsic = vis.get_view_point_intrinsics()
        extrinsic = vis.get_view_point_extrinsics()
        vis.draw_camera(intrinsic, extrinsic, scale=0.5, color=[0.8, 0.2, 0.8])
        # vis.update_view_point(intrinsic, extrinsic)

    if window_visible:
        vis.load_view_point("ScreenCameraLocation.json")
        vis.run()

    del vis


def render_obj(obj_file, w, h, window_visible, view_file, render_img, show=True):

    # create window
    vis = VisOpen3D(width=w, height=h, visible=window_visible)

    mesh = open3d.io.read_triangle_mesh(obj_file)
    vis.add_geometry(mesh)

    vis.load_view_point(view_file)
    image = vis.capture_screen_float_buffer(show=show)
    vis.capture_screen_image(render_img)

    # draw camera
    if window_visible:
        vis.load_view_point(view_file)
        intrinsic = vis.get_view_point_intrinsics()
        extrinsic = vis.get_view_point_extrinsics()
        vis.draw_camera(intrinsic, extrinsic, scale=0.5, color=[0.8, 0.2, 0.8])

    if window_visible:
        vis.load_view_point(view_file)
        vis.run()

    del vis


def render_mesh(mesh, w, h, window_visible, view_file, render_img, show=True):

    # create window
    vis = VisOpen3D(width=w, height=h, visible=window_visible)

    vis.add_geometry(mesh)

    vis.load_view_point(view_file)
    image = vis.capture_screen_float_buffer(show=show)
    vis.capture_screen_image(render_img)

    # draw camera
    if window_visible:
        vis.load_view_point(view_file)
        intrinsic = vis.get_view_point_intrinsics()
        extrinsic = vis.get_view_point_extrinsics()
        vis.draw_camera(intrinsic, extrinsic, scale=0.5, color=[0.8, 0.2, 0.8])

    if window_visible:
        vis.load_view_point(view_file)
        vis.run()

    del vis


def interactive_plot(meshes):
    viewer = open3d.visualization.VisualizerWithKeyCallback()
    viewer.create_window()
    for mesh in meshes:
        viewer.add_geometry(mesh)
    viewer.run()
    viewer.destroy_window()


if __name__ == "__main__":
    # main()
    w = 1024
    h = 768

    obj_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/data/unified_buildnet_components/RELIGIOUSchurch_mesh1574/style_mesh_group46_165_155_door__unknown/model.obj"
    window_visible = False
    view_file = "ScreenCameraUnitCubeLocation.json"
    render_img = "capture_screen_image.png"

    render_obj(obj_file, w, h, window_visible, view_file, render_img, show=True)

    interactive_plot([open3d.io.read_triangle_mesh(obj_file)])
