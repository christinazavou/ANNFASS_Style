import os

from unittest import TestCase
import open3d as o3d

from contentstylecomponent import ContentStyleComponentDataset

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def longest_common_prefix(strs):
    longest_pre = ""
    if not strs: return longest_pre
    shortest_str = min(strs, key=len)
    for i in range(len(shortest_str)):
        if all([x.startswith(shortest_str[:i+1]) for x in strs]):
            longest_pre = shortest_str[:i+1]
        else:
            break
    return longest_pre


class TestAnnfassComponentDataset(TestCase):
    def test_me(self):
        inp_path = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/"
        inp_path += "style_detection/logs/buildnet_content_style_splits"
        dset = ContentStyleComponentDataset(inp_path, content_pts=512, style_pts=4096)
        vis = 0
        i = 0
        while vis < 20:
            content_xyz, content_detailed_xyz, style_xyz, \
            (content_file, content_detailed_file, style_file) = dset.__getitem__(i)
            i += 1
            if content_xyz is None:
                continue
            vis += 1
            content_component = os.path.basename(content_file)
            style_component = os.path.basename(style_file)
            building = longest_common_prefix([content_component, style_component])
            content_component = content_component.replace(building, "")
            style_component = style_component.replace(building, "")
            content_pcd = o3d.geometry.PointCloud()
            content_detailed_pcd = o3d.geometry.PointCloud()
            style_pcd = o3d.geometry.PointCloud()
            content_pcd.points = o3d.utility.Vector3dVector(content_xyz)
            content_detailed_pcd.points = o3d.utility.Vector3dVector(content_detailed_xyz)
            style_pcd.points = o3d.utility.Vector3dVector(style_xyz)
            o3d.visualization.draw_geometries([content_pcd.translate([-0.5, 0, 0]),
                                               content_detailed_pcd,
                                               style_pcd.translate([0.5, 0, 0])],
                                              window_name=f"{building} {content_component} {style_component}")
