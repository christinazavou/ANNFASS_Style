import argparse
import json
import os
import sys

import numpy as np

path_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(path_dir)

from common.mesh_utils import write_ply, read_ply
from common.utils import UndirectedGraph, parse_buildings_csv, str2bool,\
    BUILDNET_STYLISTIC_ELEMENTS, ANNFASS_STYLISTIC_ELEMENTS, STYLES
from common.chamfer import get_min_chamfer_pytorch


# def get_component_clouds(obj):
#     component_clouds = dict()
#
#     for component_id in obj.components:
#         if "__" in component_id:
#             vertices, new_faces = get_component_mesh(obj, component_id)
#             mesh = TriangleMesh(vertices, np.array(new_faces))
#             mesh = get_normalized_mesh(mesh)
#             pcd = resample_mesh(mesh, 1000)
#
#             component_clouds[component_id] = pcd
#
#             # print(pcd.mean(), pcd.min(), pcd.max())
#             # coords, xyzs, ids = get_voxelized_data(None, pcd, 128, transform=None)
#             # visualize_result(xyzs, xyzs, 128)
#             # write_ply(f"/home/graphicslab/Desktop/annfass_playground/{component_id}.ply", xyzs,)
#     return component_clouds


# def get_component_clouds(obj: ObjMeshWithComponents, points: SampledPoints, groups: dict):
#     component_clouds = dict()
#     for point in points:
#         for group_key, group_components in groups.items():
#             group_name = ''.join(x for x in group_components[0] if x.isalpha() or x=="_")
#             group_name = f"group{group_key}_{group_name}"
#             component_clouds.setdefault(group_name, [])
#             if obj.faces[point.face_idx].component in group_components:
#                 component_clouds[group_name] += [point.coords]
#     final_clouds = dict()
#     for name, vertices in component_clouds.items():
#         assert len(vertices) > 0, f"no sampled points for grouped component {name}"
#         final_clouds[name] = np.array(vertices)
#     return final_clouds


def get_component_clouds(ply_building_dir):
    building_name = os.path.basename(ply_building_dir)
    final_clouds = dict()
    for root, dirs, files in os.walk(ply_building_dir):
        for file in files:
            if file.endswith(".ply"):
                component = file.replace(building_name+"_", "").replace(".ply", "")
                cloud_data = read_ply(os.path.join(root, file))
                final_clouds[component] = cloud_data[0]
    return final_clouds


class DuplicatesGrouper:

    def __init__(self, chamfer_factor=0.01):
        self.chamfer_factor = chamfer_factor

    def __call__(self, clouds, names=None, debug_dir=None):
        self.names = names
        self.clouds = clouds
        self.debug_dir = debug_dir
        if debug_dir is not None:
            assert names is not None and len(names) == len(clouds)
        return self.group_duplicates()

    # def _save_distances(self, distances):
    #     if self.debug_dir is not None:
    #         os.makedirs(self.debug_dir, exist_ok=True)
    #         with open(os.path.join(self.debug_dir, "distance.csv"), "w") as fout:
    #             fout.write("name,"+",".join(self.names)+"\n")
    #             for idx, name in enumerate(self.names):
    #                 fout.write(name+","+",".join([str(d) for d in distances[idx]])+"\n")
    #
    # def _save_thresholds(self, thresholds):
    #     if self.debug_dir is not None:
    #         os.makedirs(self.debug_dir, exist_ok=True)
    #         with open(os.path.join(self.debug_dir, f"threshold_{self.chamfer_factor}.csv"), "w") as fout:
    #             fout.write("name,"+",".join(self.names)+"\n")
    #             for idx, name in enumerate(self.names):
    #                 fout.write(name+","+",".join([str(t) for t in thresholds[idx]])+"\n")

    def group_duplicates(self):
        num_clouds = len(self.clouds)
        ch_distances = np.zeros((num_clouds, num_clouds))
        ch_thresholds = np.zeros((num_clouds, num_clouds))

        graph = UndirectedGraph(num_clouds)

        for idx1 in range(0, num_clouds - 1):
            for idx2 in range(idx1 + 1, num_clouds):

                name1 = self.names[idx1]
                name2 = self.names[idx2]

                words1 = name1.lower().split("__")[0].split("_")
                words2 = name2.lower().split("__")[0].split("_")
                if len(set(words1) & set(words2)) == 0:
                    ch_thresholds[idx1, idx2] = np.nan
                    ch_distances[idx1, idx2] = np.nan
                    continue  # most probably not same components thus won't compare

                xyz1_min, xyz1_max = self.clouds[idx1].min(axis=0), self.clouds[idx1].max(axis=0)
                xyz2_min, xyz2_max = self.clouds[idx2].min(axis=0), self.clouds[idx2].max(axis=0)
                diagonal1 = np.linalg.norm(xyz1_min - xyz1_max)
                diagonal2 = np.linalg.norm(xyz2_min - xyz2_max)
                chamfer_threshold = self.chamfer_factor * min(diagonal1, diagonal2)
                ch_thresholds[idx1, idx2] = chamfer_threshold

                fp = os.path.join(self.debug_dir, f"{name1}_{name2}.ply") if self.debug_dir is not None else None
                ch_distance = get_min_chamfer_pytorch(self.clouds[idx1], self.clouds[idx2], angle=30, filepath=fp, stop_at=chamfer_threshold)
                ch_distances[idx1, idx2] = ch_distance

                if ch_distance <= chamfer_threshold:
                    graph.add_edge(idx1, idx2)

        # self._save_thresholds(ch_thresholds)
        # self._save_distances(ch_distances)

        duplicate_groups_indices = graph.connected_components()
        return [[self.names[i] for i in group] for group in duplicate_groups_indices]


def process_building(ply_building_dir, out_dir, debug_dir=None):
    if os.path.exists(out_dir) and os.listdir(out_dir) != []:
        return
    if not os.path.exists(ply_building_dir):
        return

    # obj = ObjMeshWithComponents(obj_file)
    # points = SampledPoints()(points_file, faces_file, obj.faces)
    # with open(groups_file, "rb") as fin:
    #     groups = json.load(fin)

    # clouds = get_component_clouds(obj)
    # clouds = get_component_clouds(obj, points, groups)

    clouds = get_component_clouds(ply_building_dir)

    clouds_per_style_and_structure = {}
    for component_name, component_cloud in clouds.items():

        style = [s for s in STYLES if s.lower() in component_name.lower()]
        if len(style) > 1:
            print(f"Component {component_name} belongs in more than one style?")
        assert len(style) > 0, f"Component {component_name} has none style?"
        style = style[0]

        structure = [s for s in STYLISTIC_ELEMENTS if s.lower() in component_name.lower()]
        if len(structure) == 0:
            print(f"Component {component_name} has none STYLISTIC structure..")
            continue
        if len(structure) > 1:
            print(f"Component {component_name} belongs in more than one structure?")
        structure = structure[0]

        clouds_per_style_and_structure.setdefault((style, structure), {})
        clouds_per_style_and_structure[(style, structure)][component_name] = component_cloud

    dg = DuplicatesGrouper()

    duplicates = []
    for (style, structure), style_structure_clouds in clouds_per_style_and_structure.items():
        duplicates += dg(list(style_structure_clouds.values()), list(style_structure_clouds.keys()), debug_dir)

    os.makedirs(out_dir, exist_ok=True)
    duplicates_file = os.path.join(out_dir, "duplicates.json")
    duplicates_map = {}
    for i, group in enumerate(duplicates):
        duplicates_map[group[0]] = group
        if debug_dir is not None:
            o_dir = os.path.join(debug_dir, f"group{i}")
            os.makedirs(o_dir, exist_ok=True)
            for name in group:
                o_file = os.path.join(o_dir, f"{name}.ply")
                write_ply(o_file, clouds[name])
        o_file = os.path.join(out_dir, f"{group[0]}.ply")
        write_ply(o_file, clouds[group[0]])
    with open(duplicates_file, 'w') as fout_json:
        json.dump(duplicates_map, fout_json, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--buildings_csv', type=str, required=True)
    parser.add_argument('--ply_per_component_dir', type=str, default="samplePoints/ply_nocut_pgc")
    parser.add_argument('--out_dir', type=str, default="unique_components_after_group_and_sfe")
    parser.add_argument('--debug', type=str2bool, default=False)
    args = parser.parse_args()

    if "buildnet" in args.root_dir.lower():
        print("Using buildnet structures")
        STYLISTIC_ELEMENTS = BUILDNET_STYLISTIC_ELEMENTS
    else:
        print("Using annfass structures")
        STYLISTIC_ELEMENTS = ANNFASS_STYLISTIC_ELEMENTS

    buildings = parse_buildings_csv(args.buildings_csv)

    for bi, building in enumerate(buildings):
        print(f"Processing {building}")
        if args.debug:
            process_building(os.path.join(args.root_dir, args.ply_per_component_dir, building),
                             os.path.join(args.root_dir, args.out_dir, building),
                             os.path.join(args.root_dir, f"{args.out_dir}_debug", building))
        else:
            process_building(os.path.join(args.root_dir, args.ply_per_component_dir, building),
                             os.path.join(args.root_dir, args.out_dir, building))
