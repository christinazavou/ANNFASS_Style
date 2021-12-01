import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from common.mesh_utils import FullSampledPoints, ObjMeshComponentsReference, write_full_ply
from common.utils import STYLES, str2bool
from common.multiprocessing_utils import log_process_time


def write_stats_csv(pts_cnt: dict, csv_file: str):
    if pts_cnt != {}:
        values = np.array(list(pts_cnt.values()))
        pts_cnt['minimum_cnt'] = np.min(values)
        pts_cnt['maximum_cnt'] = np.max(values)
        pts_cnt['avg_cnt'] = np.mean(values)
        pts_cnt['total'] = len(values)
        pd.DataFrame.from_dict(pts_cnt, orient='index').to_csv(csv_file)


def pts2ply_w_label(obj_file, sampled_pts_file, sampled_face_file, out_dir, rnv_file=None, color_file=None, cut_at=-1):

    faces, obj, pts, rnvs, colors = get_data(cut_at, obj_file, sampled_face_file, sampled_pts_file, rnv_file, color_file)

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "{}.ply".format(os.path.basename(obj_file)[:-4]))
    lbl_stats_file = out_file.replace(".ply", "_stats_lbl.csv")
    cmp_stats_file = out_file.replace(".ply", "_stats_cmp.csv")
    components_file = out_file.replace(".ply", "_components.json")
    lbl_cnt = {}
    pts_component_cnt = {}  # only style components
    labels = []
    components = []

    for p_idx in range(len(pts)):
        face_idx = faces[p_idx]
        component_id = obj.faces[face_idx].component
        label = [s for s in STYLES if s.lower() in component_id.lower()]
        if len(label) == 0:
            lbl = len(STYLES)  # todo or -1 ?
            cmp = -1
        else:
            lbl = STYLES.index(label[0])
            cmp = obj.components.index(component_id)
            pts_component_cnt.setdefault(component_id, 0)
            pts_component_cnt[component_id] += 1
        labels.append(lbl)
        components.append(cmp)
        lbl_cnt.setdefault(lbl, 0)
        lbl_cnt[lbl] += 1
    labels = np.array(labels).reshape(len(pts), 1)
    components = np.array(components).reshape((len(pts), 1))

    if len(rnvs) > 0 and len(colors) > 0:
        rnvs = rnvs.reshape((len(pts), 1))
        colors = colors.reshape((len(pts), 3))
        all_labels = np.concatenate([labels, components, rnvs, colors], 1)
        write_full_ply(out_file, pts, all_labels, label_names=['label', 'component', 'rnv', 'red', 'green', 'blue'])
    elif len(rnvs) > 0:
        rnvs = rnvs.reshape((len(pts), 1))
        all_labels = np.concatenate([labels, components, rnvs], 1)
        write_full_ply(out_file, pts, all_labels, label_names=['label', 'component', 'rnv'])
    elif len(colors) > 0:
        colors = colors.reshape((len(pts), 3))
        all_labels = np.concatenate([labels, components, colors], 1)
        write_full_ply(out_file, pts, all_labels, label_names=['label', 'component', 'red', 'green', 'blue'])
    else:
        all_labels = np.concatenate([labels, components], 1)
        write_full_ply(out_file, pts, all_labels, label_names=['label', 'component'])

    write_stats_csv(pts_component_cnt, cmp_stats_file)
    pd.DataFrame.from_dict(lbl_cnt, orient='index').to_csv(lbl_stats_file)
    with open(components_file, "w") as fout:
        json.dump(obj.components, fout, indent=2)


def pts2ply_w_label_w_group(obj_file, sampled_pts_file, sampled_face_file, group_file, out_dir, rnv_file=None, color_file=None, cut_at=-1):

    faces, obj, pts, rnvs, colors = get_data(cut_at, obj_file, sampled_face_file, sampled_pts_file, rnv_file, color_file)

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "{}.ply".format(os.path.basename(obj_file)[:-4]))
    lbl_stats_file = out_file.replace(".ply", "_stats_lbl.csv")
    cmp_stats_file = out_file.replace(".ply", "_stats_cmp.csv")
    components_file = out_file.replace(".ply", "_components.json")
    lbl_cnt = {}
    pts_component_cnt = {}  # only style components
    labels = []
    components = []

    with open(group_file, "r") as fin:
        groups = json.load(fin)

    grouped_components_names = []
    for group_idx, group in enumerate(groups):
        group_meshes_ids = groups[group]
        component_id = f"group{group_idx}_{group_meshes_ids[0]}"
        grouped_components_names.append(component_id)

    for p_idx in range(len(pts)):
        face_idx = faces[p_idx]
        component_id = obj.faces[face_idx].component
        for group_idx, group in enumerate(groups):
            group_meshes_ids = groups[group]
            if component_id in group_meshes_ids:
                component_id = f"group{group_idx}_{group_meshes_ids[0]}"
        label = [s for s in STYLES if s.lower() in component_id.lower()]
        if len(label) == 0:
            lbl = len(STYLES)  # todo or -1 ?
            cmp = -1
        else:
            lbl = STYLES.index(label[0])
            cmp = grouped_components_names.index(component_id)
            pts_component_cnt.setdefault(component_id, 0)
            pts_component_cnt[component_id] += 1
        labels.append(lbl)
        components.append(cmp)
        lbl_cnt.setdefault(lbl, 0)
        lbl_cnt[lbl] += 1
    labels = np.array(labels).reshape(len(pts), 1)
    components = np.array(components).reshape((len(pts), 1))

    if len(rnvs) > 0 and len(colors) > 0:
        rnvs = rnvs.reshape((len(pts), 1))
        colors = colors.reshape((len(pts), 3))
        all_labels = np.concatenate([labels, components, rnvs, colors], 1)
        write_full_ply(out_file, pts, all_labels, label_names=['label', 'component', 'rnv', 'red', 'green', 'blue'])
    elif len(rnvs) > 0:
        rnvs = rnvs.reshape((len(pts), 1))
        all_labels = np.concatenate([labels, components, rnvs], 1)
        write_full_ply(out_file, pts, all_labels, label_names=['label', 'component', 'rnv'])
    elif len(colors) > 0:
        colors = colors.reshape((len(pts), 3))
        all_labels = np.concatenate([labels, components, colors], 1)
        write_full_ply(out_file, pts, all_labels, label_names=['label', 'component', 'red', 'green', 'blue'])
    else:
        all_labels = np.concatenate([labels, components], 1)
        write_full_ply(out_file, pts, all_labels, label_names=['label', 'component'])

    write_stats_csv(pts_component_cnt, cmp_stats_file)
    pd.DataFrame.from_dict(lbl_cnt, orient='index').to_csv(lbl_stats_file)
    with open(components_file, "w") as fout:
        json.dump(grouped_components_names, fout, indent=2)


def pts2ply_w_label_per_stylistic_component(obj_file, group_file, sampled_pts_file, sampled_face_file, out_dir,
                                            rnv_file=None, color_file=None, cut_at=-1):
    if os.path.exists(out_dir) and os.listdir(out_dir):
        return  # don't override

    # cut_at here refers to per component..
    faces, obj, pts, rnvs, colors = get_data(-1, obj_file, sampled_face_file, sampled_pts_file, rnv_file, color_file)
    has_rnvs = True if len(rnvs) > 0 else False
    has_colors = True if len(colors) > 0 else False

    with open(group_file, "r") as fin:
        groups = json.load(fin)

    stats_file = out_dir + "_stats.txt"
    cnt = {}
    for group_idx, group in enumerate(groups):
        group_meshes_ids = groups[group]
        component_name = f"group{group_idx}_{group_meshes_ids[0]}"

        label = [s for s in STYLES if s.lower() in component_name.lower()]
        if len(label) == 0:
            continue
        lbl = STYLES.index(label[0])
        out_file = os.path.join(out_dir, "{}_{}.ply".format(os.path.basename(obj_file)[:-4], component_name))
        component_pts = [p for p, i in zip(pts, faces) if obj.faces[i].component in group_meshes_ids]
        component_rnvs = [r for r, i in zip(rnvs, faces) if obj.faces[i].component in group_meshes_ids]
        component_colors = [c for c, i in zip(colors, faces) if obj.faces[i].component in group_meshes_ids]
        if len(component_pts) == 0:
            cnt[component_name] = 0
            continue
        cnt[component_name] = len(component_pts)

        if cut_at != -1 and cut_at < len(component_pts):
            component_pts = component_pts[:cut_at]  # they were randomly generated so this sample is still random,
            # while face idx is still known
            component_rnvs = component_rnvs[:cut_at]
            component_colors = component_colors[:cut_at]

        cp_features = np.vstack(component_pts)
        cp_labels = np.ones((len(component_pts), 1)).astype(int) * lbl

        if has_rnvs and has_colors:
            component_rnvs = np.vstack(component_rnvs)
            component_colors = np.vstack(component_colors)
            all_labels = np.concatenate([cp_labels, component_rnvs, component_colors], 1)
            write_full_ply(out_file, cp_features, all_labels, label_names=['label', 'rnv', 'red', 'green', 'blue'])
        elif has_rnvs:
            component_rnvs = np.vstack(component_rnvs)
            all_labels = np.concatenate([cp_labels, component_rnvs], 1)
            write_full_ply(out_file, cp_features, all_labels, label_names=['label', 'rnv'])
        elif has_colors:
            component_colors = np.vstack(component_colors)
            all_labels = np.concatenate([cp_labels, component_colors], 1)
            write_full_ply(out_file, cp_features, all_labels, label_names=['label', 'red', 'green', 'blue'])
        else:
            write_full_ply(out_file, cp_features, cp_labels, label_names=['label'])

    write_stats_csv(cnt, stats_file)


def pts2ply_w_label_per_stylistic_component_without_groups(obj_file, sampled_pts_file, sampled_face_file, out_dir,
                                                           rnv_file=None, color_file=None, cut_at=-1):
    if os.path.exists(out_dir) and os.listdir(out_dir):
        return  # don't override

    # cut_at here refers to per component..
    faces, obj, pts, rnvs, colors = get_data(-1, obj_file, sampled_face_file, sampled_pts_file, rnv_file, color_file)
    has_rnvs = True if len(rnvs) > 0 else False
    has_colors = True if len(colors) > 0 else False

    stats_file = out_dir + "_stats.txt"
    cnt = {}
    for component_id in obj.components:
        label = [s for s in STYLES if s.lower() in component_id.lower()]
        if len(label) == 0:
            continue
        lbl = STYLES.index(label[0])
        out_file = os.path.join(out_dir, "{}_{}.ply".format(os.path.basename(obj_file)[:-4], component_id))
        component_pts = [p for p, i in zip(pts, faces) if obj.faces[i].component == component_id]
        component_rnvs = [r for r, i in zip(rnvs, faces) if obj.faces[i].component == component_id]
        component_colors = [c for c, i in zip(colors, faces) if obj.faces[i].component == component_id]
        if len(component_pts) == 0:
            cnt[component_id] = 0
            continue
        cnt[component_id] = len(component_pts)

        if cut_at != -1 and cut_at < len(component_pts):
            component_pts = component_pts[:cut_at]  # they were randomly generated so this sample is still random,
            # while face idx is still known
            component_rnvs = component_rnvs[:cut_at]
            component_colors = component_colors[:cut_at]

        cp_features = np.vstack(component_pts)
        cp_labels = np.ones((len(component_pts), 1)).astype(int) * lbl

        if has_rnvs and has_colors:
            component_rnvs = np.vstack(component_rnvs)
            component_colors = np.vstack(component_colors)
            all_labels = np.concatenate([cp_labels, component_rnvs, component_colors], 1)
            write_full_ply(out_file, cp_features, all_labels, label_names=['label', 'rnv', 'red', 'green', 'blue'])
        elif has_rnvs:
            component_rnvs = np.vstack(component_rnvs)
            all_labels = np.concatenate([cp_labels, component_rnvs], 1)
            write_full_ply(out_file, cp_features, all_labels, label_names=['label', 'rnv'])
        elif has_colors:
            component_colors = np.vstack(component_colors)
            all_labels = np.concatenate([cp_labels, component_colors], 1)
            write_full_ply(out_file, cp_features, all_labels, label_names=['label', 'red', 'green', 'blue'])
        else:
            write_full_ply(out_file, cp_features, cp_labels, label_names=['label'])

    write_stats_csv(cnt, stats_file)


def get_data(cut_at, obj_file, sampled_face_file, sampled_pts_file, rnv_file=None, color_file=None):
    obj = ObjMeshComponentsReference(obj_file)
    print(obj)
    pts, faces, rnvs, colors = FullSampledPoints()(sampled_pts_file, sampled_face_file,
                                                   ridge_valley_file=rnv_file,
                                                   color_file=color_file,
                                                   faces=obj.faces)
    if cut_at != -1:
        if cut_at < len(pts):
            pts = pts[:cut_at]
            faces = faces[:cut_at]
            rnvs = rnvs[:cut_at]
            colors = colors[:cut_at]
        else:
            print("Warning: cut_at ({}) > samples ({})".format(cut_at, len(pts)))
    return faces, obj, pts, rnvs, colors


def process_buildings(data, process_id, **kwargs):

    cut_at = kwargs['cut_at'] if 'cut_at' in kwargs else -1
    per_component = kwargs['per_component'] if 'per_component' in kwargs else False

    t_start_proc = time.time()
    print(f"Starting pts2ply process {process_id} for {len(data)} case(s)...")

    for building, group_file, pts_file, face_file, obj_file, rnv_file, color_file, out_dir in data:

        if all([os.path.exists(pts_file),
                os.path.exists(face_file),
                (group_file is None or os.path.exists(group_file))]):
            print("Processing {}".format(building))
            if per_component:
                if group_file is not None:
                    pts2ply_w_label_per_stylistic_component(obj_file, group_file, pts_file, face_file,
                                                            os.path.join(out_dir, building),
                                                            rnv_file, color_file, cut_at)
                else:
                    pts2ply_w_label_per_stylistic_component_without_groups(obj_file, pts_file, face_file,
                                                                           os.path.join(out_dir, building),
                                                                           rnv_file, color_file, cut_at)
            else:
                if group_file is not None:
                    pts2ply_w_label_w_group(obj_file, pts_file, face_file, group_file, out_dir, rnv_file, color_file, cut_at)
                else:
                    pts2ply_w_label(obj_file, pts_file, face_file, out_dir, rnv_file, color_file, cut_at)

    log_process_time(process_id, t_start_proc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", default="normalizedObj", type=str)
    parser.add_argument("--pts_dir", default="samplePoints/point_cloud_10K", type=str)
    parser.add_argument("--ply_dir_prefix", default="samplePoints/ply", type=str)
    parser.add_argument("--groups_dir", default=None, type=str)
    parser.add_argument("--building", default=None, type=str)
    parser.add_argument("--cut_at", default=-1, type=int)
    parser.add_argument("--per_component", default=False, type=str2bool)
    parser.add_argument("--rnv", default="ridge_or_valley", type=str)
    parser.add_argument("--color", default="colorPly", type=str)
    args = parser.parse_args()

    ply_dir_suffix = ""
    if args.cut_at == -1:
        ply_dir_suffix += "_nocut"
    else:
        ply_dir_suffix += f"_cut{round(args.cut_at/1000, 1)}K"
    if args.per_component:
        if args.groups_dir is None:
            ply_dir_suffix += "_pc"
        else:
            ply_dir_suffix += "_pgc"
    else:
        if args.group_dir is not None:
            ply_dir_suffix += "_wg"

    pl_dir = os.path.join(args.ply_dir_prefix+ply_dir_suffix)
    print(f"Generating output ply files in {pl_dir}")

    rnvd = os.path.join(args.pts_dir.replace("point_cloud", args.rnv))
    colord = os.path.join(args.pts_dir.replace("point_cloud", args.color))

    pt_file = os.path.join(args.pts_dir, "{}.pts".format(args.building))
    f_file = os.path.join(args.pts_dir.replace("point_cloud", "faces"), "{}.txt".format(args.building))

    g_file = None
    if args.groups_dir != None:
        g_file = os.path.join(args.groups_dir, args.building, "groups.json")

    o_file = os.path.join(args.obj_dir, args.building, "{}.obj".format(args.building))
    if "style_mesh" in pt_file:
        o_file = os.path.join(args.obj_dir, args.building, "{}_style_mesh.obj".format(args.building))

    r_file = None
    if os.path.exists(rnvd):
        r_file = os.path.join(rnvd, "{}.txt".format(args.building))

    c_file = None
    if os.path.exists(colord):
        c_file = os.path.join(colord, "{}.ply".format(args.building))

    process_buildings([(args.building, g_file, pt_file, f_file, o_file, r_file, c_file, pl_dir)],
                      0,
                      cut_at=args.cut_at,
                      per_component=args.per_component)
