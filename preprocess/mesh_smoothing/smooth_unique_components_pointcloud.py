import argparse
import os
import sys

from fps import farthest_point_sampling
from laplacian import pointcloud_taubin_smoothing_with_laplacian

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.mesh_utils import write_ply, normalize_coords, read_ply
from common.utils import parse_buildings_csv
from preprocess.mesh_sampling.geometry_utils import ShapeFeatureExporter


def smooth_component(input_dir, model, component, out_dir, sfe):

    iterations, lamb, mu = 3, 0.4, 0.41
    tau_ext = f"{iterations}_{int(lamb*100)}"
    tau_file = f"{model}_{component}_sfe_tau{tau_ext}.ply"
    tau_file = os.path.join(out_dir, tau_file)

    ratio = 0.7
    # fps_file = f"{model}_{component}_sfe_fps.ply"
    fps_file = f"{model}_{component}_sfe_fps{int(ratio*100)}.ply"
    fps_file = os.path.join(out_dir, fps_file)

    sfe_file = f"{model}_{component}_sfe.ply"
    sfe_file = os.path.join(out_dir, sfe_file)

    if os.path.exists(fps_file) and os.path.exists(sfe_file) and os.path.exists(tau_file):
        return

    if os.path.exists(sfe_file):
        vertices_sfe, _, _ = read_ply(sfe_file)
    else:
        shape_feature_exporter_result = ShapeFeatureExporter(os.path.join(input_dir, model, component+".ply"),
                                                             "--do-not-rescale-shape "
                                                             "--export-point-samples "
                                                             "--num-point-samples 10000",
                                                             sfe)

        vertices_sfe = shape_feature_exporter_result[0][:1000]
        vertices_sfe = normalize_coords(vertices_sfe, "box")
        write_ply(sfe_file, vertices_sfe)

    if not os.path.exists(fps_file):
        try:
            vertices_sfe_fps = farthest_point_sampling(vertices_sfe)
            write_ply(fps_file, vertices_sfe_fps)
        except RuntimeError as e:
            print(e)

    if not os.path.exists(tau_file):
        try:
            vertices_sfe_tau = pointcloud_taubin_smoothing_with_laplacian(vertices_sfe, 3, 0.4, 0.41)
            write_ply(tau_file, vertices_sfe_tau)
        except RuntimeError as e:
            print(e)
            print(f"didnt generate {tau_file}")


if __name__ == '__main__':
    sfe = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/preprocess/mesh_sampling/shapefeatureexporter/build/ShapeFeatureExporter"

    parser = argparse.ArgumentParser()
    parser.add_argument('--unique_mesh', type=str, default="/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/uncom_mesh")
    parser.add_argument('--out_dir', type=str, default="/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/uncom_samples")
    parser.add_argument('--buildings_csv', type=str, default="/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/buildings.csv")
    # parser.add_argument('--unique_mesh', type=str, default="/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_april/uncom_mesh")
    # parser.add_argument('--out_dir', type=str, default="/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_april/uncom_samples")
    # parser.add_argument('--buildings_csv', type=str, default="/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_april/buildings.csv")
    # parser.add_argument('--sfe', type=str, required=True)
    args = parser.parse_args()

    buildings = parse_buildings_csv(args.buildings_csv)
    os.makedirs(args.out_dir, exist_ok=True)

    for building in buildings:
        if not os.path.exists(os.path.join(args.unique_mesh, building)):
            continue
        for f in os.listdir(os.path.join(args.unique_mesh, building)):
            if f.endswith(".ply"):
                name = f.replace(".ply", "")
                smooth_component(args.unique_mesh, building, name, args.out_dir, sfe)
