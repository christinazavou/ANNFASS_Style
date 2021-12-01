import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from common.utils import str2bool, parse_buildings_csv, STYLES
from common.multiprocessing_utils import run_function_in_parallel
from common.mesh_utils import export_selection_obj
from create_point_cloud import create_point_clouds


# create_point_clouds([['Agiasophia Fixed', "/home/graphicslab/Downloads/Agiasofia Fixed/Agiasophia Fixed.obj"]], 0,
#                     n_samples=30000, debug=True, export_curvatures=False, export_pca=False,
#                     sfe='/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/preprocess/mesh_sampling/shapefeatureexporter/build/ShapeFeatureExporter',
#                     override=True, remove=False, points_dir='/home/graphicslab/Downloads/Agiasofia Points',
#                     faces_dir='/home/graphicslab/Downloads/Agiasofia Faces')
# exit()

parser = argparse.ArgumentParser()
parser.add_argument("--num_processes", type=int, default=8, help="Number of processes to use [default: 1]")
parser.add_argument("--num_samples", type=int, default=10000,
                    help="Number of initial samples on mesh edges and faces [default: 500,000]")
parser.add_argument("--buildings_csv", required=True, help="Buildings to process", type=str)
parser.add_argument("--debug", default=False, type=str2bool)
parser.add_argument("--sfe", type=str, required=True, help="executable file")
parser.add_argument("--obj_dir", type=str, required=True, help="Data with normalized obj files directory")
parser.add_argument("--export_curvatures", type=str2bool, default=False, help="Whether to export curvatures")
parser.add_argument("--export_pca", type=str2bool, default=False, help="Whether to export pca")
parser.add_argument("--override", type=str2bool, default=True, help="Whether to override tmp file")
parser.add_argument("--remove", type=str2bool, default=True, help="Whether to remove tmp file")
parser.add_argument("--stylistic_selection", type=str2bool, default=False, help="Whether to run on stylistic mesh")
parser.add_argument("--out_samples_dir", type=str, required=True)
ARGS = parser.parse_args()
print("ARGS: {}".format(ARGS))

n_points = ARGS.num_samples / 1000.0
N_POINTS = "{n_points:d}K".format(n_points=int(n_points)) if n_points >= 1 else "{n_points:.3f}K".format(
    n_points=n_points)
OUTPUT_DIR = ARGS.out_samples_dir

extension = ""
if ARGS.stylistic_selection:
    extension += "_style_mesh"
if ARGS.export_curvatures:
    if ARGS.export_pca:
        extension += "_wc_wpca"
    else:
        extension += "_wc"
else:
    if ARGS.export_pca:
        extension += "_wpca"

pts_dir = os.path.join(OUTPUT_DIR, "point_cloud_{}{}".format(N_POINTS, extension))
face_dir = os.path.join(OUTPUT_DIR, "faces_{}{}".format(N_POINTS, extension))

os.makedirs(pts_dir, exist_ok=True)
os.makedirs(face_dir, exist_ok=True)

print("Starting point cloud creation of {} points.".format(N_POINTS))
buildings = parse_buildings_csv(ARGS.buildings_csv)

model_list = []
for building in buildings:

    pts_file = os.path.join(pts_dir, "{}.pts".format(building))
    faces_file = os.path.join(face_dir, "{}.txt".format(building))

    if os.path.exists(pts_file) and os.path.exists(faces_file):
        continue  # dont override

    orig_obj_file = os.path.join(ARGS.obj_dir, building, "{}.obj".format(building))
    style_select_obj_file = os.path.join(ARGS.obj_dir, building, "{}_style_mesh.obj".format(building))

    if os.path.exists(orig_obj_file):
        if not ARGS.stylistic_selection:
            model_list.append([building, orig_obj_file])
        else:
            # NOTE THIS WONT OVERRIDE
            if os.path.exists(style_select_obj_file):
                model_list.append([building, style_select_obj_file])
            else:
                export_selection_obj(orig_obj_file, style_select_obj_file, STYLES)
                model_list.append([building, style_select_obj_file])

print("models to process: {}\n".format(model_list))

# Preprocess models
t1 = time.time()
run_function_in_parallel(create_point_clouds, ARGS.num_processes, model_list,
                         n_samples=ARGS.num_samples, debug=ARGS.debug, export_curvatures=ARGS.export_curvatures,
                         export_pca=ARGS.export_pca, sfe=ARGS.sfe, override=ARGS.override, remove=ARGS.remove,
                         points_dir=pts_dir, faces_dir=face_dir)
total_time = time.time() - t1
print("Finished all processes. Time passed: {hours:d}:{minutes:d}:{seconds:d}"
      .format(hours=int((total_time / 60 ** 2) % (60 ** 2)),
              minutes=int((total_time / 60) % 60),
              seconds=int(total_time % 60)))

# for debugging, comment out the above and use this:
# create_point_clouds(model_list[:2], 0,
#                     n_samples=ARGS.num_samples, debug=ARGS.debug, export_curvatures=ARGS.export_curvatures,
#                     export_pca=ARGS.export_pca, sfe=ARGS.sfe, override=ARGS.override, remove=ARGS.remove,
#                     points_dir=pts_dir, faces_dir=face_dir)
