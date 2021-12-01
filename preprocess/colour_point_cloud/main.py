import argparse
import traceback

from utils import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from common.utils import parse_buildings_csv, str2bool
from common.mesh_utils import ObjMeshWithComponentsAndMaterials, SampledPoints


def process_building_wrapper(building, obj_file, points_file, faces_file, output_dir, debug=False, uv_bounds=None):

    if not os.path.exists(obj_file):
        print(obj_file)
        return
    if not os.path.exists(points_file):
        print(points_file)
        return
    if not os.path.exists(faces_file):
        print(faces_file)
        return

    out_file = os.path.join(output_dir, "{}.ply".format(building))
    if os.path.exists(out_file):
        return

    obj = ObjMeshWithComponentsAndMaterials(obj_file)
    print(obj)

    sampled_points = SampledPoints()(points_file, faces_file, obj.faces)

    debug_textures_dir = None
    if debug:
        debug_textures_dir = os.path.join(output_dir, "debugTextures", building)
        os.makedirs(debug_textures_dir, exist_ok=True)

    sampled_points_with_colour, rescaled_textures = process_building(obj, sampled_points, uv_bounds)

    write_ply_with_colour(sampled_points_with_colour, out_file)

    if debug_textures_dir is not None:
        debug_building_textures(obj, sampled_points_with_colour, rescaled_textures, debug_textures_dir)


if __name__ == '__main__':

    # todo: maybe run utils per component & will be faster?

    parser = argparse.ArgumentParser()
    parser.add_argument("--objects_dir", required=True, type=str)
    parser.add_argument("--points_dir", required=True, type=str)
    parser.add_argument("--faces_dir", required=True, type=str)
    parser.add_argument("--buildings_csv", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--uv_bounds", default=None, type=str, choices=["None"])#, "material", "material_component"])
    parser.add_argument("--debug", default=False, type=str2bool)
    args = parser.parse_args()

    print("args:{}\n".format(args))

    os.makedirs(args.output_dir, exist_ok=True)
    buildings = parse_buildings_csv(args.buildings_csv)

    for building_name in buildings:
        print("Processing building {}".format(building_name))
        if "ply" in args.points_dir:
            extension = "ply"
        else:
            extension = "pts"

        try:
            process_building_wrapper(building_name,
                                     f"{args.objects_dir}/{building_name}/{building_name}.obj",
                                     f"{args.points_dir}/{building_name}.{extension}",
                                     f"{args.faces_dir}/{building_name}.txt",
                                     args.output_dir,
                                     debug=args.debug,
                                     uv_bounds=args.uv_bounds)
        except Exception as e:
            print(f"!!!!!!! EXCEPTION FOR BULDING {building_name} !!!!!!!")
            print(e)
            traceback.print_exc()
