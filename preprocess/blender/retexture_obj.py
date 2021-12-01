import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from preprocess.blender.mesh_utils import bpy, unwrap_and_deselect, select_related_faces_and_update_bounds, \
    material_has_texture_image, add_shader_nodes_for_rescaling
from preprocess.blender.scene_utils import cleanup
from preprocess.blender.io_utils import load_obj
from common.utils import parse_buildings_csv


def unwrap_and_rescale_material(material):
    material_bounds = {'min_u': 1e9, 'min_v': 1e9, 'max_u': 1e-9, 'max_v': 1e-9}
    selected_faces_cnt, selected_objs_cnt = select_related_faces_and_update_bounds(material_bounds, material.name)
    if selected_faces_cnt == 0:
        print(f"Zero selected faces for material {material.name}")
        return
    if material_bounds['min_u'] < 0 or material_bounds['max_u'] > 1 \
            or material_bounds['min_v'] < 0 or material_bounds['max_v'] > 1:
        scale_x = round(material_bounds['max_u'] - material_bounds['min_u'], 2)
        scale_y = round(material_bounds['max_v'] - material_bounds['min_v'], 2)
        if scale_x == 0 or scale_y == 0:
            raise Exception(f"Zero scale appears for  material {material.name}")
        print(f"Will unwrap {selected_faces_cnt} faces from {selected_objs_cnt} components "
              f"with material {material.name} and will rescale with {scale_x}, {scale_y}, 1")
        unwrap_and_deselect()
        add_shader_nodes_for_rescaling(material, scale_x, scale_y)


def necessary_unwrap_of_mesh_and_rescales_of_materials():
    avg_time = 0
    materials_re_textured = 0
    for material in bpy.data.materials:
        if material_has_texture_image(material):
            s_time = time.time()
            unwrap_and_rescale_material(material)
            materials_re_textured += 1
            avg_time += (time.time() - s_time)
    print("Avg time per material: {}".format(avg_time//materials_re_textured))


# def process_building(obj_dir, out_dir, building):
def process_building(obj_dir, building):
    if os.path.exists(f"{obj_dir}_refinedTextures/{building}/{building}.obj"):
        print(f"Skipping {obj_dir}_refinedTextures/{building}/{building}.obj since it already exists.")
        return
    if not os.path.exists(f"{obj_dir}/{building}/{building}.obj"):
        print(f"Skipping {obj_dir}/{building}/{building}.obj since it doesnt exist.")
        return
    cleanup(materials=True)
    load_obj("{}/{}/{}.obj".format(obj_dir, building, building))
    try:
        necessary_unwrap_of_mesh_and_rescales_of_materials()
    except Exception as e:
        print(e)
        return
    os.makedirs(f"{obj_dir}_refinedTextures/{building}", exist_ok=True)
    bpy.ops.export_scene.obj(filepath=f"{obj_dir}_refinedTextures/{building}/{building}.obj",
                             axis_up='Y',
                             axis_forward='-Z',
                             use_selection=False)


if __name__ == '__main__':

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('-obj_dir_in', type=str, required=True)
        parser.add_argument('-buildings_csv', type=str, required=True)
        args = parser.parse_known_args(argv)[0]
    else:
        raise Exception('please give args')

    buildings = parse_buildings_csv(args.buildings_csv)

    for model in buildings:
        process_building(args.obj_dir_in, model)

# /home/graphicslab/OtherApps/blender-2.91.2-linux64/blender --background --python retexture_obj.py -- -obj_dir_in /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj -buildings_csv /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/buildings_temples_with_style.csv
