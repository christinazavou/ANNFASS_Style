import argparse
import logging
import os
import sys
from os.path import dirname, realpath, exists, join, basename

STYLE_DIR = dirname(dirname(dirname(dirname(realpath(__file__)))))
sys.path.append(STYLE_DIR)
from common.utils import set_logger_file, parse_buildings_csv
from preprocess.blender.scene_utils import select_objects, cleanup, bpy
from preprocess.blender.io_utils import load_obj, save_obj
from preprocess.blender.mesh_utils import obj_mesh, normalize_obj, Vector


LOGGER = logging.getLogger(__name__)


def detect_triangle_orientation(v0, v1, v2):
    # given that normal is pointing in expected direction
    if v0.z > v1.z and v0.z > v2.z:
        # v0 is fv0_
        return 0, 1, 2
    else:
        if v1.z > v0.z and v1.z > v2.z:
            # v1 is fv0_
            return 1, 2, 0
        else:
            # v2 is fv0_
            return 2, 0, 1


def add_volume_to_face(face, bm, obj):
    fnorm_ = obj.matrix_world @ face.normal
    fv0_ = obj.matrix_world @ face.verts[0].co
    fv1_ = obj.matrix_world @ face.verts[1].co
    fv2_ = obj.matrix_world @ face.verts[2].co
    idx0, idx1, idx2 = detect_triangle_orientation(fv0_, fv1_, fv2_)
    fv0_ = obj.matrix_world @ face.verts[idx0].co
    fv1_ = obj.matrix_world @ face.verts[idx1].co
    fv2_ = obj.matrix_world @ face.verts[idx2].co
    fv0ext_ = fv0_ + 0.1 * fnorm_
    fv1ext_ = fv1_ + 0.1 * fnorm_
    fv2ext_ = fv2_ + 0.1 * fnorm_
    #
    v0ext = bm.verts.new( face.verts[idx0].co + 0.1 * face.normal )
    v1ext = bm.verts.new( face.verts[idx1].co + 0.1 * face.normal )
    v2ext = bm.verts.new( face.verts[idx2].co + 0.1 * face.normal )
    #
    # note: normal is important for visualization without culling ...
    # note: vertices should be defined anti-clockwise
    ftop1 = bm.faces.new( [face.verts[idx1], v0ext, face.verts[idx0]])
    ftop2 = bm.faces.new( [face.verts[idx1], v1ext, v0ext])
    fbottom1 = bm.faces.new( [face.verts[idx1], v2ext, face.verts[idx2]])
    fbottom2 = bm.faces.new( [face.verts[idx1], v1ext, v2ext])
    fside1 = bm.faces.new( [face.verts[idx2], v2ext, face.verts[idx0]])
    fside2 = bm.faces.new( [face.verts[idx0], v2ext, v0ext])
    fext = bm.faces.new( [v0ext, v1ext, v2ext])
    bm.to_mesh(obj.data)


def get_face_normals_dict(plane_mesh, component):
    face_normals = dict()
    for face in plane_mesh.faces:
        fn = component.matrix_world @ face.normal
        fn = Vector((round(fn.x, 1), round(fn.y, 1), round(fn.z, 1))).freeze()
        face_normals.setdefault(fn, [])
        face_normals[fn].append(face)
    return face_normals


def process_component():
    component = bpy.data.objects[0]
    plane_mesh = obj_mesh(component)

    face_normals = get_face_normals_dict(plane_mesh, component)

    face_normals_keys = list(face_normals.keys())
    if len(face_normals_keys) == 2 and face_normals_keys[0] == -face_normals_keys[1]:
        print("this component is a doubled-faced plane.")
    elif len(face_normals_keys) > 1:
        print("this component is not a plane with one unique face normal.")
        return False
    face_normal = face_normals_keys[0]

    for face in face_normals[face_normal]:
        add_volume_to_face(face, plane_mesh, component)
    component.data.update()

    # unify
    select_objects(bpy.context.scene)
    bpy.context.view_layer.objects.active = component
    bpy.ops.object.join()

    #normalize
    obj = bpy.context.active_object
    normalize_obj(obj)

    return True


if __name__ == '__main__':

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('-root_data', type=str)
        parser.add_argument('-in_dir', type=str, default="unified_normalized_components")
        parser.add_argument('-out_dir', type=str, default="unified_normalized_components")
        parser.add_argument('-buildings_csv', type=str, default="buildings.csv")
        parser.add_argument('-logs_dir', type=str)
        args = parser.parse_known_args(argv)[0]
    else:
        raise Exception('please give args')

    if not exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    _log_file = join(args.logs_dir, f'{basename(__file__)}.log')
    LOGGER = set_logger_file(_log_file, LOGGER)

    LOGGER.info("Starting...")

    LOGGER.info("root_data {}".format(args.root_data))
    LOGGER.info("in_dir {}".format(args.in_dir))
    LOGGER.info("out_dir {}".format(args.out_dir))
    LOGGER.info("buildings_csv {}".format(args.buildings_csv))

    buildings = parse_buildings_csv(join(args.root_data, args.buildings_csv))
    LOGGER.info("buildings: {}".format(buildings))

    for idx, building in enumerate(buildings):
        building_dir = os.path.join(args.root_data, args.in_dir, building)
        if not os.path.exists(building_dir):
            continue
        for grouped_component in os.listdir(building_dir):
            if not any(c in grouped_component for c in ['door', 'window']):
                continue
            print(f"processing building {building}, grouped_component {grouped_component}")
            obj_file_in = join(args.root_data, args.in_dir, building, grouped_component, "model.obj")
            obj_file_out = join(args.root_data, args.out_dir, building, grouped_component, "model.obj")
            if not os.path.exists(obj_file_in):
                continue
            if not exists(obj_file_out):
                cleanup(True)
                load_obj(obj_file_in)
                to_save = process_component()
                if to_save:
                    print("saving...")
                    os.makedirs(dirname(obj_file_out), exist_ok=True)
                    save_obj(obj_file_out, use_selection=True, axis_up='Y')


# blender --background --python add_volume_to_planes.py -- -root_data /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings -in_dir groups_june17_uni_nor_components -out_dir groups_june17_uni_nor_planes_with_volume -buildings_csv buildings.csv -logs_dir /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/logsvolume

