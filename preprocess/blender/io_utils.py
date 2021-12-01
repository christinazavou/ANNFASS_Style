import os

import bpy


def load_dae(filepath):
    assert filepath.endswith(".dae"), "No .dae extension in " + filepath
    bpy.ops.wm.collada_import(filepath=filepath)


def load_fbx(filepath):
    assert filepath.endswith(".fbx"), "No .fbx extension in " + filepath
    imported_object = bpy.ops.import_scene.fbx(filepath=filepath)


def load_obj(filepath, rotated=False):
    assert filepath.endswith(".obj"), "No .obj extension in " + filepath
    if not rotated:
        bpy.ops.import_scene.obj(filepath=filepath)
    else:
        bpy.ops.import_scene.obj(filepath=filepath, axis_up='Z', axis_forward='Y')


def load_ply(filepath):
    assert filepath.endswith(".ply"), "No .ply extension in " + filepath
    bpy.ops.import_mesh.ply(filepath=filepath)


def save_simple_mesh(filepath, use_selection=True):
    assert filepath.endswith(".ply"), "No .ply extension in " + filepath
    bpy.ops.export_mesh.ply(filepath=filepath, use_selection=use_selection, use_uv_coords=False, use_colors=False)


def save_obj(filepath, use_selection=False, axis_up='Z'):
    if axis_up != 'Z':
        bpy.ops.export_scene.obj(filepath=filepath, axis_up='Y', axis_forward='Z', use_selection=use_selection)
    else:
        bpy.ops.export_scene.obj(filepath=filepath, axis_up='Z', axis_forward='Y', use_selection=use_selection)


def save_obj_with_textures(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=filepath.replace(".obj", ".blend"))
    try:
        bpy.ops.file.pack_all()
    except:
        print("WARNING: some files could not be packed :/")
    bpy.ops.file.unpack_all(method='USE_LOCAL')
    bpy.ops.export_scene.obj(filepath=filepath)
    os.remove(filepath.replace(".obj", ".blend"))


def save_ply(filepath, uv_coords=True, colors=True, global_scale=1.):
    if global_scale != 1.:
        bpy.ops.export_mesh.ply(filepath=filepath, use_uv_coords=uv_coords, use_colors=colors, global_scale=global_scale)
    else:
        bpy.ops.export_mesh.ply(filepath=filepath, use_uv_coords=uv_coords, use_colors=colors)

