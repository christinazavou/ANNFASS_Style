import bpy


def remove_obj(obj):
    bpy.data.objects.remove(obj, do_unlink=True)


def remove_objects(except_names=None):
    for obj in bpy.data.objects:
        if except_names is not None and obj.name in except_names:
            continue
        else:
            remove_obj(obj)


def add_copy_obj(obj, scene):
    new_obj = obj.copy()
    new_obj.data = obj.data.copy()
    scene.collection.objects.link(new_obj)
    return new_obj


def hide_obj(obj, hide, in_view, in_render):
    if in_view:
        obj.hide_viewport = hide
        obj.hide_set(hide)
    if in_render:
        obj.hide_render = hide


def select_objects(scene, names=None):
    select_all = True if names is None else False
    names = [] if names is None else names
    for obj in scene.objects:
        select = True if obj.name in names else False
        obj.select_set(select or select_all)


def deselect_all_objects(scene):
    for obj in scene.objects:
        obj.select_set(False)


def get_elements_with_style(scene):
    elements = []
    for obj in scene.objects:
        if "__" in obj.name:
            elements.append(obj.name)
    return elements


def add_scene():
    bpy.ops.scene.new(type='NEW')


def add_world():
    bpy.ops.world.new()


def get_scene(scene_str):
    return bpy.data.scenes[scene_str]


def set_active_scene(scene_str):
    bpy.context.window.scene = get_scene(scene_str)


def remove_materials():
    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)


def cleanup(materials=False, except_names=None):
    if materials:
        remove_materials()
    remove_objects(except_names=except_names)
    bpy.ops.outliner.orphans_purge()

