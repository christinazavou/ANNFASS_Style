
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from renderer_utils import EeveeRenderer
from scene_utils import cleanup, remove_obj
from io_utils import load_obj
import bpy

remove_obj(bpy.data.objects['Cube'])
objs = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj"
dest = "/home/graphicslab/Desktop/checkblender/meta"


renderer = EeveeRenderer(bpy.data.scenes[0])


for building_dir in os.listdir(objs):
    obj_file = os.path.join(objs, building_dir, f"{building_dir}.obj")
    load_obj(obj_file)

    objects = []
    for obj in bpy.data.objects:
        if obj.name is not 'Light' and obj.name is not 'Camera':
            obj.hide_render = False
            objects.append(obj)
        else:
            obj.hide_render = True

    renderer.multi_view(objects, dest, building_dir, bpy.data.objects['Camera'])

    cleanup(materials=True, except_names=['Camera', 'Light'])
