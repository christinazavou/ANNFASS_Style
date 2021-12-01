import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from common.utils import parse_buildings_csv, RunningBlenderCodeOutsideBlender


if __name__ == '__main__':
    try:
        from preprocess.blender.renderer_utils import EeveeRenderer, bpy, FreestyleRenderer, RidgeValleyRenderer
        from preprocess.blender.io_utils import load_obj
        from preprocess.blender.camera_utils import get_or_add_camera
        from preprocess.blender.scene_utils import remove_objects, remove_materials
    except:
        raise RunningBlenderCodeOutsideBlender("Running blender code without blender?")

    root_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings"
    renderer = EeveeRenderer(bpy.context.scene, low_resolution=False, fast_and_cheap=False)
    # renderer = FreestyleRenderer(bpy.context.scene, low_resolution=False, fast_and_cheap=False)
    # renderer = RidgeValleyRenderer(bpy.context.scene, low_resolution=False, fast_and_cheap=False)
    # renderer = WorkbenchRenderer(bpy.context.scene, low_resolution=False, fast_and_cheap=False)
    out_dir = os.path.join(root_dir, "religious_renderings", renderer.__class__.__name__)
    camera = get_or_add_camera()
    remove_objects(except_names=['Camera'])
    buildings = parse_buildings_csv("{}/buildings.csv".format(root_dir))
    for building in buildings:
        if "religious" in building.lower():
            print(building)
            obj_file = "{}/normalizedObj/{}/{}.obj".format(root_dir, building, building)
            last_out_file = os.path.join(out_dir, "{}r4.jpg".format(building))
            if not os.path.exists(last_out_file) and os.path.exists(obj_file):
                load_obj(obj_file)
                renderer.multi_view(bpy.data.objects, out_dir, building, camera)
                remove_objects(except_names=['Camera'])
                remove_materials()

# blender --background --python render_religious_buildings.py
