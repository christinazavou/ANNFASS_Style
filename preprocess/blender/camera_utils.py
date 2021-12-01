import bpy
import math
from mathutils import Matrix, Vector


def get_or_add_camera(init_location=None):
    # TODO: maybe adjust clip_start or clip_end
    all_objects = bpy.data.objects
    active_scene = bpy.context.scene
    if 'Camera' in all_objects.keys():
        camera = all_objects['Camera']
    else:
        camera = all_objects.new("Camera", bpy.data.cameras.new("Camera"))
        active_scene.collection.objects.link(camera)
        active_scene.camera = camera  # set active camera
        if init_location is not None:
            camera.location = init_location
    return camera


def get_field_of_view(camera):
    return math.radians(camera.lens)


def camera_distance_from_sphere(radius, camera):
    return (radius * 2) / (math.tan(get_field_of_view(camera) / 2))


def set_camera_view_on_bbox_sphere(camera, bbox, radius, center):
    camera_distance = camera_distance_from_sphere(radius, bpy.data.cameras['Camera'])
    camera.matrix_world = bbox.matrix_world
    camera.matrix_world.translation = Vector((0, camera_distance, bbox.dimensions[2] / 2))
    rotate_towards_target(camera, center)


def rotate_towards_target(camera, target_center):
    direction = camera.location - target_center
    rotation = direction.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    camera.matrix_world = Matrix.Translation(camera.location) @ rotation
