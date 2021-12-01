import bmesh
import bpy
import numpy as np
from bpy_extras.object_utils import object_data_add
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def get_face_vertices(obj, face):
    return [obj.matrix_world @ v.co for v in face.verts]


def get_obj_vertices(obj):
    return [obj.matrix_world @ v.co for v in obj.data.vertices]


def get_obj_centroid(obj):
    vertices = get_obj_vertices(obj)
    min_xyz = np.amin(vertices, axis=0)
    max_xyz = np.amax(vertices, axis=0)
    min_xyz = Vector(min_xyz)
    max_xyz = Vector(max_xyz)
    mid_xyz = (max_xyz + min_xyz) / 2.
    return mid_xyz


def centralize_obj(obj):
    if obj.type == 'MESH':
        centroid = get_obj_centroid(obj)
        obj.matrix_world.translation -= centroid


def normalize_obj(obj):
    if obj.type == 'MESH':
        vertices_world = get_obj_vertices(obj)
        # normalize diagonal=1
        xyz_max = np.max(vertices_world, 0)
        xyz_min = np.min(vertices_world, 0)
        xyz_mid = (xyz_max + xyz_min) / 2
        xyz_scale = xyz_max - xyz_min
        scale = np.linalg.norm(xyz_scale)
        # obj.scale = Vector((1/scale, 1/scale, 1/scale))  # this is not good because it changes obj location
        for vertex in obj.data.vertices:
            vertex.co = (vertex.co - Vector(xyz_mid))/scale
        centralize_obj(obj)


def get_vertices(objs):
    vertices = []
    for obj in objs:
        if obj.type == 'MESH':
            vertices += get_obj_vertices(obj)
    vertices = np.array(vertices)
    return vertices


def get_centroid(objs):
    vertices = get_vertices(objs)
    min_xyz = np.amin(vertices, axis=0)
    max_xyz = np.amax(vertices, axis=0)
    return Vector((max_xyz + min_xyz) / 2.)


def triangulate_object_mesh(obj):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
    bm.to_mesh(me)
    bm.free()


def triangulate_all():
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            triangulate_object_mesh(obj)


def add_plane_given_vertices(context, vertices, name="NewPlane"):
    assert len(vertices) == 4
    edges = []
    faces = [[0, 1, 2, 3]]
    mesh = bpy.data.meshes.new(name=name)
    mesh.from_pydata(vertices, edges, faces)
    object_data_add(context, mesh)


def add_triangle_given_vertices(context, vertices, name="NewTriangle"):
    assert len(vertices) == 3
    edges = []
    faces = [[0, 1, 2]]
    mesh = bpy.data.meshes.new(name=name)
    mesh.from_pydata(vertices, edges, faces)
    object_data_add(context, mesh)


def uvs_are_unnormalized(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='TOGGLE')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.verify()
    for f in bm.faces:
        for l in f.loops:
            luv = l[uv_layer]
            if not (0 <= luv.uv.x <= 1) or not (0 <= luv.uv.y <= 1):
                bpy.ops.object.mode_set(mode='OBJECT')
                print("Obj {} has unnormalized uvs".format(obj.name))
                return True
    bpy.ops.object.mode_set(mode='OBJECT')
    return False


def smart_uv_project(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.03)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)


def unwrap(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)


def update_uv_dict(obj, material_uv_bounds_dict):
    if obj.type != 'MESH':
        return
    print(f"update uv dict given {obj.name}")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.verify()
    for f in bm.faces:
        mat = obj.material_slots[f.material_index].material
        if material_has_texture_image(mat):
            material_name = mat.name
            if material_name in material_uv_bounds_dict:
                material_bounds = material_uv_bounds_dict[material_name]
            else:
                material_bounds = {'min_u': 1e9, 'min_v': 1e9, 'max_u': 1e-9, 'max_v': 1e-9}
            update_material_bounds_edit_mode(material_bounds, f, uv_layer)
    bpy.ops.object.mode_set(mode='OBJECT')


def update_material_bounds_edit_mode(material_uv_bounds, face, uv_layer):
    for l in face.loops:
        luv = l[uv_layer]
        material_uv_bounds['min_u'] = min(luv.uv.x, material_uv_bounds['min_u'])
        material_uv_bounds['min_v'] = min(luv.uv.y, material_uv_bounds['min_v'])
        material_uv_bounds['max_u'] = max(luv.uv.x, material_uv_bounds['max_u'])
        material_uv_bounds['max_v'] = max(luv.uv.y, material_uv_bounds['max_v'])


def update_material_bounds_object_mode(material_uv_bounds, face, obj):
    for i in face.loop_indices:
        l = obj.data.loops[i]
        for ul in obj.data.uv_layers:
            luv = ul.data[l.index]
            material_uv_bounds['min_u'] = min(luv.uv.x, material_uv_bounds['min_u'])
            material_uv_bounds['min_v'] = min(luv.uv.y, material_uv_bounds['min_v'])
            material_uv_bounds['max_u'] = max(luv.uv.x, material_uv_bounds['max_u'])
            material_uv_bounds['max_v'] = max(luv.uv.y, material_uv_bounds['max_v'])


def material_has_texture_image(mat):
    material_has_img = False
    if mat and mat.use_nodes:
        for n in mat.node_tree.nodes:
            if n.type == 'BSDF_PRINCIPLED' and n.inputs[0].links:
                if n.inputs[0].links[0].from_node.type == 'TEX_IMAGE':
                    material_has_img = True
                    break
    return material_has_img


def add_shader_nodes_for_rescaling(material, scale_x, scale_y):
    img_node = None
    for n in material.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED' and n.inputs[0].links:
            img_node = n.inputs[0].links[0].from_node
    if img_node is None:
        return
    texture_coord_node = material.node_tree.nodes.new('ShaderNodeTexCoord')
    mapping_node = material.node_tree.nodes.new('ShaderNodeMapping')
    material.node_tree.links.new(texture_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
    material.node_tree.links.new(mapping_node.outputs['Vector'], img_node.inputs['Vector'])
    mapping_node.inputs['Scale'].default_value = (scale_x, scale_y, 1)


def select_related_faces_and_update_bounds(material_uv_bounds_dict, material_name):
    selected_faces_cnt = 0
    selected_objects_cnt = 0
    for obj in bpy.data.objects:
        obj_select = False
        if obj.type == 'MESH':
            for f in obj.data.polygons:
                f.select = False
                mat = obj.material_slots[f.material_index].material
                if mat and mat.use_nodes:
                    if mat.name == material_name:
                        bpy.context.view_layer.objects.active = obj  # make sure our object is active otherwise object mode cant be set and faces wont be really selected
                        bpy.ops.object.mode_set(mode='OBJECT')  # make sure we are in object mode otherwise faces wont be selected
                        f.select = True
                        selected_faces_cnt += 1
                        obj_select = True
                        selected_objects_cnt += 1
                        update_material_bounds_object_mode(material_uv_bounds_dict, f, obj)
        obj.select_set(obj_select)
    return selected_faces_cnt, selected_objects_cnt


# NOTE: USE THIS FUNCTION MANUALLY IN BLENDER TO UNWRAP SPECIFIC FACES OF A MATERIAL !!
def select_related_faces(material_name):
    selected_faces_cnt = 0
    selected_objects_cnt = 0
    for obj in bpy.data.objects:
        obj_select = False
        if obj.type == 'MESH':
            for f in obj.data.polygons:
                f.select = False
                mat = obj.material_slots[f.material_index].material
                if mat and mat.use_nodes:
                    if mat.name == material_name:
                        bpy.context.view_layer.objects.active = obj  # make sure our object is active otherwise object mode cant be set and faces wont be really selected
                        bpy.ops.object.mode_set(mode='OBJECT')  # make sure we are in object mode otherwise faces wont be selected
                        f.select = True
                        selected_faces_cnt += 1
                        obj_select = True
                        selected_objects_cnt += 1
        obj.select_set(obj_select)
    return selected_faces_cnt, selected_objects_cnt


def unwrap_and_deselect():
    bpy.ops.object.mode_set(mode='EDIT')
    # bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')
    # make sure nothing is selected
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for f in obj.data.polygons:
                f.select = False


def clean_mesh():
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.delete_loose()
    bpy.ops.object.editmode_toggle()


def objects_are_touching(obj1, obj2):
    bm1 = bmesh.new()
    bm2 = bmesh.new()
    bm1.from_mesh(obj1.data)
    bm2.from_mesh(obj2.data)
    bm1.transform(obj1.matrix_world)
    bm2.transform(obj2.matrix_world)
    obj_now_BVHtree = BVHTree.FromBMesh(bm1)
    obj_next_BVHtree = BVHTree.FromBMesh(bm2)
    inter = obj_now_BVHtree.overlap(obj_next_BVHtree)
    if inter != []:
        return True
    return False


def overall_scene_mesh(scene):
    bm = bmesh.new()
    for obj in scene.objects:
        if obj.type == 'MESH':
            bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    return bm


def obj_mesh(obj):
    bm = bmesh.new()
    assert obj.type == 'MESH', f"Can't get the mesh of an object with {obj.type} type"
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    return bm


def point_inside_bbox_mesh(bbox_mesh, bbox_obj, point):
    for face_idx, face in enumerate(bbox_mesh.faces):
        face_center = bbox_obj.matrix_world @ face.calc_center_median()
        face_norm = bbox_obj.matrix_world @ face.normal
        point_dir = point - face_center
        if point_dir.dot(face_norm) > 0:
            return False
    return True


def set_normals_outwards(scene):
    for obj in scene.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='TOGGLE')
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')


def simplify_mesh():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.decimate()
    bpy.ops.mesh.dissolve_degenerate()
    bpy.ops.object.editmode_toggle()

