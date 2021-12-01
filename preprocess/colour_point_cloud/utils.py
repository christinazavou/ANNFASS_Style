import math
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from common.mesh_utils import Material, MaterialWithTexture, get_component_points
from common import SampledPointWithColor

CANTUSE = {}

class AngleCalculationException(Exception):
    def __init__(self, message):
        super().__init__(message)


class PointOutsideFaceException(Exception):
    def __init__(self, message):
        super().__init__(message)


class UVInterpolationOutsideBoundsException(Exception):
    def __init__(self, message):
        super().__init__(message)


def get_colour_from_simple_material(material):
    return np.array([int(material.kd["r"]*255), int(material.kd["g"]*255), int(material.kd["b"]*255)])


def get_colour_from_material_with_texture(mat_h:int, mat_w:int, pix:np.ndarray, uv):
    # FOR BLENDER VT
    u = uv[0]
    v = uv[1]
    v = 1 - v
    return get_colour_with_bilinear_interpolation(mat_h, mat_w, pix, u, v)


def get_pixel(material, uv):
    # FOR BLENDER VT
    u = uv[0]
    v = uv[1]
    v = 1 - v

    uv_coord_pixel_row = (material.h - 1) * v
    uv_coord_pixel_col = (material.w - 1) * u

    uv_pixel_row_1 = int(np.floor(uv_coord_pixel_row))
    uv_pixel_col_1 = int(np.floor(uv_coord_pixel_col))
    return uv_pixel_row_1, uv_pixel_col_1


def get_colour_with_bilinear_interpolation(mat_h: int, mat_w: int, pix: np.ndarray, u: float, v: float):

    uv_coord_pixel_row = (mat_h - 1) * v
    uv_coord_pixel_col = (mat_w - 1) * u

    uv_pixel_row_1 = int(np.floor(uv_coord_pixel_row))
    uv_pixel_row_2 = int(np.ceil(uv_coord_pixel_row))
    uv_pixel_col_1 = int(np.floor(uv_coord_pixel_col))
    uv_pixel_col_2 = int(np.ceil(uv_coord_pixel_col))

    d1 = uv_coord_pixel_row % 1
    d2 = 1 - d1
    d3 = uv_coord_pixel_col % 1
    d4 = 1 - d3

    if uv_pixel_row_2 == mat_h:
        print("V1 same as V3, V2 same as V4, d3 {} d4 {}".format(d3, d4))
    if uv_pixel_col_2 == mat_w:
        print("V1 same as V2, V3 same as V4, d1 {}, d2 {}".format(d1, d2))

    area1 = d2 * d4
    area2 = d1 * d4
    area3 = d2 * d3
    area4 = d1 * d3

    colorV1 = np.array(pix[uv_pixel_row_1, uv_pixel_col_1][0:3])
    colorV2 = np.array(pix[uv_pixel_row_1, uv_pixel_col_2][0:3])
    colorV3 = np.array(pix[uv_pixel_row_2, uv_pixel_col_1][0:3])
    colorV4 = np.array(pix[uv_pixel_row_2, uv_pixel_col_2][0:3])

    color = colorV1 * area1 + colorV2 * area2 + colorV3 * area3 + colorV4 * area4

    return color


def valid_uv_coords(uv):
    return np.all(np.logical_and(uv <= 1, uv >= 0))


def normalize_uv_coords(coords):
    assert not np.any(np.isnan(coords))
    return _blender_normalize(coords)


def normalize_uv_coords_given_bounds(coords, uv_min, uv_max):
    assert not np.any(np.isnan(coords))
    return _within_bounds_normalize(coords, uv_min, uv_max)


def _blender_normalize(coords):
    # blender is using tile repeat thus:
    return coords % 1   # -2.2, 3.3 --> 0.8, 0.3


def _within_bounds_normalize(coords, uv_min, uv_max):
    return (coords - uv_min) / (uv_max - uv_min)


def areaOfTriangle(v1, v2, v3):
    edge1 = (v2 - v1)
    edge2 = (v3 - v1)

    area = np.linalg.norm(np.cross(edge1, edge2)) / 2.0
    return area


def lies_on_edge(point, a, b, c):
    try:
        ap_ab_angle = angle_in_degrees_between_3d_vectors((b - a), (point - a))
        ap_ac_angle = angle_in_degrees_between_3d_vectors((c - a), (point - a))
        cp_cb_angle = angle_in_degrees_between_3d_vectors((b - c), (point - c))
        # print("angles {}, {}, {}".format(ap_ab_angle, ap_ac_angle, cp_cb_angle))
        return min(ap_ab_angle, ap_ac_angle, cp_cb_angle)
    except AngleCalculationException as e:
        print(str(e))
        raise Exception("Issue with angle calculation when point is {} and a,b,c are: {},{},{}".format(point, a, b, c))


def angle_in_degrees_between_3d_vectors(a, b):
    if np.all(np.equal(a, b)):
        return 0
    d = np.dot(a, b) / (vector_length(a) * vector_length(b))
    try:
        angle = math.degrees(math.acos(d))
        return angle
    except:
        if np.isclose(d, 1, atol=0.001) and (np.any(np.equal(a, b)) or len(np.unique(a/b)) == 1):
            return 0
        raise AngleCalculationException("Couldn't calculate angle between a ({}) and b ({})".format(a, b))


def vector_length(vec):
    return np.linalg.norm(vec)


def triangle_interpolation_for_uv_coords(vertices_coords, uv_coords, point):
    a_uv, b_uv, c_uv = uv_coords

    if not valid_uv_coords(a_uv):
        a_uv = normalize_uv_coords(a_uv)
    if not valid_uv_coords(b_uv):
        b_uv = normalize_uv_coords(b_uv)
    if not valid_uv_coords(c_uv):
        c_uv = normalize_uv_coords(c_uv)

    return triangle_interpolation_for_normalized_uv_coords(vertices_coords, a_uv, b_uv, c_uv, point)


def triangle_interpolation_for_normalized_uv_coords(vertices_coords, a_uv, b_uv, c_uv, point):
    a, b, c = vertices_coords

    if np.all(np.equal(a, b)) and np.all(np.equal(a, c)):
        # print("the face is actually a point:/")
        return a_uv

    if np.all(np.isclose(a, point, atol=1e-6)):
        return a_uv  # point lies on vertex a
    if np.all(np.isclose(b, point, atol=1e-6)):
        return b_uv  # point lies on vertex b
    if np.all(np.isclose(c, point, atol=1e-6)):
        return c_uv  # point lies on vertex c

    area_abc = areaOfTriangle(a, b, c)

    if area_abc == 0:
        # print("the face is actually a point:/")
        return a_uv

    area_pcb = areaOfTriangle(point, b, c) / area_abc
    area_apc = areaOfTriangle(a, point, c) / area_abc
    area_abp = areaOfTriangle(a, b, point) / area_abc

    if np.sum(np.equal([area_pcb, area_apc, area_abp], [0, 0, 0])) > 0:
        if np.isclose(lies_on_edge(point, *vertices_coords), 0, atol=0.1):
            # print("point lies on edge")
            pass
        else:
            print("HMM")
    elif not np.isclose(area_pcb+area_apc+area_abp, 1, atol=0.05):
        if np.all(np.isclose(a, b, atol=1e-4)) or np.all(np.isclose(a, c, atol=1e-4)) or np.all(np.isclose(b, c, atol=1e-4)):
            # print("face is almost like an edge")
            pass
        else:
            print("Please check if the point ({}) lies outside the face due to numerical issue."
                  "\na: {}, b: {}, c: {}\nWe will use the barycenter of the face as point P.".format(point, a, b, c))
            area_pcb = area_apc = area_abp = 1 / 3.
    elif area_pcb > 1 or area_apc > 1 or area_abp > 1:
        print("Please check if the point ({}) lies outside the face due to numerical issue."
              "\na: {}, b: {}, c: {}\nWe will use the barycenter of the face as point P.".format(point, a, b, c))
        area_pcb = area_apc = area_abp = 1/3.

    uv_point = a_uv * area_pcb + b_uv * area_apc + c_uv * area_abp
    if not valid_uv_coords(uv_point):
        if not np.all(np.isclose(uv_point[uv_point > 1], 1, atol=0.01)):
            raise UVInterpolationOutsideBoundsException("uv interpolated: {}. "
                "a,b,c: {}, {}, {}. a_uv,b_uv,c_uv: {},{},{}".format(uv_point, a, b, c, a_uv, b_uv, c_uv))
        uv_point = np.clip(uv_point, 0, 1)

    return uv_point


class TextureRescaled:

    def __init__(self, init_img: np.ndarray):
        assert init_img.ndim == 3 and init_img.dtype >= np.uint8
        self.init_img = init_img
        self.init_w = self.init_img.shape[1]
        self.init_h = self.init_img.shape[0]

    def rescale(self, repeat_w: int, repeat_h: int):
        # numpy.ndarray: [h, w, 3]
        # PIL.Image: [w, h, 3]
        repeat_img = Image.fromarray(self.init_img).resize((repeat_w, repeat_h))
        new_img = np.zeros((self.init_h, self.init_w, 3)).astype(np.uint8)
        for i in range(0, self.init_h, repeat_h):
            i_max = min(i + repeat_h, self.init_h)
            for j in range(0, self.init_w, repeat_w):
                j_max = min(j+repeat_w, self.init_w)
                new_img[i:i_max, j:j_max, :] = np.array(repeat_img)[0:i_max-i, 0:j_max-j]
        return new_img


class DebugComponentTexture:

    def __init__(self, init_img: np.ndarray):
        self.init_img = init_img[:, :, 0:3]

    def update(self, row, col, colour=np.array([255, 0, 0])):
        row_low = max(0, row - 8)
        row_high = min(self.init_img.shape[0], row + 8)
        col_low = max(0, col - 8)
        col_high = min(self.init_img.shape[1], col + 8)
        self.init_img[row_low: row_high, col_low: col_high] = colour

    def save(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.fromarray(self.init_img)
        img.save(output_path)


def write_ply_with_colour(sampled_points_with_colour, filename):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(sampled_points_with_colour)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")
        for spwc in sampled_points_with_colour:
            f.write("{} {} {} {} {} {} 255\n".format(spwc.coords[0], spwc.coords[1], spwc.coords[2],
                                                     int(spwc.color_interp[0]),
                                                     int(spwc.color_interp[1]),
                                                     int(spwc.color_interp[2])))


# def find_point_colour(point, obj, uv_bounds=None, rescaled_textures=None):
#     face = obj.faces[point.face_idx]
#     mat = obj.materials[face.material]
#
#     if isinstance(mat, MaterialWithTexture):
#         scale_w, scale_h = 1, 1
#         rescaled_textures.setdefault(face.material, {(scale_w, scale_h): {'img': mat.pix, 'used_by': set()}})
#
#         assert face.vt_indices
#         f_v_coords = obj.vertex_coords[face.v_indices]
#         f_uv_coords = obj.texture_coords[face.vt_indices]
#         uv = triangle_interpolation_for_uv_coords(f_v_coords, f_uv_coords, point.coords)
#
#         use_img = mat.pix
#         if uv_bounds is not None:
#             if uv_bounds == 'material_component':
#                 uv_dict = obj.min_max_uv_per_material_per_component[face.material][face.component]
#                 real_uv_bounds = np.array(list(uv_dict.values()))
#                 scale_w = abs(uv_dict['min_u']) + abs(uv_dict['max_u'])
#                 scale_h = abs(uv_dict['min_v']) + abs(uv_dict['max_v'])
#             elif uv_bounds == 'material':
#                 uv_dict = obj.min_max_uv_per_material_per_component[face.material]
#                 uv_min = np.array([[val['min_u'], val['min_v']] for val in uv_dict.values()]).min(0)
#                 uv_max = np.array([[val['max_u'], val['max_v']] for val in uv_dict.values()]).max(0)
#                 real_uv_bounds = np.concatenate([uv_min, uv_max])
#                 scale_w = abs(uv_min[0]) + abs(uv_max[0])
#                 scale_h = abs(uv_min[1]) + abs(uv_max[1])
#             else:
#                 raise Exception(f"unknown uv_bounds {uv_bounds}")
#             scale_w = max(1, np.round(scale_w).astype(int))
#             scale_h = max(1, np.round(scale_h).astype(int))
#             if not all(np.logical_and(0 <= real_uv_bounds, real_uv_bounds <= 1)):
#                 if scale_w > 1 or scale_h > 1:
#                     if (scale_w, scale_h) in rescaled_textures[face.material]:
#                         use_img = rescaled_textures[face.material][(scale_w, scale_h)]['img']
#                     else:
#                         use_img = TextureRescaled(mat.pix).rescale(int(mat.w / scale_w), int(mat.h / scale_h))
#                         rescaled_textures[face.material][(scale_w, scale_h)] = {'img': use_img, 'used_by': set()}
#
#         rescaled_textures[face.material][(scale_w, scale_h)]['used_by'].add(face.component)
#         return uv, get_colour_from_material_with_texture(use_img.shape[0], use_img.shape[1], use_img, uv)
#
#     elif isinstance(mat, Material):
#         return None, get_colour_from_simple_material(mat)
#     else:
#         raise Exception("unknown material")


def find_point_colour(point, obj, uv_bounds=None, rescaled_textures=None):
    face = obj.faces[point.face_idx]
    mat = obj.materials[face.material]

    if isinstance(mat, MaterialWithTexture):
        scale_w, scale_h = 1, 1
        rescaled_textures.setdefault(face.material, {(scale_w, scale_h): {'img': mat.pix, 'used_by': set()}})

        assert face.vt_indices
        f_v_coords = obj.vertex_coords[face.v_indices]
        f_uv_coords = obj.texture_coords[face.vt_indices]
        uv = triangle_interpolation_for_uv_coords(f_v_coords, f_uv_coords, point.coords)

        use_img = mat.pix
        if mat.scale_x != 1 or mat.scale_y != 1:

            scale_w = mat.scale_y
            scale_h = mat.scale_x
            if (scale_w, scale_h) in rescaled_textures[face.material]:
                use_img = rescaled_textures[face.material][(scale_w, scale_h)]['img']
            else:
                rescale_w = int(mat.w / scale_w)
                rescale_h = int(mat.h / scale_h)
                if rescale_w == 0 or rescale_h == 0:
                    CANTUSE[mat.name] = [mat.w, mat.h, scale_w, scale_h]
                    use_img = mat.pix
                    rescaled_textures[face.material][(scale_w, scale_h)] = {'img': use_img, 'used_by': set()}
                else:
                    use_img = TextureRescaled(mat.pix).rescale(int(mat.w / scale_w), int(mat.h / scale_h))
                    rescaled_textures[face.material][(scale_w, scale_h)] = {'img': use_img, 'used_by': set()}

        rescaled_textures[face.material][(scale_w, scale_h)]['used_by'].add(face.component)
        return uv, get_colour_from_material_with_texture(use_img.shape[0], use_img.shape[1], use_img, uv)

    elif isinstance(mat, Material):
        return None, get_colour_from_simple_material(mat)
    else:
        raise Exception("unknown material")


def process_building(obj, sampled_points, uv_bounds=None):
    sampled_points_with_colour = []
    rescaled_textures = dict()
    for point in tqdm(sampled_points):
        try:
            uv, color = find_point_colour(point, obj, uv_bounds, rescaled_textures)
            sampled_points_with_colour.append(SampledPointWithColor(point.coords, point.face_idx, uv, color))
        except PointOutsideFaceException as e:
            print(e)
        except UVInterpolationOutsideBoundsException as e:
            print(e)
    print("CAN't use: ", CANTUSE)
    return sampled_points_with_colour, rescaled_textures


def debug_building_textures(obj, sampled_points_with_colour, rescaled_textures, debug_textures_dir):
    for component_id in tqdm(obj.components):
        component_points = get_component_points(obj, sampled_points_with_colour, component_id)
        for material_name, material in obj.materials.items():
            component_material_points = [p for p in component_points if obj.faces[p.face_idx].material == material_name]

            if len(component_material_points) == 0:
                continue

            if material_name in rescaled_textures:
                for scale_w, scale_h in rescaled_textures[material_name]:
                    img = rescaled_textures[material_name][(scale_w, scale_h)]['img']
                    used_by = rescaled_textures[material_name][(scale_w, scale_h)]['used_by']
                    if component_id in used_by:
                        debug_texture = DebugComponentTexture(img)
                        for point in component_material_points:
                            pr, pc = get_pixel(material, point.uv_interp)
                            debug_texture.update(pr, pc, colour=np.array([255, 0, 0]))
                        debug_texture.save(os.path.join(debug_textures_dir, component_id, f"{material_name}_{scale_w}_{scale_h}.jpg"))
