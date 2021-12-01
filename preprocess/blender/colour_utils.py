import bpy


DAYLIGHT_COLOR = (1, 1, 243 / 255., 1)
SUNLIGHT_COLOR = (238./255, 220./255, 165./255, 1)
DISTINCT_COLOURS_40 = [
    "#a9a9a9",
    "#2f4f4f",
    "#556b2f",
    "#a0522d",
    "#800000",
    "#191970",
    "#808000",
    "#008000",
    "#3cb371",
    "#008b8b",
    "#9acd32",
    "#00008b",
    "#32cd32",
    "#daa520",
    "#8fbc8f",
    "#8b008b",
    "#ff0000",
    "#ff8c00",
    "#6a5acd",
    "#00ff00",
    "#9400d3",
    "#00fa9a",
    "#e9967a",
    "#dc143c",
    "#00bfff",
    "#0000ff",
    "#adff2f",
    "#ff7f50",
    "#ff00ff",
    "#db7093",
    "#f0e68c",
    "#ffff54",
    "#6495ed",
    "#dda0dd",
    "#ff1493",
    "#afeeee",
    "#ee82ee",
    "#7fffd4",
    "#ffe4c4",
    "#ffc0cb"
]
RED_COLOR = (0.9, 0.1, 0.1, 1)


def hex_to_rgb(hex):
    color = tuple(int(hex[1:][i:i + 2], 16) for i in (0, 2, 4))
    return (color[0] / 255., color[1] / 255., color[2] / 255.0, 1)


def to_rgb(value):
    assert isinstance(value, str) and '#' in value, "wrong color code type"
    return hex_to_rgb(value)


def get_40distinct_materials():
    material_colors = list()
    for color_idx in range(len(DISTINCT_COLOURS_40)):
        mat = bpy.data.materials.new(name="Color{}".format(color_idx))
        mat.diffuse_color = to_rgb(DISTINCT_COLOURS_40[color_idx])
        material_colors.append(mat)
    return material_colors


def colorize(material, objects):
    for obj in objects:
        if obj.type == 'MESH':
            obj.active_material = material


def remove_materials(objects,):
    for obj in objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
