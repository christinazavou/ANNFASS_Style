import os
import sys
from shutil import copytree


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.utils import parse_buildings_with_style_csv


buildings_csv = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/buildings_religious_with_style.csv"
obj_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/normalizedObj"
groups_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/groups"
groups_dir_to_copy = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17"


buildings = parse_buildings_with_style_csv(buildings_csv)


def copy_dirs():
    for b in os.listdir(groups_dir_to_copy):
        if any(b == t[1] for t in buildings):
            copytree(os.path.join(groups_dir_to_copy, b), os.path.join(groups_dir, b))


def rename_text_occurrences(in_file, out_file, style):
    with open(in_file, "r") as fin:
        text = fin.read()
    text = text.replace("unknown", style)
    with open(out_file, "w") as fout:
        fout.write(text)


def rename_objs():
    for b in os.listdir(obj_dir):
        if any(b==t[1] for t in buildings):
            obj_file = os.path.join(obj_dir, b, f"{b}.obj")
            style = [t[0] for t in buildings if t[1] == b][0]
            rename_text_occurrences(obj_file, obj_file, style)


def rename_groups():
    for b in os.listdir(groups_dir):
        if any(b==t[1] for t in buildings):
            group_file = os.path.join(groups_dir, b, f"groups.json")
            style = [t[0] for t in buildings if t[1] == b][0]
            rename_text_occurrences(group_file, group_file, style)
            group_file = os.path.join(groups_dir, b, f"groups_fixed.json")
            rename_text_occurrences(group_file, group_file, style)


# rename_objs()
# copy_dirs()
rename_groups()
