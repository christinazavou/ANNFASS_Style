import argparse
import json
import os
import sys
from os.path import join, exists

import pandas as pd

pd.set_option('display.max_columns', None)

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.utils import STYLES, parse_buildings_with_style_csv


building_stats = {}
part_stats = {}
for style in STYLES:
    building_stats[style + "_groupedcomponent"] = {}
    building_stats[style + "_uniquegroupedcomponent"] = {}
    part_stats[style + "_groupedcomponent"] = {}
    part_stats[style + "_uniquegroupedcomponent"] = {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups_dirs", default="/media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17,/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/groups", type=str)
    parser.add_argument("--unique_dirs", default="/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/unique_point_clouds,/media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds", type=str)
    parser.add_argument("--stats_csv", default="/media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/style_stats_selected_parts_31102021.csv", type=str)
    # parser.add_argument("--groups_dirs", default="/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/groups", type=str)
    # parser.add_argument("--unique_dirs", default="/media/graphicslab/BigData1/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/unique_point_clouds", type=str)
    # parser.add_argument("--stats_csv", default="/media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/style_stats_selected_parts_annfass_31102021.csv", type=str)
    # parser.add_argument("--groups_dirs", default="/media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17", type=str)
    # parser.add_argument("--unique_dirs", default="/media/graphicslab/BigData1/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17_unique_point_clouds", type=str)
    # parser.add_argument("--stats_csv", default="/media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/style_stats_selected_parts_buildnet_31102021.csv", type=str)

    parser.add_argument("--buildings_csv", default="/media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/buildings_with_style.csv", type=str)
    parser.add_argument("--parts", default="window,dome,door,column,tower", type=str)
    parser.add_argument("--include_txt", default="", type=str)
    args = parser.parse_args()

    parts = args.parts.split(",")
    assert len(parts) > 0

    buildings = parse_buildings_with_style_csv(args.buildings_csv)
    if args.include_txt != "":
        include = pd.read_csv(args.include_txt, sep="/", header=None)
        include = [(row[0], row[1]) for i, row in include.iterrows()]
    else:
        include = None

    for groups_dir in args.groups_dirs.split(","):
        for building in os.listdir(groups_dir):
            if not any(b == building for s, b in buildings):
                continue
            groups_file = join(groups_dir, building, "groups.json")
            if not exists(groups_file):
                continue
            with open(groups_file, "r") as fin:
                groups = json.load(fin)
            for group_name, group_components in groups.items():
                if include and not (building, int(group_name)) in include:
                    continue
                for style in STYLES:
                    building_stats[style + "_groupedcomponent"].setdefault(building, 0)
                    building_stats[style + "_uniquegroupedcomponent"].setdefault(building, 0)
                styles = []
                for c in group_components:
                    part = [p for p in parts if p.lower() in c.lower()]
                    if len(part) == 0:
                        continue
                    if len(part) > 1:
                        raise Exception
                    part = part[0]
                    for s in STYLES:
                        if s.lower() in c.lower():
                            styles.append(s)
                if len(styles) > 0:
                    style = styles[0]
                    if style.lower() == "unknown":
                        style = [s for (s, b) in buildings if b == building]
                        assert len(style) == 1
                        style = style[0]
                    building_stats[style + "_groupedcomponent"][building] += 1
                    part_stats[style + "_groupedcomponent"].setdefault(part, 0)
                    part_stats[style + "_groupedcomponent"][part] += 1

    for unique_dir in args.unique_dirs.split(","):
        for building in os.listdir(unique_dir):
            if not any(b == building for s, b in buildings):
                continue
            duplicates_file = join(unique_dir, building, "duplicates.json")
            if not exists(duplicates_file):
                continue
            with open(duplicates_file, "r") as fin:
                duplicates = json.load(fin)
            for unique in duplicates.keys():
                unique_component_group = int(unique.replace("style_mesh_", "").split("_")[0].replace("group", ""))
                if include and (building, unique_component_group) not in include:
                    continue
                part = [p for p in parts if p.lower() in unique.lower()]
                if len(part) == 0:
                    continue
                if len(part) > 1:
                    raise Exception
                part = part[0]
                styles = []
                for s in STYLES:
                    if s.lower() in unique.lower():
                        styles.append(s)
                if len(styles) > 0:
                    style = styles[0]
                    if style.lower() == "unknown":
                        style = [s for (s, b) in buildings if b == building]
                        assert len(style) == 1
                        style = style[0]
                    building_stats[style + "_uniquegroupedcomponent"][building] += 1
                    part_stats[style + "_uniquegroupedcomponent"].setdefault(part, 0)
                    part_stats[style + "_uniquegroupedcomponent"][part] += 1

    building_df = pd.DataFrame.from_dict(building_stats)
    part_df = pd.DataFrame.from_dict(part_stats)
    new_names = {}
    for style in STYLES:
        new_names[style+"_groupedcomponent"] = style
        new_names[style+"_uniquegroupedcomponent"] = style+".1"
    building_df = building_df.rename(columns=new_names)
    part_df = part_df.rename(columns=new_names)
    new_order = []
    for style in sorted(STYLES):
        new_order.append(style)
    for style in sorted(STYLES):
        new_order.append(style+".1")
    building_df = building_df[new_order]
    part_df = part_df[new_order]
    building_df_bkp = building_df
    part_df_bkp = part_df
    building_df.loc['Count'] = building_df_bkp.fillna(0).astype(bool).sum()
    building_df.loc['Total'] = building_df_bkp.sum()
    building_df.to_csv(args.stats_csv[:-4]+"_building.csv", index_label="building", sep=" ")
    part_df = part_df.sort_index(ascending=True)
    part_df.loc['Total'] = part_df_bkp.sum()
    part_df.to_csv(args.stats_csv[:-4]+"_part.csv", index_label="building", sep=" ")
