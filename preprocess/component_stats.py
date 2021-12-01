import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common.utils import STYLES


# root_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may"
root_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings"

unique_dir = "unique_point_clouds"
ply_dir = "samplePoints/stylePly_cut10.0K_pgc"

coarse_dir = "samplePoints/stylePly_cut10.0K_pgc_content512"
detail_dir = "samplePoints/stylePly_cut10.0K_pgc_style4096"
out_name = "component_stats_512_4096.csv"
# coarse_dir = "samplePoints/stylePly_cut10.0K_pgc_content256"
# detail_dir = "samplePoints/stylePly_cut10.0K_pgc_style2048"
# out_name = "component_stats_256_2048.csv"


stats = {}
for building in os.listdir(os.path.join(root_dir, ply_dir)):
    if os.path.isdir(os.path.join(root_dir, ply_dir, building)):
        stats.setdefault(building, {
            'fse': {s: [] for s in STYLES},
            'coarse': {s: [] for s in STYLES},
            'detail': {s: [] for s in STYLES},
            'unique': {s: [] for s in STYLES}
        })
        for component_file in os.listdir(os.path.join(root_dir, ply_dir, building)):
            style = [s for s in STYLES if s.lower() in component_file.lower()]
            for s in style:
                stats[building]['fse'][s].append(component_file)


for building in os.listdir(os.path.join(root_dir, coarse_dir)):
    for component_file in os.listdir(os.path.join(root_dir, coarse_dir, building)):
        style = [s for s in STYLES if s.lower() in component_file.lower()]
        for s in style:
            stats[building]['coarse'][s].append(component_file)


for building in os.listdir(os.path.join(root_dir, detail_dir)):
    for component_file in os.listdir(os.path.join(root_dir, detail_dir, building)):
        style = [s for s in STYLES if s.lower() in component_file.lower()]
        for s in style:
            stats[building]['detail'][s].append(component_file)


for building in os.listdir(os.path.join(root_dir, unique_dir)):
    for component_file in os.listdir(os.path.join(root_dir, detail_dir, building)):
        if component_file.endswith(".ply"):
            style = [s for s in STYLES if s.lower() in component_file.lower()]
            for s in style:
                stats[building]['unique'][s].append(component_file)


# df = pd.DataFrame.from_dict(stats).to_csv(os.path.join(root_dir, out_name))


total_stats = {
    'fse': {s: 0 for s in STYLES},
    'unique': {s: 0 for s in STYLES},
    'coarse': {s: 0 for s in STYLES},
    'detail': {s: 0 for s in STYLES},
    'unique_coarse': {s: 0 for s in STYLES},
    'unique_detail': {s: 0 for s in STYLES},
}
for s in STYLES:
    for b in stats:
        total_stats['fse'][s] += len(stats[b]['fse'][s])
        total_stats['unique'][s] += len(stats[b]['unique'][s])
        total_stats['coarse'][s] += len(stats[b]['coarse'][s])
        total_stats['detail'][s] += len(stats[b]['detail'][s])
        total_stats['unique_coarse'][s] += len(set(stats[b]['coarse'][s]) & set(stats[b]['unique'][s]))
        total_stats['unique_detail'][s] += len(set(stats[b]['detail'][s]) & set(stats[b]['unique'][s]))

# pd.json_normalize(total_stats).to_csv(os.path.join(root_dir, out_name))
df = pd.DataFrame.from_dict(total_stats).to_csv(os.path.join(root_dir, out_name))


# def reform_dict(dictionary, t=tuple(), reform={}):
#     for key, val in dictionary.items():
#         t = t + (key,)
#         if isinstance(val, dict):
#             reform_dict(val, t, reform)
#         else:
#             reform.update({t: val})
#         t = t[:-1]
#     return reform
#
#
# for b in stats:
#     stats[b] = stats[b]['fse']
# pd.DataFrame.from_dict(reform_dict(stats), orient='index').to_csv(os.path.join(root_dir, out_name))
