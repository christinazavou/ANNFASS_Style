import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import STYLES


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default="/media/graphicslab/BigData/zavou/ANNFASS_DATA", type=str)
parser.add_argument("--data_repo", default="ANNFASS_Buildings_may", type=str)
parser.add_argument("--groups_dir", default="groups", type=str)
parser.add_argument("--building", default="11_Hadjigeorgakis_Kornesios_Mansion", type=str)
args = parser.parse_args()

groups = json.load(open(os.path.join(args.root_dir, args.data_repo, args.groups_dir, args.building, "groups.json"), 'r'))

fq = defaultdict(int)
for style in STYLES:
    fq[style] = 0
for group_name, group in groups.items():
    try:
        group_styles = set()
        for group_component in group:
            try:
                current_styles = [x for x in STYLES if x.lower() in group_component.lower()]
                assert len(current_styles) == 1, f"Wow component {group_component} has more or less than 1 style"
                current_style = current_styles[0]
                group_styles.add(current_style)
            except AssertionError as e:
                print(e)
                pass
        assert len(group_styles) == 1, "group {} has {}".format(group_name, group_styles)
        fq[group_styles.pop()] += 1
    except AssertionError as e:
        print(e)
        pass

to_print = " ".join([str(x) for x in fq.values()])
print(to_print)
