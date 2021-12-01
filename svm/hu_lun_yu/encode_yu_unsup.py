import os.path
import argparse

import numpy as np
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, type=str)
parser.add_argument("--suffix", required=True, type=str)
parser.add_argument("--classes", required=True, type=str, help="e.g. C1,C2,C3,C4,C5,C6,C7,C8")
args = parser.parse_args()


nb_classes = len(args.classes.split(","))


def indices_to_one_hot(vec, ):
    targets = np.array(vec).reshape(-1)
    return np.eye(nb_classes)[targets]


pslf_result = loadmat(f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/proj_style_data/data/{args.data}/pslf{args.suffix}_unsup/result.mat")
G, Vr, neworder = pslf_result['G'], pslf_result['Vr'], pslf_result['neworder']

with open(f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/proj_style_data/data/{args.data}/model_names.txt", "r") as fin:
    modelnames = fin.readlines()

out_dir = f"/media/graphicslab/BigData/zavou/ANNFASS_CODE/proj_style_data/data/{args.data}/encodings{args.suffix}_unsup"
# for i, (g, vr) in enumerate(zip(G, Vr)):
#     model = modelnames[neworder[i,0]-1]
#     building, component = model.rstrip().split("_group")
#     component = "group" + component.replace(".obj", "")
#     os.makedirs(os.path.join(out_dir, building), exist_ok=True)
#     out_file = os.path.join(out_dir, building, component + ".npy")
#     np.save(out_file, vr)
#     out_file = os.path.join(out_dir, building, component + "_labels.npy")
#     np.save(out_file, indices_to_one_hot(g)[0])
for i, (g, vr) in enumerate(zip(G, Vr)):
    model = modelnames[neworder[i,0]-1]
    modelname = model.rstrip().replace(".obj", "")
    os.makedirs(os.path.join(out_dir, modelname), exist_ok=True)
    out_file = os.path.join(out_dir, modelname, "whole.npy")
    np.save(out_file, vr)
    out_file = os.path.join(out_dir, modelname, "whole_labels.npy")
    np.save(out_file, indices_to_one_hot(g-1)[0])
