import os

inp_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/chair/style_chair_64.txt"
out_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/HU_YU_LUN_BUILDNET/splits/03001627_train_objs_norm_and_random_rot/style_chair_random_rot_64.txt"
data_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/HU_YU_LUN_BUILDNET/preprocessed_data/03001627_train_objs_norm_and_random_rot"

files = {}
for model_name_rot in os.listdir(data_dir):
    try:
        model_name, rot = model_name_rot.split("_rot")
    except:
        model_name, rot = model_name_rot, "0"
    files.setdefault(model_name, [])
    files[model_name] += [model_name_rot]

with open(inp_file, "r") as fin:
    lines = fin.readlines()

with open(out_file, "w") as fout:
    for line in lines:
        line = line.rstrip()
        for file in files[line]:
            fout.write(file+"\n")

