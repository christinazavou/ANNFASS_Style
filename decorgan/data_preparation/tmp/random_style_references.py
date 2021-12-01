
import random

in_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/chair/content_chair_train.txt"

with open(in_file, "r") as fin:
    files = fin.readlines()

files = files[100:]
random.shuffle(files)

out_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/chair/random_style_chair_references_64.txt"

with open(out_file, "w") as fout:
    files = files[0:64]
    fout.writelines(files)
