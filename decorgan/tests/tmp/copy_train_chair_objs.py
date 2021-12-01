import os
import shutil


train_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/splits/chair/content_chair_train.txt"
raw_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/03001627"
new_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/03001627_train_objs"
os.makedirs(new_dir)
with open(train_file, "r") as fin:
    for line in fin.readlines():
        model = line.rstrip()
        os.makedirs(os.path.join(new_dir, model))
        shutil.copy(
            os.path.join(raw_dir, model, "model.obj"),
            os.path.join(new_dir, model, "model.obj")
        )

