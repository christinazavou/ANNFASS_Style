import os
import random
import shutil


class_num = "02958343"
# class_num = "03001627"
class_id = "car"
# class_id = "chair"

inp_dir = f"/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/data/ShapeNet/ShapeNetCorev1/{class_num}"
# out_dir = f"/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/{class_num}_train_objs"
# out_dir = f"/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/{class_num}_rest_objs"
out_dir = f"/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/{class_num}_test_objs"
# out_dir = f"/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/{class_num}_style_objs"


# train_data = f"/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/splits/{class_id}/content_{class_id}_train_2K.txt"
# test_data = f"/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/splits/{class_id}/content_{class_id}_test.txt"
test_data = f"/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/splits/car/content_car_test_470.txt"
# style_data = f"/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/splits/{class_id}/style_{class_id}_32.txt"
keep = set()
# with open(train_data, "r") as fin:
#     for line in fin.readlines():
#         keep.add(line.rstrip())
# with open(style_data, "r") as fin:
#     for line in fin.readlines():
#         keep.add(line.rstrip())
with open(test_data, "r") as fin:
    for line in fin.readlines():
        keep.add(line.rstrip())
print(f"Will keep {len(keep)} data")

# all_data = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/splits/car/content_car_all.txt"
# with open(all_data, "r") as fin:
#     for line in fin.readlines():
#         keep.add(line.rstrip())
# with open(train_data, "r") as fin:
#     for line in fin.readlines():
#         keep.remove(line.rstrip())
#
os.makedirs(out_dir, exist_ok=True)
for model in keep:
    if os.path.exists(os.path.join(inp_dir, model)):
        shutil.copytree(os.path.join(inp_dir, model),
                        os.path.join(out_dir, model))

