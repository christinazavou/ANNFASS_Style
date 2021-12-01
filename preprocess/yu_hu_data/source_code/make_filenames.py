import os

# buildings_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj"
# with open("filenames_buildings.txt", "w") as fout:
#     for obj_dir in os.listdir(buildings_dir):
#         if os.path.exists(os.path.join(buildings_dir, obj_dir, "{}.obj".format(obj_dir))):
#             fout.write(obj_dir+"\n")
#
# with open("filenames_chairs.txt", "w") as fout:
#     for obj_dir in os.listdir("/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/03001627_train_objs"):
#         fout.write(obj_dir+"\n")


with open("filenames_cars.txt", "w") as fout:
    for obj_dir in os.listdir("/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/raw_shapenet/02958343_train_objs"):
        fout.write(obj_dir+"\n")
