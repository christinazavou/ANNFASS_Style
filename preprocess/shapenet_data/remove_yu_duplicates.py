# NOTE:
# einai poli paromoia ... opotan eite vro akrivos ta idia data to remove ite ta afiso de nomizo oti tha kanei polli diafora..

# ta funriture400 einai arketa diaforetika kai ta classes einai vasi texture details

# filenames = "filenames_cars.txt"
filenames = "filenames_chairs.txt"

remove = list()
with open(filenames, "r") as fin:
    for line in fin.readlines():
        remove.append(line.rstrip())
remove_set = set(remove)
print(f"remove {len(remove_set)} unique files")

# input_file = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/splits/car/content_car_train_2K.txt"
# output_file = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/splits/car/content_car_train_2Knoyu.txt"
input_file = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/splits/chair/content_chair_train.txt"
output_file = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/splits/chair/content_chair_trainnoyu.txt"
with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for in_line in fin.readlines():
        if in_line.rstrip() not in remove:
            fout.write(in_line)
# input_file = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/splits/car/style_car_32.txt"
# output_file = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/splits/car/style_car_32noyu.txt"
input_file = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/splits/chair/style_chair_64.txt"
output_file = "/media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan-logs/splits/chair/style_chair_64noyu.txt"
with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for in_line in fin.readlines():
        if in_line.rstrip() not in remove:
            fout.write(in_line)
# SOS: tora prepei na pintoso extra mesa sto style gia na einai 32