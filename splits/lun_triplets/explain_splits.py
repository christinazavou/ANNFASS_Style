import os


splits = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/compressed_files/Data-all/Data/furniture/response/splits3"
for split in range(10):
    train_file = os.path.join(splits, f"train_triplets_{split}.txt")
    train_shapes = set()
    with open(train_file, "r") as fin:
        for line in fin.readlines():
            triplet = line.rstrip().split(" ")
            for shape in triplet:
                train_shapes.add(shape)
    print(len(train_shapes) / 278)
