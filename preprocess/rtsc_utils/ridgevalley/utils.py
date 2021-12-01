import os
import numpy as np


def txt_to_ply(ridge_or_valley_txt, ridge_or_valley_ply):
    ridge_or_valley_vertices = np.loadtxt(ridge_or_valley_txt)
    assert ridge_or_valley_vertices.shape[0] % 2 == 0

    with open(ridge_or_valley_ply, "w") as fout:
        header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
element edge {} 
property int vertex1                  
property int vertex2                  
end_header""".format(ridge_or_valley_vertices.shape[0], int(ridge_or_valley_vertices.shape[0]/2))
        fout.write(header+"\n")
        for ridge_vertex in ridge_or_valley_vertices:
            fout.write("{} {} {}\n".format(ridge_vertex[0], ridge_vertex[1], ridge_vertex[2]))
        for idx1 in range(0, ridge_or_valley_vertices.shape[0], 2):
            fout.write("{} {}\n".format(idx1, idx1+1))


def create_ply_files():
    ridge_txt_file = os.path.join(obj_dir, building, "{}.obj.ridge.txt".format(building))
    if os.path.exists(ridge_txt_file):
        ridge_ply_file = os.path.join(obj_dir, building, "{}.ridge.ply".format(building))
        txt_to_ply(ridge_txt_file, ridge_ply_file)
    valley_txt_file = os.path.join(obj_dir, building, "{}.obj.valley.txt".format(building))
    if os.path.exists(valley_txt_file):
        valley_ply_file = os.path.join(obj_dir, building, "{}.valley.ply".format(building))
        txt_to_ply(valley_txt_file, valley_ply_file)


if __name__ == '__main__':
    obj_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_march/normalizedObj"
    for root, dirs, files in os.walk(obj_dir):
        for building in dirs:
            if building == "textures":
                continue
            print("run for {}".format(building))
            create_ply_files()
