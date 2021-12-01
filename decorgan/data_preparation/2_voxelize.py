import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("class_id", type=str, help="shapenet category id")
parser.add_argument("target_dir", type=str, default="./preprocessed")
FLAGS = parser.parse_args()

class_id = FLAGS.class_id
target_dir = os.path.join(FLAGS.target_dir, class_id)
if not os.path.exists(target_dir):
    print("ERROR: this dir does not exist: "+target_dir)
    exit()


def run(in_file, out_file):
    if os.path.exists(out_file):
        return
    print(in_file)

    maxx = 0.5
    maxy = 0.5
    maxz = 0.5
    minx = -0.5
    miny = -0.5
    minz = -0.5

    command = "./binvox -bb "+str(minx)+" "+str(miny)+" "+str(minz)+" "+str(maxx)+" "+str(maxy)+" "+str(maxz)+" "+" -d 512 -e "+in_file

    os.system(command)


for f_or_d in os.listdir(target_dir):
    if os.path.isdir(os.path.join(target_dir, f_or_d)):
        for f_or_d_2 in os.listdir(os.path.join(target_dir, f_or_d)):
            if os.path.isfile(os.path.join(target_dir, f_or_d, f_or_d_2)) and f_or_d_2 == "model.obj":
                this_name = os.path.join(target_dir, f_or_d, f_or_d_2)
                out_name = os.path.join(target_dir, f_or_d, "model.binvox")
                run(this_name, out_name)
            else:
                this_name = os.path.join(target_dir, f_or_d, f_or_d_2, "model.obj")
                if not os.path.exists(this_name):
                    print(f"couldnt run for non existing {this_name}")
                    continue
                out_name = os.path.join(target_dir, f_or_d, f_or_d_2, "model.binvox")
                run(this_name, out_name)
