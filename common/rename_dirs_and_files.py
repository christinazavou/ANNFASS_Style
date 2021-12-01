import os
import shutil
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--in_dir", type=str, dest="in_dir")
    parser.add_argument("--mode", type=str, dest="mode", default="dir")
    parser.add_argument("--remove", type=str, dest="remove", default=" ")
    parser.add_argument("--replace", type=str, dest="replace", default="")
    args = parser.parse_args()

    assert args.in_dir is not None

    for root, dirs, files in os.walk(args.in_dir):
        for directory in dirs:
            if args.mode == "dir" and args.remove in directory:
                shutil.move(os.path.join(root, directory), os.path.join(root, directory.replace(args.remove, args.replace)))

    for root, dirs, files in os.walk(args.in_dir):
        for file in files:
            if args.mode == "file" and args.remove in file:
                shutil.move(os.path.join(root, file), os.path.join(root, file.replace(args.remove, args.replace)))
