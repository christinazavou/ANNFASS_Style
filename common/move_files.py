import os
import shutil
from argparse import ArgumentParser

from utils import str2bool

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--in_dir", type=str, dest="in_dir")
    parser.add_argument("--out_dir", type=str, dest="out_dir")
    parser.add_argument("--ext", type=str, dest="ext", default=".obj")
    parser.add_argument("--exclude", type=str, dest="exclude", default="refinedTextures")
    parser.add_argument("--rename", type=str, dest="rename", default="whole/model")
    parser.add_argument("--copy", type=str2bool, dest="copy", default=False)
    args = parser.parse_args()

    assert args.in_dir is not None and args.out_dir is not None

    for root, dirs, files in os.walk(args.in_dir):
        for file in files:
            if file.endswith(args.ext):
                if args.exclude is not None and args.exclude != "" and args.exclude in file:
                    continue
                old_file = os.path.join(root, file)
                old_filename, extension = os.path.splitext(old_file)
                new_file = old_filename.replace(args.in_dir, args.out_dir)
                if args.rename != "":
                    new_file = os.path.join(new_file, args.rename+extension)
                else:
                    new_file = f"{new_file}{extension}"
                os.makedirs(os.path.dirname(new_file), exist_ok=True)
                if args.copy:
                    shutil.copy(old_file, new_file)
                else:
                    shutil.move(old_file, new_file)
