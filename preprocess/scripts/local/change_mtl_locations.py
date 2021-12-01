import argparse
import os


def change_mtl_location():
    with open(mtl_file, "r") as fin:
        lines = fin.read()
        lines = lines.replace(args.remove, args.add)
    with open(mtl_file, "w") as fout:
        fout.write(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', type=str, required=True)
    parser.add_argument('--depth', type=str, default='building')
    parser.add_argument('--remove', type=str, required=True)
    parser.add_argument('--add', type=str, required=True)
    args = parser.parse_args()

    for subdir in os.listdir(args.dir_in):
        if args.depth == 'building':
            mtl_file = os.path.join(args.dir_in, subdir, f"{subdir}.mtl")
            change_mtl_location()
        else:
            assert args.depth == 'component'
            for subsubdir in os.listdir(os.path.join(args.dir_in, subdir)):
                mtl_file = os.path.join(args.dir_in, subdir, subsubdir, "model.mtl")
                change_mtl_location()
