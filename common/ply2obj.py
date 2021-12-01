import os
from argparse import ArgumentParser

from plyfile import PlyData


def convert(ply, obj):
    with open(obj, 'w') as f:
        f.write("# OBJ file\n")

        verteces = ply['vertex']

        for v in verteces:
            p = [v['x'], v['y'], v['z']]
            f.write("v %.6f %.6f %.6f \n" % tuple(p) )

        for v in verteces:
            n = [ v['nx'], v['ny'], v['nz'] ]
            f.write("vn %.6f %.6f %.6f\n" % tuple(n))

        if 'face' in ply:
            for i in ply['face']['vertex_indices']:
                f.write("f")
                for j in range(i.size):
                    ii = [ i[j]+1, i[j]+1, i[j]+1 ]
                    f.write(" %d/%d/%d" % tuple(ii) )
                f.write("\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ply_file", type=str, dest="ply_file")
    parser.add_argument("--ply_dir", type=str, dest="ply_dir")
    args = parser.parse_args()

    assert args.ply_file is not None or args.ply_dir is not None

    if args.ply_file is not None:
        obj_file = args.ply_file.replace(".ply", ".obj")
        convert(PlyData.read(args.ply_file), obj_file)
    else:
        for root, dirs, files in os.walk(args.ply_dir):
            for file in files:
                if file.endswith(".ply"):
                    ply_file = os.path.join(root, file)
                    obj_file = ply_file.replace(".ply", ".obj")
                    convert(PlyData.read(ply_file), obj_file)
