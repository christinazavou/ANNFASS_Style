import os
import shlex
import subprocess

import numpy as np

"""
    This file contains some helper functions for geometry processing
    Author: Kaichun Mo
    Edit by: Marios Loizou, Christina Zavou
"""


_THRESHOLD_TOL_32 = 2.0 * np.finfo(np.float32).eps
_THRESHOLD_TOL_64 = 2.0 * np.finfo(np.float32).eps

THEA_TOOLS_DIR = ""
nTries = 10


class ShapeFeatureExporterException(Exception):
    def __init__(self, message):
        super().__init__(message)


def ShapeFeatureExporter(inputPath, args, featureExporter='ShapeFeatureExporter', override=True, remove=True):
    """Use ShapeFeatureExporter for Poisson Sampling and exporting features"""

    _, file_ext = os.path.splitext(inputPath)

    tmpPath = inputPath.replace(file_ext, f"_no_mtl_{file_ext[1:]}.txt")
    outputPath = tmpPath + ".txt"

    # sometimes trimesh can read obj with mtllib
    with open(inputPath, "r") as fin:
        contents = fin.read()
    with open(tmpPath, "w") as fout:
        fout.write(contents.replace("mtllib", "#mtllib"))

    result = (False, )
    for i in range(nTries):

        if override or not os.path.exists(outputPath):
            sfeCMD = featureExporter + ' ' + args + " --input-shape=\"" + tmpPath + '"'
            print(sfeCMD)
            proc = subprocess.Popen(shlex.split(sfeCMD), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
        else:
            print("will process existing mesh samples")

        # Get samples and features
        if os.path.isfile(outputPath):
            faceIdx = []; pc = []; curvature = []; principalDirections = []
            with open(outputPath, 'r') as fin:
                if "--export-point-samples" in args:
                    line = fin.readline().rstrip().split(' ')
                    assert len(line) == 1
                    assert line[0].isdigit(), ShapeFeatureExporterException("no points sampled")
                    nPoints = int(line[0])
                    for _ in range(nPoints):
                        rec = fin.readline().rstrip().split(' ')
                        assert(len(rec) == 7)
                        pc.append([float(p) for p in rec[:-1]])
                        faceIdx.append(int(rec[-1]))
                if "--export-curvatures" in args:
                    curvInfo = fin.readline().rstrip().split(' ')
                    assert (len(curvInfo) == 2)
                    assert(int(curvInfo[1]) == 64)
                    for _ in range(int(curvInfo[0])):
                        rec = fin.readline().rstrip().split(' ')
                        assert(len(rec) == int(curvInfo[1]))
                        curvature.append([float(c) if not np.isnan(float(c)) else 0 for c in rec])
                    if len(pc):
                        assert(len(curvature) == len(pc))
                if "--export-principal-directions" in args:
                    principalInfo = fin.readline().rstrip().split(' ')
                    assert(len(principalInfo) == 2)
                    assert (int(principalInfo[1]) == 24)
                    for _ in range(int(principalInfo[0])):
                        rec = fin.readline().rstrip().split(' ')
                        assert (len(rec) == int(principalInfo[1]))
                        principalDirections.append([float(c) if not np.isnan(float(c)) else 0 for c in rec])
                    if len(pc):
                        assert(len(principalDirections) == len(pc))
                    elif len(curvature):
                        assert (len(principalDirections) == len(curvature))

            if len(pc):
                assert(len(faceIdx) == len(pc))

            pc = np.array(pc, dtype=np.float32)
            result = (pc[:, 0:3] if len(pc) else None,
                      pc[:, 3:6] if len(pc) else None,
                      faceIdx if len(faceIdx) else None,
                      np.array(curvature, dtype=np.float32) if len(curvature) else None,
                      np.array(principalDirections, dtype=np.float32) if len(principalDirections) else None,
                      True)

            os.remove(tmpPath)

            if remove:
                os.remove(outputPath)

            break

    return result


if __name__ == '__main__':
    root_dir = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_march/normalizedObj"
    obj_file = root_dir+"/15_Paphos_Gate/15_Paphos_Gate.obj"
    root_dir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/preprocess/mesh_sampling"
    sfe_file = root_dir+"/shapefeatureexporter/build/ShapeFeatureExporter"
    ShapeFeatureExporter(obj_file,
                         "--do-not-rescale-shape --export-point-samples --num-point-samples 5000",
                         featureExporter=sfe_file,
                         override=True,
                         remove=False)
