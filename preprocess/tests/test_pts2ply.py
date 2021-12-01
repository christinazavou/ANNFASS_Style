from unittest import TestCase

import numpy as np

from preprocess.point_cloud_generation.pts2ply import get_data


class Test(TestCase):
    def test_me(self):
        ofile = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_march/normalizedObj/01_Cathedral_of_Holy_Wisdom/01_Cathedral_of_Holy_Wisdom.obj"
        ffile = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_march/samplePoints/faces_10000K/01_Cathedral_of_Holy_Wisdom.txt"
        pfile = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_march/samplePoints/point_cloud_10000K/01_Cathedral_of_Holy_Wisdom.pts"
        rnvfile = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_march/samplePoints/ridge_or_valley_10000K/01_Cathedral_of_Holy_Wisdom.txt"
        faces, obj, pts, rnvs = get_data(100000, ofile, ffile, pfile, rnvfile)
        indices = np.where(rnvs == 1)
        outfile = "/home/graphicslab/Documents/01_Cathedral_of_Holy_Wisdom/01_Cathedral_of_Holy_Wisdom.ridge.pts"
        np.savetxt(outfile, pts[indices])
        indices = np.where(rnvs == -1)
        outfile = "/home/graphicslab/Documents/01_Cathedral_of_Holy_Wisdom/01_Cathedral_of_Holy_Wisdom.valley.pts"
        np.savetxt(outfile, pts[indices])

    def test_prev(self):
        ofile = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_march/normalizedObj/01_Cathedral_of_Holy_Wisdom/01_Cathedral_of_Holy_Wisdom.obj"
        ffile = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_march/samplePoints/faces_10000K/01_Cathedral_of_Holy_Wisdom.txt"
        pfile = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_march/samplePoints/point_cloud_10000K/01_Cathedral_of_Holy_Wisdom.pts"
        rnvfile = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_march/samplePoints/ridge_valley_10000K/01_Cathedral_of_Holy_Wisdom.txt"
        faces, obj, pts, rnvs = get_data(100000, ofile, ffile, pfile, rnvfile)
        indices = np.where(rnvs == 1)
        outfile = "/home/graphicslab/Documents/01_Cathedral_of_Holy_Wisdom/01_Cathedral_of_Holy_Wisdom.ridgevalley.pts"
        np.savetxt(outfile, pts[indices])


