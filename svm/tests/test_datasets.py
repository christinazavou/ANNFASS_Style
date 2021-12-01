from unittest import TestCase

from sklearn_impl.datasets import StylisticComponentPlyWithCurvaturesDataset, \
    StylisticComponentPlyWithCurvaturesDataLoader

rootdir = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs"
datadir = rootdir+"/annfass_splits_march/ply15Kwcpercomponent/fold0/split_test/test.txt"


class TestStylisticComponentPlyWithCurvaturesDataset(TestCase):
    def test_me(self):
        dataset = StylisticComponentPlyWithCurvaturesDataset(datadir)
        res = iter(dataset).__next__()
        print(res)


class TestStylisticComponentPlyWithCurvaturesDataLoader(TestCase):
    def test_me(self):
        dataloader = StylisticComponentPlyWithCurvaturesDataLoader(datadir, batch_size=16, shuffle=True, num_workers=4)
        res = iter(dataloader).__next__()
        print(res[0].shape, res[1].shape)

        dataloader = StylisticComponentPlyWithCurvaturesDataLoader(datadir, batch_size=-1, shuffle=True, num_workers=4)
        res = iter(dataloader).__next__()
        print(res[0].shape, res[1].shape)
        assert res[1].max() == 7

        print(res[2].shape)   # todo: remove names