from unittest import TestCase
import torch
from MinkowskiEngine import SparseTensor

from models.modules.resnet_block import BottleneckBase


class BottleneckExp4(BottleneckBase):
    expansion = 4


class TestBottleneckBase(TestCase):
    def test_forward(self):
        samples = 100000
        coords = torch.randn(samples, 4)
        input = torch.randn(samples, 32)

        sinput = SparseTensor(input, coords)

        # todo: note that different inplanes, planes will fail .. i should add those tests
        btln = BottleneckBase(32, 32)
        btln.forward(sinput)
        btlne3 = BottleneckExp4(32, 8)
        btlne3.forward(sinput)

