import torch.nn as nn

import MinkowskiEngine as ME

from models.model import Model


class ExampleNetwork(Model):

	def __init__(self, in_channels, out_channels, config, D):

		super(ExampleNetwork, self).__init__(in_channels, out_channels, config, D)

		self.network_initialization(in_channels, out_channels, config, D)

	def network_initialization(self, in_channels, out_channels, config, D):
		self.conv1 = nn.Sequential(
			ME.MinkowskiConvolution(
				in_channels=in_channels,
				out_channels=64,
				kernel_size=3,
				stride=2,
				dilation=1,
				has_bias=False,
				dimension=D),
			ME.MinkowskiBatchNorm(64),
			ME.MinkowskiReLU())

		self.conv2 = nn.Sequential(
			ME.MinkowskiConvolution(
				in_channels=64,
				out_channels=128,
				kernel_size=3,
        stride=2,
        dimension=D),
			ME.MinkowskiBatchNorm(128),
			ME.MinkowskiReLU())

		self.pooling = ME.MinkowskiGlobalPooling()
		self.linear = ME.MinkowskiLinear(128, out_channels)

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.pooling(out)

		return self.linear(out)