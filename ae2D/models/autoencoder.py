import torch
import torch.nn as nn


# class ConvAutoencoder(nn.Module):
#     def __init__(self, in_channels=3):
#         super(ConvAutoencoder, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(16, 4, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             # nn.BatchNorm2d()
#         )
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(4, 16, 2, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, in_channels, 2, stride=2),
#             # nn.ReLU()
#         )
#
#     def forward(self, x):
#         # print("input x ", x.min(), x.max())
#         # print("input x ", x.size())
#         x = self.encoder(x)
#         # print("encoded x ", x.size())
#         x = self.decoder(x)
#         x = torch.sigmoid(x)
#         # print("decoded x ", x.size())
#         return x


class ConvAutoencoder(nn.Module):

    def __init__(self, in_channels=3, z_dim=16, kernel_size=3, pool_size=2, batch_norm=False):
        super(ConvAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.encoder = nn.Sequential(
                nn.Conv2d(self.in_channels, self.z_dim*4, self.kernel_size, padding=1),
                nn.BatchNorm2d(self.z_dim*4),
                nn.ReLU(),
                nn.MaxPool2d(self.pool_size, self.pool_size),
                nn.Conv2d(self.z_dim*4, self.z_dim*2, self.kernel_size, padding=1),
                nn.BatchNorm2d(self.z_dim*2),
                nn.ReLU(),
                nn.MaxPool2d(self.pool_size, self.pool_size),
                nn.Conv2d(self.z_dim*2, self.z_dim, self.kernel_size, padding=1),
                nn.BatchNorm2d(self.z_dim),
                nn.ReLU(),
                nn.MaxPool2d(self.pool_size, self.pool_size),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(self.in_channels, self.z_dim*4, self.kernel_size, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(self.pool_size, self.pool_size),
                nn.Conv2d(self.z_dim*4, self.z_dim*2, self.kernel_size, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(self.pool_size, self.pool_size),
                nn.Conv2d(self.z_dim*2, self.z_dim, self.kernel_size, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(self.pool_size, self.pool_size),
            )

        if self.batch_norm:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(self.z_dim, self.z_dim*2, 2, stride=2),
                nn.BatchNorm2d(self.z_dim*2),
                nn.ReLU(),
                nn.ConvTranspose2d(self.z_dim*2, self.z_dim*4, 2, stride=2),
                nn.BatchNorm2d(self.z_dim*4),
                nn.ReLU(),
                nn.ConvTranspose2d(self.z_dim*4, self.in_channels, 2, stride=2),
                nn.BatchNorm2d(self.in_channels),
            )
        else:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(self.z_dim, self.z_dim*2, 2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(self.z_dim*2, self.z_dim*4, 2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(self.z_dim*4, self.in_channels, 2, stride=2),
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x
