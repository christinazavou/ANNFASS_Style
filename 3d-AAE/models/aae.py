import torch
import torch.nn as nn

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['G']['use_bias']
        self.relu_slope = config['model']['G']['relu_slope']
        self.num_output = config['num_features'] if 'num_features' in config else 3
        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=2048 * self.num_output, bias=self.use_bias),
        )

    def forward(self, input):
        output = self.model(input.squeeze())
        output = output.view(-1, self.num_output, 2048)
        return output


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['D']['use_bias']
        self.relu_slope = config['model']['D']['relu_slope']
        self.dropout = config['model']['D']['dropout']

        self.model = nn.Sequential(

            nn.Linear(self.z_size, 512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(512, 128, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(64, 1, bias=True)
        )

    def forward(self, x):  # [batch, z_size]
        logit = self.model(x)  # [batch, 1]
        return logit


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']
        self.num_input = config['num_features'] if 'num_features' in config else 3

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.num_input, out_channels=64, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
                      bias=self.use_bias),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True)
        )

        self.mu_layer = nn.Linear(256, self.z_size, bias=True)
        self.std_layer = nn.Linear(256, self.z_size, bias=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.conv(x)  # [batch, out_channels (512), Length (num of points)]
        output2 = output.max(dim=2)[0]  # [batch, out_channels]
        logit = self.fc(output2)  # [batch, 256]
        mu = self.mu_layer(logit)  # [batch, z_size]
        logvar = self.std_layer(logit)  # [batch, z_size]
        z = self.reparameterize(mu, logvar)  # [batch, z_size]
        return z, mu, logvar
