import torch.nn as nn


# NOTE: this is not ranknet!!!!


class EmbeddingNet(nn.Module):
    def __init__(self, input_dim=65, out_dim=8):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.PReLU(),
            # nn.Linear(input_dim//2, input_dim//2),
            # nn.PReLU(),
            nn.Linear(input_dim//2, out_dim)
        )

    def forward(self, x):
        output = self.fc(x)
        return output


class TripletNet(nn.Module):
    def __init__(self, embeddingNet):
        super(TripletNet, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, i1, i2, i3):
        E1 = self.embeddingNet(i1)
        E2 = self.embeddingNet(i2)
        E3 = self.embeddingNet(i3)
        return E1, E2, E3


def get_model(args, device):
    # Model
    embeddingNet = EmbeddingNet(args.num_input, args.num_output)

    model = TripletNet(embeddingNet)
    # model = nn.DataParallel(model, device_ids=args.gpu_devices)
    model = model.to(device)

    return model
