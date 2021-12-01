
import torch

with torch.no_grad():
    x = torch.randn((1, 10))
    x1 = torch.nn.functional.normalize(x, p=2, dim=1)
    x2 = torch.nn.GroupNorm(1, 10)(x)

    print(x)

    print(x1, torch.sum(x1), torch.norm(x1))

    print(x2, torch.sum(x2), torch.norm(x2))
