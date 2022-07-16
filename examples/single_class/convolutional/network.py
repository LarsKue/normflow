
import torch
import torch.nn as nn


import examples.single_class.convolutional.settings as s


class ConvolutionalAutoRegressiveNetwork(nn.Module):
    def __init__(self, in_channels, in_features, num_params=2, out_channels=None):
        super().__init__()

        self.in_channels = in_channels
        self.in_features = in_features

        self.num_params = num_params

        if out_channels is None:
            out_channels = 3 - in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            # in_channels x 128 x 128
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1, stride=1),
            nn.PReLU(num_parameters=16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
            nn.PReLU(num_parameters=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
            nn.PReLU(num_parameters=64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1),
            nn.PReLU(num_parameters=32),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1),
            nn.PReLU(num_parameters=16),
            nn.Conv2d(in_channels=16, out_channels=num_params * self.out_channels, kernel_size=1, stride=1)
        )

        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x1: torch.Tensor, *, condition: torch.Tensor = None) -> torch.Tensor:
        x = self.conv.forward(x1)
        parts = torch.tensor_split(x, sections=self.num_params, dim=1)
        params = torch.stack(parts, dim=-1)

        return params


class ParamsNetwork(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, params: int = 2):
        print(in_channels, out_channels)
        super().__init__()

        # TODO:
        #  more affine layers per SCC Block
        #  Cosine Annealing LR
        #  bigger kernels, padding
        #  more parameters
        #  ActNorm --> larger LR
        #  Finally, GLOW (at some point)

        # use padding to avoid shrinking the input whilst still using kernel sizes > 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4 * in_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4 * in_channels, out_channels=8 * in_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            # nn.Conv2d(in_channels=8 * in_channels, out_channels=16 * in_channels, kernel_size=5, padding=2),
            # nn.LeakyReLU(),
            # nn.Conv2d(in_channels=16 * in_channels, out_channels=8 * in_channels, kernel_size=5, padding=2),
            # nn.LeakyReLU(),
            nn.Conv2d(in_channels=8 * in_channels, out_channels=4 * in_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4 * in_channels, out_channels=params * out_channels, kernel_size=3, padding=1)
        ).to(s.DEVICE)

    def forward(self, x1: torch.Tensor, *, condition: torch.Tensor = None) -> torch.Tensor:
        # print(f"Input Shape: {x1.shape}")
        z = self.conv.forward(x1)
        # print(f"Output Shape: {z.shape}")
        return z
