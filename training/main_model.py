import torch
import torch.nn as nn
from typing import List
from torchinfo import summary


class ResidualTDNNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

        if in_dim != out_dim:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_dim)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual
        return x


class TDNNASR(nn.Module):
    def __init__(
            self,
            input_dim: int = 560,
            block_dims: List[int] = [512, 512, 512],
            proj_dim: int = 128,
            num_classes: int = 409
    ):
        super().__init__()

        first_dim = block_dims[0]
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, first_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(first_dim),
            nn.ReLU()
        )

        layers = []
        in_dim = first_dim
        for out_dim in block_dims:
            layers.append(ResidualTDNNBlock(in_dim, out_dim))
            in_dim = out_dim

        self.blocks = nn.Sequential(*layers)

        final_dim = block_dims[-1]
        self.output_layer = nn.Sequential(
            nn.Conv1d(final_dim, proj_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(proj_dim, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        x = x.transpose(1, 2)
        return x


if __name__ == "__main__":
    model = TDNNASR(
        input_dim=560,
        block_dims=[512, 512, 512],
        proj_dim=128,
        num_classes=409
    )

    summary(
        model,
        input_size=(1, 800, 560),
        device="cpu",
        dtypes=[torch.float32],
        col_names=["input_size", "output_size", "num_params", "mult_adds"]
    )