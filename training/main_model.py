import torch
import torch.nn as nn
from torchinfo import summary

class ResidualTDNNBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm1d(dim)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        x = x + residual
        return x

class TDNNASR(nn.Module):
    def __init__(self, input_dim=560, hidden_dim=512, num_classes=40, num_blocks=2):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )

        self.blocks = nn.Sequential(
            *[ResidualTDNNBlock(hidden_dim) for _ in range(num_blocks)]
        )

        self.output_layer = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

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
        hidden_dim=256,
        num_classes=409,
        num_blocks=2
    )

    # shape: (batch_size, seq_len, input_dim)
    summary(
        model,
        input_size=(1, 800, 560),
        device="cpu",
        dtypes=[torch.float32],
        col_names=["input_size", "output_size", "num_params", "mult_adds"]
    )