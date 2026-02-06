import os
import sys
sys.path.append(".")

import json
import torch
import torch.nn as nn
import numpy as np
from typing import List
from torchinfo import summary
from utils.get_feat import load_and_resample_audio,compute_feat

public_device = "cuda" if torch.cuda.is_available() else "cpu"

class ResidualTDNNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dilation: int = 1, stride: int = 1):
        super().__init__()
        padding = dilation
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_dim, out_dim,
            kernel_size=(3, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1)
        )

        if in_dim != out_dim or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1), stride=(stride, 1)),
                nn.BatchNorm2d(out_dim)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class TDNNASR(nn.Module):
    def __init__(
            self,
            input_dim: int = 560,
            block_dims: List[int] = None,
            dilations: List[int] = None,
            strides: List[int] = None,
            proj_dim: int = 128,
            num_classes: int = 409,
            vocab_data: dict = None
    ):
        super().__init__()

        assert None not in [block_dims, dilations, strides], "block_dims and dilations must not be None"

        assert len(block_dims) == len(dilations) == len(strides), "block_dims, dilations and strides must have same length"

        self.block_dims = block_dims
        self.dilations = dilations
        self.strides = strides

        self.vocab_data = vocab_data
        self.reduction = int(np.prod(strides))

        first_dim = block_dims[0]
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, first_dim, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(first_dim),
            nn.ReLU()
        )

        layers = []
        in_dim = first_dim
        for i, out_dim in enumerate(block_dims):
            layers.append(ResidualTDNNBlock(in_dim, out_dim, dilation=dilations[i], stride=strides[i]))
            in_dim = out_dim

        self.blocks = nn.Sequential(*layers)

        final_dim = block_dims[-1]
        self.output_layer = nn.Sequential(
            nn.Conv2d(final_dim, proj_dim, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(proj_dim, num_classes, kernel_size=(1, 1))
        )

        self.receptive_field = self.compute_receptive_field_frames()

    def compute_receptive_field_frames(self):
        if self.block_dims is None or self.dilations is None or self.strides is None:
            raise ValueError("block_dims, dilations, and strides must be provided")
        receptive_field = 1
        total_stride = 1
        for d, s in zip(self.dilations, self.strides):
            receptive_field += 2 * d * total_stride
            total_stride *= s
        return receptive_field

    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x

    def forward_wave(self, wave_filepath, need_sentence):
        samples, sr = load_and_resample_audio(wave_filepath)
        feats = compute_feat(samples, sample_rate=16000)

        feats = torch.from_numpy(feats).float().to(public_device)
        feats = feats.unsqueeze(0).unsqueeze(-1).transpose(1, 2)

        with torch.no_grad():
            batch_results = self.forward(feats)
            final_output = batch_results.squeeze(0).squeeze(-1).transpose(0, 1)

        sentence = None
        if need_sentence:
            assert self.vocab_data is not None, "vocab_data must not be None"
            predicted_ids = torch.argmax(final_output, dim=-1).cpu().numpy()
            tokens = []
            prev_token = None
            for idx in predicted_ids:
                if idx == 0:
                    prev_token = None
                    continue
                token = self.vocab_data["id_to_token"].get(str(idx), "")
                if token != prev_token:
                    tokens.append(token)
                    prev_token = token
            sentence = "".join(tokens)

        return final_output, sentence

if __name__ == "__main__":
    vocab_data = json.load(open(r"basic_data/vocab_data.json"))

    model = TDNNASR(
        input_dim=80,
        block_dims=[512] * 36,
        dilations=[1,2,4] * 12,
        strides=[2] * 2 + [1] * 34,
        proj_dim=512,
        num_classes=vocab_data['vocab_size'],
        vocab_data=vocab_data
    ).to(public_device)

    print("the receptive field of model is {}".format(model.receptive_field))

    if os.path.exists("weights/best.pth"):
        model.load_state_dict(torch.load("weights/best.pth", map_location=public_device, weights_only=True))
        print("best.pth was found and loaded")

    summary(
        model,
        input_size=(1, 80, 512, 1),
        device=public_device,
        dtypes=[torch.float32],
        col_names=["input_size", "output_size", "num_params", "mult_adds"]
    )

    final_output, sentence = model.forward_wave(wave_filepath=r"examples/zh.wav", need_sentence=True)
    print(final_output.size())
    print(sentence)
