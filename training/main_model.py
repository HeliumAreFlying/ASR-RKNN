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
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        if in_dim != out_dim or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=stride),
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
            block_dims: List[int] = None,
            dilations: List[int] = None,
            strides: List[int] = None,
            proj_dim: int = 128,
            num_classes: int = 409,
            vocab_data: dict = None,
            max_window_size: int = 512,
            max_window_shift: int = 384
    ):
        super().__init__()

        assert None not in [block_dims, dilations, strides], "block_dims and dilations must not be None"

        assert len(block_dims) == len(dilations) == len(strides), "block_dims, dilations and strides must have same length"

        self.vocab_data = vocab_data
        self.reduction = int(np.prod(strides))
        self.max_window_size = max_window_size
        self.max_window_shift = max_window_shift

        first_dim = block_dims[0]
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, first_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(first_dim),
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
            nn.Conv1d(final_dim, proj_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(proj_dim, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x

    def forward_wave(self, wave_filepath, need_sentence):
        samples, sr = load_and_resample_audio(wave_filepath)
        feats = compute_feat(samples, sample_rate=16000, window_size=7, window_shift=1)
        feats = torch.from_numpy(feats).float().to(public_device)
        total_frames = feats.size(0)
        num_classes = self.output_layer[-1].out_channels

        total_out_frames = (total_frames + self.reduction - 1) // self.reduction
        final_output = torch.zeros(total_out_frames, num_classes, device=public_device)

        start_idx = 0
        batch_inputs = []
        batch_starts = []

        with torch.no_grad():
            while start_idx < total_frames:
                end_idx = min(start_idx + self.max_window_size, total_frames)
                chunk = feats[start_idx:end_idx]
                current_length = chunk.size(0)

                if current_length < self.max_window_size:
                    pad_size = self.max_window_size - current_length
                    pad_tensor = torch.zeros(pad_size, feats.size(1), device=public_device)
                    chunk = torch.cat([chunk, pad_tensor], dim=0)

                batch_inputs.append(chunk)
                batch_starts.append(start_idx)
                start_idx += self.max_window_shift

            batch_tensor = torch.stack(batch_inputs, dim=0)
            batch_tensor = batch_tensor.transpose(1,2)
            batch_results = self.forward(batch_tensor)
            batch_results = batch_results.transpose(1, 2)

            for i, start_pos in enumerate(batch_starts):
                result_chunk = batch_results[i]

                actual_chunk_len = min(self.max_window_size, total_frames - start_pos)
                output_chunk_len = actual_chunk_len // self.reduction

                out_start = start_pos // self.reduction
                out_end = out_start + output_chunk_len

                if out_end > final_output.size(0):
                    out_end = final_output.size(0)
                    output_chunk_len = out_end - out_start

                if output_chunk_len > 0:
                    final_output[out_start:out_end] = result_chunk[:output_chunk_len]

        sentence = None
        if need_sentence:
            assert self.vocab_data is not None, "vocab_data must not be None"

            logits_for_decode = final_output.clone()
            logits_for_decode[:, 0] = -128

            predicted_ids = torch.argmax(logits_for_decode, dim=-1).cpu().numpy()

            sentence = ""
            prev_token = None
            for idx in predicted_ids:
                if idx == 0:
                    token = ""
                else:
                    token = self.vocab_data["id_to_token"].get(str(idx + 1), "")

                if token != prev_token:
                    sentence += token
                    prev_token = token

        return final_output, sentence

if __name__ == "__main__":
    vocab_data = json.load(open(r"basic_data/vocab_data.json"))

    model = TDNNASR(
        input_dim=560,
        block_dims=[512] * 9,
        dilations=[1, 2, 4, 2, 1, 2, 4, 2, 1],
        strides=[1, 1, 1, 1, 1, 1, 1, 1, 2],
        proj_dim=128,
        num_classes=vocab_data['vocab_size'] + 1,
        vocab_data=vocab_data,
        max_window_size=512,
        max_window_shift=384
    ).to(public_device)

    summary(
        model,
        input_size=(1, 560, 512),
        device=public_device,
        dtypes=[torch.float32],
        col_names=["input_size", "output_size", "num_params", "mult_adds"]
    )

    final_output, sentence = model.forward_wave(wave_filepath=r"examples/en.wav", need_sentence=True)
    print(final_output.size())
    print(len(sentence),sentence[:16])