import os
import sys
sys.path.append(".")

import json
import torch
import torch.nn as nn
from typing import List, Optional
from torchinfo import summary
from utils.get_feat import load_and_resample_audio, compute_feat

public_device = "cuda" if torch.cuda.is_available() else "cpu"


class BiLSTMASR(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 512,
        num_layers: int = 4,
        proj_dim: int = 512,
        num_classes: int = 409,
        vocab_data: Optional[dict] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        self.vocab_data = vocab_data

        self.pre_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.post_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(-1).transpose(1, 2)
        x = self.pre_proj(x)
        x, _ = self.lstm(x)
        logits = self.post_proj(x)
        return logits.transpose(0, 1).log_softmax(dim=-1)

    def forward_wave(self, wave_filepath, need_sentence):
        samples, sr = load_and_resample_audio(wave_filepath)
        feats = compute_feat(samples, sample_rate=16000)

        feats = torch.from_numpy(feats).float().to(public_device)
        feats = feats.unsqueeze(0).unsqueeze(-1).transpose(1, 2)

        with torch.no_grad():
            logits = self.forward(feats)
            final_output = logits.squeeze(1)

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

    model = BiLSTMASR(
        input_dim=80,
        hidden_dim=512,
        num_layers=4,
        proj_dim=512,
        num_classes=vocab_data['vocab_size'],
        vocab_data=vocab_data
    ).to(public_device)

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