import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from main_model import TDNNASR, public_device
from utils.get_feat import load_and_resample_audio, compute_feat

TRAINING_DATA_ROOT = "D:/zhvoice"
BATCH_SIZE = 8
LEARNING_RATE = 2.5e-4
NUM_EPOCHS = 100
VAL_SPLIT = 0.1
SAVE_DIR = "weights"
LOG_STEP = 10
DEVICE = public_device

os.makedirs(SAVE_DIR, exist_ok=True)


class AudioDataset(Dataset):
    def __init__(self, meta_file, vocab_file):
        with open(meta_file, 'r', encoding='utf-8') as f:
            self.meta_data = json.load(f)
        self.items = list(self.meta_data.items())

        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        self.token_to_id = self.vocab['token_to_id']

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel_path, text = self.items[idx]
        wav_path = os.path.join(TRAINING_DATA_ROOT, rel_path)

        try:
            samples, sr = load_and_resample_audio(wav_path)
            feats = compute_feat(samples, sample_rate=16000, window_size=7, window_shift=1)
            feats = torch.from_numpy(feats).float()

            label_ids = []
            for char in text:
                if char in self.token_to_id:
                    token_id = self.token_to_id[char]
                    label_ids.append(token_id + 1)

            label_tensor = torch.tensor(label_ids, dtype=torch.long)
            return feats, label_tensor

        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return None, None


def collate_fn(batch):
    batch = [(f, l) for f, l in batch if f is not None]
    if not batch:
        return None, None
    feats, labels = zip(*batch)
    padded_feats = pad_sequence(feats, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return padded_feats, padded_labels


def train():
    dataset = AudioDataset('basic_data/clean_meta_data.json', 'basic_data/vocab_data.json')

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    vocab_size = dataset.vocab['vocab_size']
    num_classes = vocab_size + 1

    model = TDNNASR(
        input_dim=560,
        block_dims=[512] * 9,
        dilations=[1, 2, 4, 2, 1, 2, 4, 2, 1],
        strides=[1, 1, 1, 1, 1, 1, 1, 1, 2],
        proj_dim=128,
        num_classes=num_classes,
        vocab_data=dataset.vocab
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = GradScaler()

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.transpose(1, 2).unsqueeze(-1)

            optimizer.zero_grad()

            with autocast():
                out = model(x)
                out = out.squeeze(-1).transpose(1, 2)
                log_probs = torch.nn.functional.log_softmax(out, dim=-1).transpose(0, 1)

                input_lengths = torch.full((x.size(0),), out.size(1), dtype=torch.long).to(DEVICE)
                target_lengths = (y != 0).sum(dim=1).clamp(min=1)

                loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)(log_probs, y, input_lengths,
                                                                                 target_lengths)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if (batch_idx + 1) % LOG_STEP == 0:
                print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{batch_idx + 1}], Train Loss: {loss.item():.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                if x is None:
                    continue
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = x.transpose(1, 2).unsqueeze(-1)

                out = model(x)
                out = out.squeeze(-1).transpose(1, 2)
                log_probs = torch.nn.functional.log_softmax(out, dim=-1).transpose(0, 1)

                input_lengths = torch.full((x.size(0),), out.size(1), dtype=torch.long).to(DEVICE)
                target_lengths = (y != 0).sum(dim=1).clamp(min=1)

                loss = nn.CTCLoss(blank=0, reduction='mean')(log_probs, y, input_lengths, target_lengths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pth'))
            print(f"Saved Best Model (Val Loss: {avg_val_loss:.4f})")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'last.pth'))


if __name__ == "__main__":
    train()