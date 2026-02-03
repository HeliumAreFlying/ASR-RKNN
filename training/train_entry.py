import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import sys
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).parent.parent))
from main_model import TDNNASR, public_device
from utils.get_feat import load_and_resample_audio, compute_feat

TRAINING_DATA_ROOT = "../datasets/zhvoice"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10000
VAL_SPLIT = 0.1
SAVE_DIR = "weights"
LOG_STEP = 10
DEVICE = public_device
TB_LOG_DIR = "tb_logs"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TB_LOG_DIR, exist_ok=True)

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
            feats = compute_feat(samples, sample_rate=16000)
            feats = torch.from_numpy(feats).float()
            feat_len = feats.size(0)

            label_ids = []
            for char in text:
                if char in self.token_to_id:
                    token_id = self.token_to_id[char]
                    label_ids.append(token_id + 1)

            label_tensor = torch.tensor(label_ids, dtype=torch.long)
            return feats, label_tensor, feat_len

        except Exception as e:
            return None, None, None

def collate_fn(batch):
    batch = [(f, l, fl) for f, l, fl in batch if f is not None and l is not None and len(l) > 0]
    if not batch:
        return None, None, None
    feats, labels, feat_lens = zip(*batch)
    padded_feats = pad_sequence(feats, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    feat_lens = torch.tensor(feat_lens, dtype=torch.long)
    return padded_feats, padded_labels, feat_lens

def train():
    dataset = AudioDataset('basic_data/clean_meta_data.json', 'basic_data/vocab_data.json')

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset,
                              num_workers=10,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              collate_fn=collate_fn,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            num_workers=10,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            collate_fn=collate_fn)

    vocab_size = dataset.vocab['vocab_size']
    num_classes = vocab_size + 1

    model = TDNNASR(
        input_dim=80,
        block_dims=[384] * 13,
        dilations=[1, 2, 4, 4, 4, 2, 1, 2, 4, 4, 4, 2, 1],
        strides=[1] * 12 + [2],
        proj_dim=384,
        num_classes=num_classes,
        vocab_data=dataset.vocab
    ).to(public_device)

    optimizer = torch.optim.RAdam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = GradScaler()

    best_val_loss = float('inf')
    reduction = model.reduction
    writer = SummaryWriter(TB_LOG_DIR)
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None or batch[0] is None:
                continue
            x, y, feat_lens = batch
            x, y, feat_lens = x.to(DEVICE), y.to(DEVICE), feat_lens.to(DEVICE)
            x = x.transpose(1, 2).unsqueeze(-1)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=DEVICE):
                out = model(x)
                out = out.squeeze(-1).transpose(1, 2)
                log_probs = torch.nn.functional.log_softmax(out, dim=-1).transpose(0, 1)

                output_lengths = (feat_lens + reduction - 1) // reduction
                target_lengths = (y != 0).sum(dim=1)

                loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)(log_probs, y, output_lengths, target_lengths)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            global_step += 1

            if (batch_idx + 1) % LOG_STEP == 0:
                writer.add_scalar('Train/Step_Loss', loss.item(), global_step)

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar('Train/Epoch_Avg_Loss', avg_train_loss, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None or batch[0] is None:
                    continue
                x, y, feat_lens = batch
                x, y, feat_lens = x.to(DEVICE), y.to(DEVICE), feat_lens.to(DEVICE)
                x = x.transpose(1, 2).unsqueeze(-1)

                out = model(x)
                out = out.squeeze(-1).transpose(1, 2)
                log_probs = torch.nn.functional.log_softmax(out, dim=-1).transpose(0, 1)

                output_lengths = (feat_lens + reduction - 1) // reduction
                target_lengths = (y != 0).sum(dim=1)

                loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)(log_probs, y, output_lengths, target_lengths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Val/Epoch_Avg_Loss', avg_val_loss, epoch)
        writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pth'))

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'last.pth'))

    writer.close()

if __name__ == "__main__":
    train()