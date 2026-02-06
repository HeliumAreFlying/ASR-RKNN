import os
import sys
import json
import editdistance
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).parent.parent))
from main_model import TDNNASR, public_device

FEAT_ROOT = "basic_data/zhvoice_feats"
META_DIR = "basic_data/feat_meta_data.json"
VOCAB_DIR = "basic_data/vocab_data.json"
SAVE_DIR = "weights"
TB_LOG_DIR = "tb_logs"
BATCH_SIZE = 64
ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-4
WARMUP_EPOCHS = 5
NUM_EPOCHS = 10000
VAL_SPLIT = 0.01
LOG_STEP = 10
NUM_WORKERS = 10
DEVICE = public_device

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TB_LOG_DIR, exist_ok=True)


class AudioDataset(Dataset):
    def __init__(self, meta_file, vocab_file, feat_root, is_train=True):
        with open(meta_file, 'r', encoding='utf-8') as f:
            self.meta_data = json.load(f)
        self.items = list(self.meta_data.items())
        self.feat_root = feat_root
        self.is_train = is_train

        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.token_to_id = self.vocab['token_to_id']

    def spec_augment(self, feat, max_f=15, max_t=40, num_f_masks=2, num_t_masks=2):
        T, F = feat.shape
        for _ in range(num_f_masks):
            f = np.random.randint(0, max_f)
            f0 = np.random.randint(0, F - f)
            feat[:, f0:f0 + f] = 0
        for _ in range(num_t_masks):
            t = np.random.randint(0, max_t)
            if T > t:
                t0 = np.random.randint(0, T - t)
                feat[t0:t0 + t, :] = 0
        return feat

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feat_rel_path, text = self.items[idx]
        feat_path = os.path.join(self.feat_root, feat_rel_path)
        try:
            feats = np.load(feat_path)
            if self.is_train:
                feats = self.spec_augment(feats)

            feats = torch.from_numpy(feats).float()
            feat_len = feats.size(0)
            label_ids = [self.token_to_id[char] for char in text if char in self.token_to_id]
            label_tensor = torch.tensor(label_ids, dtype=torch.long)
            return feats, label_tensor, feat_len
        except:
            return None, None, None


def collate_fn(batch):
    batch = [(f, l, fl) for f, l, fl in batch if f is not None and l is not None and len(l) > 0]
    if not batch: return None, None, None
    feats, labels, feat_lens = zip(*batch)
    padded_feats = pad_sequence(feats, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    feat_lens = torch.tensor(feat_lens, dtype=torch.long)
    return padded_feats, padded_labels, feat_lens


def compute_cer(pred_ids, target_ids):
    total_edits, total_chars = 0, 0
    for p, t in zip(pred_ids, target_ids):
        p_decoded = [p[i] for i in range(len(p)) if p[i] != 0 and (i == 0 or p[i] != p[i - 1])]
        t_clean = [token for token in t if token != 0]
        total_edits += editdistance.eval(p_decoded, t_clean)
        total_chars += len(t_clean)
    return total_edits / total_chars if total_chars > 0 else float('inf')


def train():
    full_dataset = AudioDataset(META_DIR, VOCAB_DIR, FEAT_ROOT, is_train=True)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.is_train = False

    train_loader = DataLoader(train_dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, pin_memory=True)

    model = TDNNASR(
        input_dim=80,
        block_dims=[256] * 8 + [384] * 8 + [512] * 12,
        dilations=[1, 2, 4] * 9 + [1],
        strides=[2] + [1] * 8 + [2] + [1] * 18,
        proj_dim=512,
        num_classes=full_dataset.vocab['vocab_size'], vocab_data=full_dataset.vocab
    ).to(DEVICE)

    optimizer = torch.optim.RAdam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    reduction = model.reduction
    writer = SummaryWriter(TB_LOG_DIR)
    best_cer, global_step = float('inf'), 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        if epoch < WARMUP_EPOCHS:
            curr_lr = LEARNING_RATE * ((epoch + 1) / WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            if batch is None or batch[0] is None: continue
            x, y, feat_lens = batch
            x, y, feat_lens = x.to(DEVICE), y.to(DEVICE), feat_lens.to(DEVICE)
            x = x.transpose(1, 2).unsqueeze(-1).contiguous()

            with torch.amp.autocast(device_type="cuda"):
                out = model(x)
                out = out.squeeze(-1).transpose(1, 2)
                log_probs = torch.nn.functional.log_softmax(out, dim=-1).transpose(0, 1)
                output_lengths = torch.clamp((feat_lens // reduction), max=out.size(1))
                target_lengths = (y != 0).sum(dim=1)
                loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)(log_probs, y, output_lengths,
                                                                                 target_lengths)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUMULATION_STEPS
            global_step += 1
            if (batch_idx + 1) % LOG_STEP == 0:
                writer.add_scalar('Train/Loss', loss.item() * ACCUMULATION_STEPS, global_step)

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        model.eval()
        val_loss, all_preds, all_targets = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None or batch[0] is None: continue
                x, y, feat_lens = batch
                x, y, feat_lens = x.to(DEVICE), y.to(DEVICE), feat_lens.to(DEVICE)
                x = x.transpose(1, 2).unsqueeze(-1).contiguous()
                out = model(x)
                out = out.squeeze(-1).transpose(1, 2)
                log_probs = torch.nn.functional.log_softmax(out, dim=-1).transpose(0, 1)
                output_lengths = torch.clamp((feat_lens // reduction), max=out.size(1))
                target_lengths = (y != 0).sum(dim=1)
                val_loss += nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)(log_probs, y, output_lengths,
                                                                                      target_lengths).item()
                preds = torch.argmax(out, dim=-1)
                for i in range(preds.size(0)):
                    all_preds.append(preds[i][:output_lengths[i]].cpu().tolist())
                    all_targets.append(y[i][:target_lengths[i]].cpu().tolist())

        cer = compute_cer(all_preds, all_targets)
        writer.add_scalar('Val/Loss', val_loss / len(val_loader), epoch)
        writer.add_scalar('Val/CER', cer, epoch)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)

        if cer < best_cer:
            best_cer = cer
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pth'))
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'last.pth'))

    writer.close()


if __name__ == "__main__":
    train()