import os
import sys
sys.path.append(".")

import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from utils.get_feat import load_and_resample_audio, compute_feat

TRAINING_DATA_ROOT = "../datasets/zhvoice"
META_FILE = "basic_data/clean_meta_data.json"
FEAT_SAVE_DIR = "basic_data//zhvoice_feats"
NUM_PROCESSES = 12


def process_single_item(item):
    rel_path, text = item
    wav_path = os.path.join(TRAINING_DATA_ROOT, rel_path)
    feat_rel_path = rel_path.rsplit('.', 1)[0] + ".npy"
    save_path = os.path.join(FEAT_SAVE_DIR, feat_rel_path)

    if os.path.exists(save_path):
        return rel_path, feat_rel_path

    try:
        samples, sr = load_and_resample_audio(wav_path, target_sr=16000)
        feats = compute_feat(samples, sample_rate=16000)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, feats.astype(np.float32))
        return rel_path, feat_rel_path
    except:
        return None


if __name__ == "__main__":
    with open(META_FILE, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    items = list(meta_data.items())
    new_meta_data = {}

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        results = list(tqdm(executor.map(process_single_item, items), total=len(items)))

    for res in results:
        if res is not None:
            orig_rel_path, feat_rel_path = res
            new_meta_data[feat_rel_path] = meta_data[orig_rel_path]

    with open("basic_data/feat_meta_data.json", 'w', encoding='utf-8') as f:
        json.dump(new_meta_data, f, ensure_ascii=False, indent=4)