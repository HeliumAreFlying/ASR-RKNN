import librosa
import numpy as np
import soundfile as sf
import kaldi_native_fbank as knf
from typing import Tuple, List, Optional

def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(filename, always_2d=True, dtype="float32")
    data = data[:, 0]
    return np.ascontiguousarray(data), int(sample_rate)

def load_and_resample_audio(filename: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    samples, sr = load_audio(filename)
    if sr != target_sr:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return np.ascontiguousarray(samples, dtype=np.float32), int(sr)

def compute_feat(samples: np.ndarray, sample_rate: int, window_size: int, window_shift: int) -> np.ndarray:
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    online_fbank.input_finished()

    if online_fbank.num_frames_ready == 0:
        return np.zeros((0, 80 * window_size), dtype=np.float32)

    features = np.stack([online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)])

    if features.shape[0] > 1:
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        features = (features - mean) / (std + 1e-8)

    T = (features.shape[0] - window_size) // window_shift + 1
    if T <= 0:
        return np.zeros((0, features.shape[1] * window_size), dtype=np.float32)

    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(T, features.shape[1] * window_size),
        strides=((window_shift * features.shape[1]) * 4, 4),
    )
    return np.ascontiguousarray(features, dtype=np.float32)

if __name__ == "__main__":
    wave_filepath = "examples/zh.wav"
    samples, sr = load_and_resample_audio(wave_filepath)
    feats = compute_feat(samples, sample_rate=16000, window_size=7, window_shift=1)
    print(feats.shape)

