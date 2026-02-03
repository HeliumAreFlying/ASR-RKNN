import numpy as np
import soundfile as sf
import kaldi_native_fbank as knf
import resampy
from typing import Tuple

def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(filename, always_2d=True, dtype="float32")
    data = data[:, 0]
    return np.ascontiguousarray(data), int(sample_rate)

def load_and_resample_audio(filename: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    samples, sr = load_audio(filename)
    if sr != target_sr:
        samples = resampy.resample(samples, sr, target_sr)
        sr = target_sr
    return np.ascontiguousarray(samples, dtype=np.float32), int(sr)

def compute_feat(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80
    opts.frame_opts.preemph_coeff = 0.97

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    fbank.input_finished()

    if fbank.num_frames_ready == 0:
        return np.zeros((0, 80), dtype=np.float32)

    features = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
    features = features - np.mean(features, axis=0, keepdims=True)
    return np.ascontiguousarray(features, dtype=np.float32)

if __name__ == "__main__":
    wave_filepath = "examples/zh.wav"
    samples, sr = load_and_resample_audio(wave_filepath, target_sr=16000)
    feats = compute_feat(samples, sample_rate=sr)
    print(feats.shape)