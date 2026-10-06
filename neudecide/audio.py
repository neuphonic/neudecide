"""Audio input: WAV loading with the standard library and resampling with numpy."""

import math
import wave
from pathlib import Path

import numpy as np


def load_audio(path):
    """Reads a PCM WAV file. Returns (mono float32 waveform in [-1, 1], sample rate)."""
    try:
        with wave.open(str(path), "rb") as f:
            channels, width, sr = f.getnchannels(), f.getsampwidth(), f.getframerate()
            raw = f.readframes(f.getnframes())
    except wave.Error as e:
        raise ValueError(
            f"{path}: can only read PCM WAV files ({e}); load other formats yourself and "
            "pass a numpy array with sample_rate="
        ) from e
    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float32) / float(1 << 23)
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    return x.reshape(-1, channels).mean(axis=1), sr


def resample(x, orig_sr, target_sr, lowpass_filter_width=6, rolloff=0.99):
    """Band-limited sinc resampling with a Hann window -- the same kernel as
    torchaudio.functional.resample's defaults, which the model was trained with."""
    x = np.asarray(x, dtype=np.float32)
    if orig_sr == target_sr or x.size == 0:
        return x
    g = math.gcd(int(orig_sr), int(target_sr))
    orig, new = int(orig_sr) // g, int(target_sr) // g
    base = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base)
    idx = np.arange(-width, width + orig, dtype=np.float64)[None, :] / orig
    t = (np.arange(0, -new, -1, dtype=np.float64)[:, None] / new + idx) * base
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
    window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = np.where(t == 0, 1.0, np.sin(t) / t)
    kernel = (kernel * window * (base / orig)).astype(np.float32)  # (new, 2 * width + orig)

    padded = np.pad(x, (width, width + orig))
    frames = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape[1])[::orig]
    out = (frames @ kernel.T).reshape(-1)
    return out[: math.ceil(new * x.shape[0] / orig)].astype(np.float32)


def prepare_audio(audio, sample_rate, target_sr):
    """A path or array -> mono float32 at `target_sr`. Arrays are (samples,) or
    (samples, channels) and default to `target_sr` when `sample_rate` is None."""
    if isinstance(audio, (str, Path)):
        audio, sample_rate = load_audio(audio)
    else:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        elif audio.ndim != 1:
            raise ValueError(
                f"audio must be 1-D (samples,) or 2-D (samples, channels), got {audio.shape}"
            )
        sample_rate = sample_rate or target_sr
    return resample(audio, sample_rate, target_sr)
