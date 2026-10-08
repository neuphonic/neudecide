"""Audio input: WAV loading with the standard library and resampling with soxr."""

import wave
from pathlib import Path

import numpy as np
import soxr


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


def resample(x, orig_sr, target_sr):
    """Band-limited resampling with soxr (its default "HQ" quality)."""
    x = np.asarray(x, dtype=np.float32)
    if orig_sr == target_sr or x.size == 0:
        return x
    return soxr.resample(x, orig_sr, target_sr)


def to_float(x):
    """Integer PCM samples -> float32 in [-1, 1], scaled as load_audio scales WAV
    files (int16 / 32768, uint8 centred on 128). Float arrays are kept as they are."""
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.signedinteger):
        return x.astype(np.float32) / float(-np.iinfo(x.dtype).min)
    if np.issubdtype(x.dtype, np.unsignedinteger):
        mid = (np.iinfo(x.dtype).max + 1) / 2
        return (x.astype(np.float32) - mid) / mid
    return x.astype(np.float32, copy=False)


def prepare_audio(audio, sample_rate, target_sr):
    """A path or array -> mono float32 at `target_sr`. Arrays are (samples,) or
    (samples, channels) and default to `target_sr` when `sample_rate` is None;
    integer arrays are scaled to [-1, 1], float arrays should be in [-1, 1] already."""
    if isinstance(audio, (str, Path)):
        audio, sample_rate = load_audio(audio)
    else:
        audio = to_float(audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        elif audio.ndim != 1:
            raise ValueError(
                f"audio must be 1-D (samples,) or 2-D (samples, channels), got {audio.shape}"
            )
        sample_rate = sample_rate or target_sr
    return resample(audio, sample_rate, target_sr)
