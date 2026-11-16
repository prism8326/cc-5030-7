"""Drum / blip synthesis. Pure numpy, fixed-length renders (bounded cost)."""
from __future__ import annotations

import numpy as np

MAX_HIT_SECONDS = 0.35


def _env(n: int, sr: int, decay: float) -> np.ndarray:
    t = np.arange(n) / sr
    return np.exp(-t / max(1e-3, decay))


def kick(sr: int, vel: float = 1.0) -> np.ndarray:
    n = int(0.28 * sr)
    t = np.arange(n) / sr
    f = 48.0 + 110.0 * np.exp(-t / 0.03)
    phase = 2.0 * np.pi * np.cumsum(f) / sr
    body = np.sin(phase) * _env(n, sr, 0.09)
    click = np.sign(np.sin(2 * np.pi * 3200 * t[: int(0.004 * sr)]))
    click = np.pad(click, (0, n - len(click))) * _env(n, sr, 0.002) * 0.4
    return (body + click) * vel * 0.9


def snare(sr: int, vel: float = 1.0) -> np.ndarray:
    n = int(0.18 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(1234)
    noise = rng.standard_normal(n) * _env(n, sr, 0.045)
    tone = np.sin(2 * np.pi * 190 * t) * _env(n, sr, 0.05)
    return (0.7 * noise + 0.5 * tone) * vel * 0.7


def hat(sr: int, vel: float = 1.0) -> np.ndarray:
    n = int(0.06 * sr)
    rng = np.random.default_rng(99)
    noise = rng.standard_normal(n)
    noise = np.diff(noise, prepend=0.0)  # crude high-pass
    return noise * _env(n, sr, 0.012) * vel * 0.5


def blip(sr: int, vel: float = 1.0, note: float = 0.0) -> np.ndarray:
    n = int(0.16 * sr)
    t = np.arange(n) / sr
    f = 440.0 * 2.0 ** (note / 12.0)
    sig = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t))
    return sig * _env(n, sr, 0.05) * vel * 0.5


def render(inst: str, sr: int, vel: float = 1.0, note: float = 0.0) -> np.ndarray:
    if inst == "kick":
        return kick(sr, vel)
    if inst == "snare":
        return snare(sr, vel)
    if inst == "hat":
        return hat(sr, vel)
    if inst == "blip":
        return blip(sr, vel, note)
    raise ValueError(f"unknown instrument {inst!r}")
