"""Measurement: onset detection on synthesized output and inter-app
hit-deviation statistics."""
from __future__ import annotations

import numpy as np


def _boxcar(e: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(e)])
    return (c[w:] - c[:-w]) / w


def find_onsets(x: np.ndarray, sr: int, thresh: float = 0.01,
                refractory_s: float = 0.08) -> np.ndarray:
    """High-pass energy onset detector.

    The signal is first differentiated: drum tails (low-frequency decaying
    bodies) are strongly attenuated while attack transients pass through, so
    a quiet hat right after a loud kick is still detected. Onsets are where
    the short-window energy of the derivative crosses `thresh`; a refractory
    shorter than the smallest inter-hit gap suppresses wobble. The detection
    delay is a small constant that cancels when comparing apps."""
    if len(x) < 64:
        return np.empty(0, dtype=int)
    y = np.diff(x)
    e = y * y
    w_s = max(1, int(0.005 * sr))
    env = _boxcar(e, w_s)
    above = env > thresh
    crossings = np.flatnonzero(above & ~np.roll(above, 1))
    if len(crossings) and crossings[0] == 0:
        crossings = crossings[1:]
    if len(crossings) == 0:
        return crossings
    refr = int(refractory_s * sr)
    keep = []
    last = -10 ** 9
    for c in crossings:
        if c - last < refr:
            continue
        # hysteresis re-arm: the envelope must have dipped below half
        # threshold since the last onset, otherwise this is the same hit
        # still ringing (e.g. a snare's noisy tail)
        if keep and env[last:c].min() > 0.5 * thresh:
            continue
        keep.append(c)
        last = c
    return np.array(keep, dtype=int)


def pair_deviations(times_a: np.ndarray, times_b: np.ndarray,
                    max_gap_s: float = 0.15) -> np.ndarray:
    """For each onset in a, nearest onset in b within max_gap. Returns
    signed deviations (b - a) in seconds."""
    devs = []
    j0 = 0
    for ta in times_a:
        j = j0
        best = None
        while j < len(times_b) and times_b[j] < ta - max_gap_s:
            j += 1
        j0 = j
        for k in (j, j + 1):
            if k < len(times_b):
                d = times_b[k] - ta
                if abs(d) <= max_gap_s and (best is None or abs(d) < abs(best)):
                    best = d
        if best is not None:
            devs.append(best)
    return np.array(devs)


def stats(devs_s: np.ndarray) -> dict:
    if len(devs_s) == 0:
        return {"n": 0, "mean_ms": 0.0, "max_ms": 0.0, "p99_ms": 0.0}
    ms = np.abs(devs_s) * 1000.0
    return {"n": int(len(ms)),
            "mean_ms": float(np.mean(ms)),
            "max_ms": float(np.max(ms)),
            "p99_ms": float(np.percentile(ms, 99))}
