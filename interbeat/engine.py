"""Audio engine: the sample-accurate scheduling core.

The engine IS the node's clock (network timestamps, filter updates and
scheduling all share it). render() is the audio callback: it maps the
buffer's frame span onto the shared beat grid via the BeatMap snapshot and
places synthesized hits at exact sample offsets. Network packets never
touch this path directly - they only steer the BeatMap through the
slew-limited TimelineFilter, so latency jitter cannot jitter sound.

Scheduling is robust to float re-anchoring of the beat map: events are
de-duplicated by their deterministic (phrase, step) id rather than by
float window arithmetic, so an event landing exactly on a buffer boundary
is never skipped or doubled.

Bounded work: candidates are generated statelessly per buffer, voices are
capped, the dedupe set is bounded, and all queues are bounded.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from . import synth
from .timeline import BeatMap, Clock

MAX_VOICES = 32
MAX_EVENTS_PER_BUFFER = 64
START_MARGIN_BEATS = 0.25
LATE_SECONDS = 0.25       # older than this: count as missed, don't fire
LOOKBACK_BEATS = 1.0      # candidate window margin behind the buffer
DEDUP_MAX = 256


class AudioEngine(Clock):
    def __init__(self, sr: int = 44100, drift_ppm: float = 0.0, t0: float = 0.0,
                 quantum: float = 4.0):
        self.sr = sr
        self.drift = drift_ppm * 1e-6
        self.frame = 0
        self.t0 = t0
        self.quantum = quantum
        self.beatmap: BeatMap | None = None
        self.session = None
        self.sequencer = None
        self.voices: list[tuple[np.ndarray, int]] = []
        self.hit_log: deque = deque(maxlen=8192)  # (beat, node_time) ground truth
        self.on_hit = None  # callback(beat, inst) for UI / taps
        self.overflows = 0
        self.missed = 0
        self._scheduled: deque = deque(maxlen=DEDUP_MAX)
        self._scheduled_set: set = set()
        self._active = False
        self._active_from = 0.0

    # Clock interface -------------------------------------------------------
    def now(self) -> float:
        return self.t0 + (self.frame / self.sr) * (1.0 + self.drift)

    def reset_watermark(self) -> None:
        """Called when the beat map makes a hard jump (first lock, deliberate
        phase jump): re-arm scheduling and the quantized entrance."""
        self._scheduled.clear()
        self._scheduled_set.clear()
        self._active = False

    def bind(self, session, sequencer) -> None:
        self.session = session
        self.sequencer = sequencer
        self.beatmap = session.beatmap
        self.quantum = session.quantum

    def _mark(self, eid) -> None:
        if len(self._scheduled) == self._scheduled.maxlen:
            self._scheduled_set.discard(self._scheduled[0])
        self._scheduled.append(eid)
        self._scheduled_set.add(eid)

    # The audio callback ----------------------------------------------------
    def render(self, n_frames: int) -> np.ndarray:
        out = np.zeros(n_frames, dtype=np.float64)
        bm = self.beatmap
        ses = self.session
        t0 = self.now()
        scale = (1.0 + self.drift) / self.sr  # node-seconds per frame
        t1 = t0 + n_frames * scale

        b0 = bm.beat_at(t0)
        b1 = bm.beat_at(t1)

        ready = ses.is_master or ses.filter.locked
        if ses.playing and ready:
            if not self._active:
                cand = ses.play_beat
                if cand < b0 + START_MARGIN_BEATS:
                    # Late join / late packet: enter at the next explicit
                    # quantization boundary, never immediately at beat 1.
                    cand = math.ceil((b0 + START_MARGIN_BEATS - 1e-9)
                                     / self.quantum) * self.quantum
                self._active_from = cand
                self._active = True
            lo_gate, hi_gate = self._active_from, math.inf
        elif not ses.playing:
            self._active = False
            lo_gate, hi_gate = -math.inf, ses.play_beat  # stop beat
        else:
            lo_gate, hi_gate = math.inf, -math.inf  # waiting for lock

        if hi_gate > lo_gate:
            w0 = max(b0, lo_gate) - LOOKBACK_BEATS
            w1 = min(b1, hi_gate) + LOOKBACK_BEATS
            candidates = self.sequencer.events_between(w0, w1)
            if len(candidates) > MAX_EVENTS_PER_BUFFER:
                self.overflows += 1
                candidates = candidates[:MAX_EVENTS_PER_BUFFER]
        else:
            candidates = []

        for beat, inst, vel, note, eid in candidates:
            if eid in self._scheduled_set:
                continue
            if beat < lo_gate or beat >= hi_gate:
                continue
            t_ev = bm.time_at(beat)
            if t_ev >= t1:
                break  # future event; re-evaluated next buffer
            self._mark(eid)
            if t_ev < t0 - LATE_SECONDS:
                self.missed += 1
                continue
            off = int(round((t_ev - t0) / scale))
            off = max(0, min(n_frames - 1, off))
            samples = synth.render(inst, self.sr, vel, note)
            if len(self.voices) >= MAX_VOICES:
                self.overflows += 1
                continue
            self.voices.append((samples, -off))
            self.hit_log.append((beat, t_ev))
            if self.on_hit is not None:
                self.on_hit(beat, inst)

        # mix active voices; pos = source index aligned to this buffer's start
        # (negative means the voice starts -pos frames into the buffer)
        still = []
        for samples, pos in self.voices:
            start = max(0, -pos)
            src0 = max(0, pos)
            n = min(n_frames - start, len(samples) - src0)
            if n > 0:
                out[start:start + n] += samples[src0:src0 + n]
            pos += n_frames
            if pos < len(samples):
                still.append((samples, pos))
        self.voices = still

        self.frame += n_frames
        return out
