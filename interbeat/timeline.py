"""Beat timeline model and the de-jittered local beat map.

A Timeline is a linear map between a node's local clock (seconds) and the
shared musical beat position:

    beat = b0 + (t - t0) * tempo / 60

The audio callback only ever reads a BeatMap. The network never touches the
BeatMap directly: received timeline samples go into TimelineFilter, which
steers the map with a bounded-rate PI loop, so packet delay jitter cannot
jitter the audible beat grid.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


class Clock:
    def now(self) -> float:
        raise NotImplementedError


class SystemClock(Clock):
    def now(self) -> float:
        return time.monotonic()


@dataclass(frozen=True)
class Timeline:
    b0: float    # beat position at the anchor
    t0: float    # local time at the anchor
    tempo: float  # beats per minute

    def beat_at(self, t: float) -> float:
        return self.b0 + (t - self.t0) * self.tempo / 60.0

    def time_at(self, beat: float) -> float:
        return self.t0 + (beat - self.b0) * 60.0 / self.tempo


class BeatMap:
    """Piecewise-linear beat<->time map consumed by the audio callback.

    Every update re-anchors at the update instant, so the map is always
    continuous in beat position; only its slope (tempo) changes, and slope
    changes are bounded by whoever steers it (the filter limits the rate).
    """

    def __init__(self, tempo: float = 120.0):
        self._tl = Timeline(0.0, 0.0, tempo)

    def steer_to(self, now: float, tempo: float) -> None:
        self._tl = Timeline(self._tl.beat_at(now), now, tempo)

    def jump(self, now: float, beat: float, tempo: float) -> None:
        """Hard re-lock. Only used before playback starts or on an explicit
        session-wide phase jump, never from routine packet arrivals."""
        self._tl = Timeline(beat, now, tempo)

    @property
    def tempo(self) -> float:
        return self._tl.tempo

    def beat_at(self, t: float) -> float:
        return self._tl.beat_at(t)

    def time_at(self, beat: float) -> float:
        return self._tl.time_at(beat)


class TimelineFilter:
    """Locks the local BeatMap onto timeline samples from the master.

    - Offset estimation uses the minimum-delay sample in a sliding window
      (min-delay filter): delayed or reordered packets lose the min() and are
      ignored, so reordering cannot move the map.
    - The first sample performs a one-time hard lock (used before playback).
    - Afterwards a bounded PI loop slews the map: a burst of jittery packets
      moves the audible grid by at most max_adj per control tick.
    """

    MAX_SAMPLES = 64  # bounded queue
    ADOPT_MAX_ERR = 0.5  # beats; beyond this the grids are unrelated

    def __init__(self, kp: float = 3.0, ki: float = 0.6, max_adj: float = 0.05,
                 window: float = 4.0):
        self.kp = kp
        self.ki = ki
        self.max_adj = max_adj
        self.window = window
        self.samples: deque = deque()  # (recv_local, master_send, beat_at_send, tempo)
        self.locked = False
        self.base_tempo: float | None = None
        self.err_int = 0.0
        self.adj = 0.0
        self.saturated = False
        self.last_err = 0.0
        self.last_delay = None  # min-link-delay estimate of the last update
        self.last_jump_reason = None  # why the last update() hard-jumped
        self._fresh = False  # first sample after (re)adoption not yet verified

    def reset(self, keep_lock: bool = False) -> None:
        self.samples.clear()
        self.err_int = 0.0
        self.adj = 0.0
        self.saturated = False
        self._fresh = True
        if not keep_lock:
            self.locked = False
            self.base_tempo = None

    def adopt_current(self, beatmap: BeatMap) -> None:
        """A stepping-down master adopts its own grid as the lock state. The
        adoption is provisional: the first sample from the new master
        verifies the grids actually agree (see update())."""
        self.samples.clear()
        self.err_int = 0.0
        self.adj = 0.0
        self.saturated = False
        self.locked = True
        self.base_tempo = beatmap.tempo
        self._fresh = True

    def add_sample(self, tempo: float, beat: float, master_send_t: float,
                   recv_local_t: float) -> None:
        # Drop duplicates/late arrivals of an already-superseded send time.
        if self.samples and master_send_t < self.samples[-1][1]:
            return
        self.samples.append((recv_local_t, master_send_t, beat, tempo))
        while len(self.samples) > self.MAX_SAMPLES:
            self.samples.popleft()

    def update(self, beatmap: BeatMap, now: float, dt: float) -> bool:
        """Steer the beat map toward the master's grid. Returns True if the
        map made a hard jump this call (the engine re-arms scheduling then)."""
        if not self.samples:
            return False
        # expire old samples
        cutoff = now - self.window
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        if not self.samples:
            return False
        recv, send, beat, tempo = min(self.samples, key=lambda s: s[0] - s[1])
        self.last_delay = recv - send
        # Extrapolate the master's beat position to now. The unknown clock
        # offset cancels; the residual error is the min link delay.
        target_beat = beat + (now - recv) * tempo / 60.0
        if not self.locked or self.base_tempo is None:
            beatmap.jump(now, target_beat, tempo)
            self.base_tempo = tempo
            self.locked = True
            self.err_int = 0.0
            self.adj = 0.0
            self._fresh = False
            self.last_jump_reason = "initial_lock"
            return True
        err = target_beat - beatmap.beat_at(now)
        if self._fresh:
            self._fresh = False
            if abs(err) > self.ADOPT_MAX_ERR:
                # The adopted/current grid is unrelated to the session grid
                # (e.g. a freshly self-elected node discovering an existing
                # session): hard re-lock instead of slewing forever.
                beatmap.jump(now, target_beat, tempo)
                self.base_tempo = tempo
                self.err_int = 0.0
                self.adj = 0.0
                self.last_jump_reason = "adopt_relock"
                return True
        if tempo != self.base_tempo:
            # Session tempo changed: adopt the new rate continuously (the map
            # re-anchors at `now`, position is preserved), PLL removes the
            # residual phase error.
            self.base_tempo = tempo
            self.err_int = 0.0
        self.last_err = err
        self.err_int = max(-0.25, min(0.25, self.err_int + err * dt))
        adj = self.kp * err + self.ki * self.err_int
        self.saturated = abs(adj) > self.max_adj
        self.adj = max(-self.max_adj, min(self.max_adj, adj))
        beatmap.steer_to(now, self.base_tempo * (1.0 + self.adj))
        return False
