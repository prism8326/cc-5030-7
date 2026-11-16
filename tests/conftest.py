import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interbeat.app import BeatApp
from interbeat.measure import find_onsets, pair_deviations, stats
from interbeat.transport import VirtualNetwork


class World:
    """Deterministic multi-app world: virtual network + per-app audio clocks.

    Each world step advances true time by dt; every app renders n = dt*sr
    frames, so sample s of an app's concatenated output corresponds to true
    time s/sr - this is what makes cross-app onset deviation measurable.
    Node clocks drift via each engine's drift_ppm.
    """

    def __init__(self, sr=8000, dt=0.01, seed=7, **link):
        self.sr = sr
        self.dt = dt
        self.n = int(round(dt * sr))
        self.net = VirtualNetwork(seed=seed)
        self.net.set_all_links(**link)
        self.apps = {}
        self.audio = {}
        self.birth = {}

    def add_app(self, node_id, index, drift_ppm=0.0, t0=0.0, quantum=4.0):
        tr = self.net.connect(node_id)
        app = BeatApp(node_id, node_id, tr, node_index=index, sr=self.sr,
                      drift_ppm=drift_ppm, t0=t0, quantum=quantum)
        tr.clock = app.engine  # arrival stamps in the node's own clock
        self.apps[node_id] = app
        self.audio[node_id] = []
        self.birth[node_id] = self.net.t
        return app

    def remove_app(self, node_id):
        del self.apps[node_id]

    def step(self, steps=1):
        for _ in range(steps):
            self.net.advance(self.dt)
            for node_id, app in self.apps.items():
                self.audio[node_id].append(app.tick(self.n))

    def run(self, seconds):
        self.step(int(round(seconds / self.dt)))

    @property
    def t(self):
        return self.net.t

    def audio_of(self, node_id):
        """Concatenated output aligned to true time: sample s <-> true s/sr.
        Late joiners are left-padded with silence for their pre-birth span."""
        pad = np.zeros(int(round(self.birth.get(node_id, 0.0) * self.sr)))
        if not self.audio[node_id]:
            return pad
        return np.concatenate([pad] + self.audio[node_id])

    def onsets(self, node_id, t_from=0.0):
        x = self.audio_of(node_id)
        on = find_onsets(x, self.sr) / self.sr
        return on[on >= t_from]

    def onset_slots(self, node_id):
        """Map each onset (true time) to its shared-beat slot (1/4-beat
        grid) using the app's OWN beat map - the mapping is exactly what is
        under test, but slot rounding tolerates up to +/-62 ms of error,
        far above the deviations we assert."""
        app = self.apps[node_id]
        on = self.onsets(node_id)
        birth = self.birth[node_id]
        local = app.engine.t0 + (on - birth) * (1.0 + app.engine.drift)
        beats = app.session.beatmap.beat_at(local)
        slots = np.round(beats * 4.0) / 4.0
        out = {}
        for s, t in zip(slots, on):
            out.setdefault(float(s), float(t))  # first onset per slot
        return out

    def deviation(self, a, b, t_from=0.0):
        """Pair onsets that fall on the SAME shared-beat slot and compare
        their true times. Never pairs across different beats, regardless
        of how dense each app's pattern is."""
        sa = self.onset_slots(a)
        sb = self.onset_slots(b)
        common = sorted(set(sa) & set(sb))
        devs = np.array([sb[s] - sa[s] for s in common
                         if sa[s] >= t_from and sb[s] >= t_from])
        return stats(devs)


@pytest.fixture
def world():
    return World()
