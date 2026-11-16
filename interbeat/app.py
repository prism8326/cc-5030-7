"""BeatApp: one sequencer application = session + engine + pattern."""
from __future__ import annotations

from . import sequencer as seq
from .engine import AudioEngine
from .session import Session
from .transport import Transport


class BeatApp:
    def __init__(self, node_id: str, name: str, transport: Transport,
                 node_index: int = 0, sr: int = 44100, drift_ppm: float = 0.0,
                 t0: float = 0.0, quantum: float = 4.0, on_event=None):
        self.events: list[tuple[str, dict]] = []
        self.engine = AudioEngine(sr=sr, drift_ppm=drift_ppm, t0=t0,
                                  quantum=quantum)
        self.session = Session(node_id, name, self.engine, transport,
                               quantum=quantum, on_event=self._on_event)
        self.sequencer = seq.Sequencer(seq.default_pattern(node_index))
        self.engine.bind(self.session, self.sequencer)
        self.session.on_map_jump = self.engine.reset_watermark
        self._user_on_event = on_event

    def _on_event(self, kind: str, **kw) -> None:
        self.events.append((kind, kw))
        if len(self.events) > 512:
            del self.events[:256]
        if self._user_on_event is not None:
            self._user_on_event(kind, **kw)

    def tick(self, n_frames: int):
        """One control+audio cycle: process network, steer the beat map,
        then render n_frames of audio. Returns the audio buffer."""
        self.session.tick()
        return self.engine.render(n_frames)

    @property
    def now(self) -> float:
        return self.engine.now()
