"""Patterns and the stateless sequencer.

Phrase position is derived from the shared beat grid:
phrase_index = floor(beat / phrase_beats). Two apps with identical beat
phase but different phrase lengths are therefore at different bar positions
- phase alignment does not imply bar alignment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    beat: float      # position inside the phrase, in beats
    inst: str        # instrument: kick / snare / hat / blip
    vel: float = 1.0
    note: float = 0.0  # semitone offset for pitched instruments


class Pattern:
    def __init__(self, phrase_beats: float, steps: list[Step]):
        self.phrase_beats = float(phrase_beats)
        self.steps = sorted(steps, key=lambda s: s.beat)

    def toggle(self, index: int) -> None:
        s = self.steps[index]
        self.steps[index] = Step(s.beat, s.inst, 0.0 if s.vel > 0 else 1.0,
                                 s.note)


def default_pattern(node_index: int) -> Pattern:
    """Three complementary patterns with different phrase lengths."""
    if node_index == 0:  # 4-beat phrase: four-on-the-floor + backbeat snare
        steps = [Step(b, "kick", 1.0) for b in (0, 1, 2, 3)]
        steps += [Step(b, "snare", 0.9) for b in (1, 3)]
        steps += [Step(b + 0.5, "hat", 0.6) for b in (0, 1, 2, 3)]
        return Pattern(4, steps)
    if node_index == 1:  # 6-beat phrase
        steps = [Step(0, "kick", 1.0), Step(3, "kick", 0.9),
                 Step(2, "snare", 0.9), Step(5, "snare", 0.8),
                 Step(1, "hat", 0.6), Step(4, "hat", 0.6),
                 Step(0.75, "blip", 0.5, 0), Step(2.75, "blip", 0.5, 3),
                 Step(4.75, "blip", 0.5, 7)]
        return Pattern(6, steps)
    # 8-beat phrase: melodic blips on every beat + downbeat kicks
    steps = [Step(b, "blip", 0.6, n)
             for b, n in zip(range(8), [0, 3, 7, 10, 12, 10, 7, 3])]
    steps += [Step(0, "kick", 1.0), Step(4, "kick", 0.9)]
    steps += [Step(b + 0.5, "hat", 0.5) for b in range(8)]
    return Pattern(8, steps)


class Sequencer:
    """Generates events for any beat window. Stateless: no unbounded queues,
    the same window always yields the same events. Each event carries a
    deterministic id (phrase, step) so the engine can de-duplicate across
    buffers without relying on float window arithmetic."""

    def __init__(self, pattern: Pattern):
        self.pattern = pattern

    def events_between(self, b0: float, b1: float) -> list[tuple]:
        """Returns (beat, inst, vel, note, event_id) tuples, beat-sorted."""
        ph = self.pattern.phrase_beats
        if b1 <= b0 or ph <= 0:
            return []
        out = []
        p0 = math.floor(b0 / ph)
        p1 = math.floor((b1 - 1e-12) / ph)
        for p in range(p0, p1 + 1):
            base = p * ph
            for i, s in enumerate(self.pattern.steps):
                if s.vel > 0.0:
                    gb = base + s.beat
                    if b0 <= gb < b1:
                        out.append((gb, s.inst, s.vel, s.note, (p, i)))
        out.sort(key=lambda e: (e[0], e[4]))
        return out
