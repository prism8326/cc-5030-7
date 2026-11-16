"""Wire protocol: JSON datagrams with independently versioned state components.

Beat position (phase), tempo and start/stop are separate session-state
components, each ordered by its own Lamport-style version, so a stale or
reordered packet of one component can never roll back another:

  HELLO     membership + piggybacked play-state (component: play_seq)
  TIMELINE  master's beat grid sample (components: tempo_seq, phase_seq,
            all scoped by the mastership term)
  PLAY      start/stop request at a quantized shared beat (component: play_seq)
  MASTER    mastership claim (component: term)

All ordering fields are Lamport clocks: a node only ever accepts a strictly
newer (or equal, for idempotent re-broadcasts) version.
"""
from __future__ import annotations

import json

HELLO = "hello"
TIMELINE = "timeline"
PLAY = "play"
MASTER = "master"

_VALID = {HELLO, TIMELINE, PLAY, MASTER}


def encode(msg: dict) -> bytes:
    return json.dumps(msg, separators=(",", ":")).encode("utf-8")


def decode(data: bytes) -> dict | None:
    try:
        msg = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(msg, dict) or msg.get("type") not in _VALID:
        return None
    if not isinstance(msg.get("node"), str):
        return None
    return msg


def hello(node: str, name: str, term: int, master_id: str | None,
          play: dict) -> dict:
    return {"type": HELLO, "node": node, "name": name, "term": term,
            "master_id": master_id, "play": play}


def timeline(node: str, term: int, tempo_seq: int, phase_seq: int,
             tempo: float, beat: float, at: float, hard: bool = False) -> dict:
    return {"type": TIMELINE, "node": node, "term": term,
            "tempo_seq": tempo_seq, "phase_seq": phase_seq,
            "tempo": tempo, "beat": beat, "at": at, "hard": hard}


def play(node: str, seq: int, playing: bool, beat: float) -> dict:
    return {"type": PLAY, "node": node, "seq": seq,
            "playing": playing, "beat": beat}


def master(node: str, term: int, master_id: str) -> dict:
    return {"type": MASTER, "node": node, "term": term, "master_id": master_id}
