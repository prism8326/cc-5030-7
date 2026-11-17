"""Layered, bounded execution tracing for the cross-application beat system.

The trace records the evidence chain behind a playback decision without
participating in it:

    clock sample -> filter estimate -> beat map -> schedule -> render -> output

Recording is strictly observational: it only reads state that the audio
callback already computed and appends plain dicts to bounded ring buffers.
It never steers the beat map, changes scheduling, or removes events.

Cost is bounded in three independent ways:

- per-stage ring buffers (oldest evidence is evicted),
- per-stage sampling for the high-rate clock/filter stages,
- a global recorded-event budget (optional).

Evictions, sampling and any internal recorder failure are reported in the
snapshot, so a consumer cannot mistake an incomplete trace for a complete
one. Missing measurements are represented by absent fields, never by zero.
"""
from __future__ import annotations

import json
import threading
from collections import deque

EVIDENCE_FORMAT = "interbeat-evidence"
EVIDENCE_VERSION = 1

STAGES = ("clock", "filter", "map", "tempo", "master", "play",
          "schedule", "render", "output")

# Bounded per-stage retention. Trigger stages are the dense ones; control
# stages are sparse. These caps keep a multi-minute trace well under a few
# hundred kilobytes per app.
DEFAULT_CAPACITIES = {
    "clock": 256,
    "filter": 256,
    "map": 128,
    "tempo": 128,
    "master": 32,
    "play": 32,
    "schedule": 1024,
    "render": 1024,
    "output": 512,
}
DEFAULT_CAPACITY = 256

# High-rate stages are sampled by default; decision-critical events pass
# force=True and are always kept.
DEFAULT_SAMPLE_EVERY = {"clock": 5, "filter": 5}


def _clean(value):
    """Make a field JSON-safe without inventing a value for it."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


class TraceRecorder:
    """Bounded evidence recorder (thread-safe, observational only)."""

    def __init__(self, node_id: str | None = None, capacities: dict | None = None,
                 sample_every: dict | None = None, max_events: int | None = None,
                 enabled: bool = True):
        self.node_id = node_id
        self.enabled = enabled
        self.max_events = max_events
        caps = dict(DEFAULT_CAPACITIES)
        caps.update(capacities or {})
        self._capacities = caps
        every = dict(DEFAULT_SAMPLE_EVERY)
        every.update(sample_every or {})
        self._sample_every = {k: max(1, int(v)) for k, v in every.items()}
        self._lock = threading.Lock()
        self._bufs: dict[str, deque] = {}
        self._seq = 0
        self._counts: dict[str, int] = {}
        self._skipped: dict[str, int] = {}
        self._evicted: dict[str, int] = {}
        self._recorded = 0
        self._internal_errors = 0

    def record(self, stage: str, kind: str, *, force: bool = False,
               **fields) -> None:
        """Append one evidence event. Never raises into the caller."""
        if not self.enabled:
            return
        try:
            with self._lock:
                seen = self._counts.get(stage, 0) + 1
                self._counts[stage] = seen
                every = self._sample_every.get(stage, 1)
                if not force and every > 1 and seen % every:
                    self._skipped[stage] = self._skipped.get(stage, 0) + 1
                    return
                if self.max_events is not None and self._recorded >= self.max_events:
                    self._evicted[stage] = self._evicted.get(stage, 0) + 1
                    return
                self._seq += 1
                ev = {"seq": self._seq, "stage": stage, "kind": kind}
                if self.node_id is not None:
                    ev["node"] = self.node_id
                for key, value in fields.items():
                    ev[key] = _clean(value)
                buf = self._bufs.get(stage)
                if buf is None:
                    buf = deque(maxlen=self._capacities.get(stage,
                                                            DEFAULT_CAPACITY))
                    self._bufs[stage] = buf
                if len(buf) == buf.maxlen:
                    self._evicted[stage] = self._evicted.get(stage, 0) + 1
                buf.append(ev)
                self._recorded += 1
        except Exception:
            # Diagnostics must never disturb audio. The failure is surfaced
            # in the snapshot instead of being silently swallowed.
            self._internal_errors += 1

    def snapshot(self, recent: int | None = None) -> dict:
        """Return a JSON-safe snapshot. `recent` limits each stage's tail,
        which keeps periodic UI diagnosis cheap on long-running sessions."""
        with self._lock:
            events = []
            for buf in self._bufs.values():
                events.extend(buf if recent is None else list(buf)[-recent:])
            events.sort(key=lambda e: e["seq"])
            return {
                "node": self.node_id,
                "enabled": self.enabled,
                "events": events,
                "recorded": self._recorded,
                "counts": dict(self._counts),
                "skipped_by_sampling": dict(self._skipped),
                "evicted": dict(self._evicted),
                "internal_errors": self._internal_errors,
                "sample_every": {k: v for k, v in self._sample_every.items()
                                 if v > 1},
                "capacity": dict(self._capacities),
            }

    def clear(self) -> None:
        with self._lock:
            for buf in self._bufs.values():
                buf.clear()
            self._counts.clear()
            self._skipped.clear()
            self._evicted.clear()
            self._recorded = 0


def build_evidence(meta: dict, traces: dict) -> dict:
    """Assemble a replayable evidence document from recorder snapshots."""
    return {
        "format": EVIDENCE_FORMAT,
        "version": EVIDENCE_VERSION,
        "meta": dict(meta),
        "nodes": traces,
    }


def write_evidence(path: str, meta: dict, traces: dict) -> dict:
    doc = build_evidence(meta, traces)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(doc, fp, ensure_ascii=False, sort_keys=True, indent=1)
        fp.write("\n")
    return doc


def read_evidence(path: str) -> dict:
    """Load an evidence document. Raises ValueError on an unusable file -
    the tool never substitutes defaults for missing or malformed evidence."""
    with open(path, "r", encoding="utf-8") as fp:
        doc = json.load(fp)
    if not isinstance(doc, dict):
        raise ValueError("evidence root is not an object")
    if doc.get("format") != EVIDENCE_FORMAT:
        raise ValueError(f"unknown evidence format {doc.get('format')!r}")
    if doc.get("version") != EVIDENCE_VERSION:
        raise ValueError(f"unsupported evidence version {doc.get('version')!r}")
    nodes = doc.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("evidence has no nodes")
    for node, snap in nodes.items():
        if not isinstance(snap, dict) or not isinstance(snap.get("events"), list):
            raise ValueError(f"node {node!r} has no event list")
    return doc


def evidence_integrity(doc: dict) -> dict:
    """Summarise retention losses so a consumer can tell whether a trace is
    complete. Absent counters are reported as unknown, not as zero."""
    evicted = {}
    skipped = {}
    errors = {}
    for node, snap in doc.get("nodes", {}).items():
        evicted[node] = snap.get("evicted")
        skipped[node] = snap.get("skipped_by_sampling")
        errors[node] = snap.get("internal_errors")
    return {
        "complete": all(not any((evicted.get(n) or {}).values())
                        and not any((skipped.get(n) or {}).values())
                        for n in doc.get("nodes", {})),
        "evicted": evicted,
        "skipped_by_sampling": skipped,
        "internal_errors": errors,
    }
