"""Offline diagnosis of layered beat evidence.

Everything here is a pure function of an evidence document: no live session,
no original process, no hidden defaults for missing measurements. If a
measurement is absent the corresponding metric is reported as ``None`` and a
``unmeasured`` finding is emitted - never as zero.

The evidence chain is reconstructed stage by stage:

    clock   node clock ticks (node time + wall time where available)
    filter  TimelineFilter estimate: phase error, PI adj, lock state
    map     beat-map hard jumps (re-lock / deliberate phase nudge)
    tempo   session tempo changes
    master  mastership changes
    play    play-state changes
    hit     actual trigger instants: due time, buffer span, placement
    output  measured audible onsets vs expected times

A single hit is decomposed additively into the three causes the task asks to
separate:

    output_error = map_error + schedule_late + audio_latency

- ``map_error``     : local beat map phase error vs the master grid
                      (``filter.err_beats`` at trigger time, 0 on a master).
- ``schedule_late`` : the render buffer already started after the hit's due
                      time, so placement was clamped to the buffer start.
- ``audio_latency`` : measured audible onset minus the placement instant
                      (output stage; only present when audio was measured).

Phase anomalies are attributed to a cause: a sustained tempo correction /
clock-rate offset is a gradual ``tempo_ramp``, a single phase discontinuity
is a ``phase_jump``. Each maps to a pipeline stage through the diagnostic
map, so an anomaly is always localised.
"""
from __future__ import annotations

import json
import math
import os

from .trace import evidence_integrity, read_evidence

MAPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_maps")

# Categories that affect cross-end rhythm vs. those that affect the monitor
# sound path. Each category must exist in the loaded causes map.
CROSS_END_CATEGORIES = {"map_error", "schedule_late", "phase_drift",
                        "tempo_ramp", "phase_jump", "missed_trigger"}
MONITOR_CATEGORIES = {"audio_latency", "clock_stall"}


def load_maps(path: str | None = None) -> "Maps":
    """Load the diagnostic map (thresholds + category->stage table).

    A missing, empty or partial map is a hard error: the tool must not fall
    back to built-in defaults and hide a broken deployment.
    """
    base = path or MAPS_DIR
    if os.path.isdir(base):
        files = sorted(os.path.join(base, f) for f in os.listdir(base)
                       if f.endswith(".json"))
        if not files:
            raise ValueError(f"诊断映射目录为空: {base}")
        merged: dict = {}
        for fp in files:
            with open(fp, "r", encoding="utf-8") as fh:
                part = json.load(fh)
            if not isinstance(part, dict):
                raise ValueError(f"诊断映射文件不是对象: {fp}")
            merged.update(part)
    elif os.path.isfile(base):
        with open(base, "r", encoding="utf-8") as fh:
            merged = json.load(fh)
        if not isinstance(merged, dict):
            raise ValueError(f"诊断映射文件不是对象: {base}")
    else:
        raise ValueError(f"诊断映射不存在: {base}")
    return Maps(merged, source=base)


class Maps:
    def __init__(self, raw: dict, source: str = "<memory>"):
        self.source = source
        th = raw.get("thresholds")
        causes = raw.get("causes")
        if not isinstance(th, dict) or not th:
            raise ValueError(f"诊断映射缺少 thresholds: {source}")
        if not isinstance(causes, dict) or not causes:
            raise ValueError(f"诊断映射缺少 causes: {source}")
        for key in ("ms", "phase", "clock", "evidence"):
            if not isinstance(th.get(key), dict):
                raise ValueError(f"诊断阈值缺少 '{key}': {source}")
        self.thresholds = th
        self.causes = causes

    def stage(self, category: str) -> str:
        info = self.causes.get(category)
        if info is None:
            # A symptom without a mapped stage is a broken map, not an
            # unknown: refuse to mislabel it silently.
            raise ValueError(f"诊断映射缺少类别 {category!r} 的阶段定位")
        return info["stage"]

    def label(self, category: str) -> str:
        return self.causes.get(category, {}).get("label", category)

    def ms(self, key: str) -> float:
        return float(self.thresholds["ms"][key])

    def phase(self, key: str) -> float:
        return float(self.thresholds["phase"][key])

    def clock(self, key: str) -> float:
        return float(self.thresholds["clock"][key])


# ---------------------------------------------------------------- helpers
def _num(ev: dict, key: str):
    v = ev.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _sorted_events(events: list, stage: str, kind: str | None = None) -> list:
    out = [e for e in events if e.get("stage") == stage
           and (kind is None or e.get("kind") == kind)]
    out.sort(key=lambda e: (e.get("seq", 0)))
    return out


def _percentile(values: list[float], q: float):
    if not values:
        return None
    vals = sorted(values)
    k = max(0, min(len(vals) - 1, int(math.ceil(q * len(vals))) - 1))
    return vals[k]


def _slope(xs: list[float], ys: list[float]):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def _downsample(items: list, limit: int = 32) -> list:
    if len(items) <= limit:
        return items
    step = len(items) / float(limit)
    return [items[int(i * step)] for i in range(limit)]


# ---------------------------------------------------------- stage analysis
def _analyze_clock(clock_events: list, maps: Maps) -> dict:
    ticks = []
    stable_flags = []
    for e in clock_events:
        if e.get("kind") != "tick":
            continue
        t = _num(e, "t")
        if t is None:
            continue
        ticks.append((t, _num(e, "wall"), _num(e, "frame"), _num(e, "sr"),
                      _num(e, "drift")))
        stable_flags.append(e.get("wall_stable", True))
    ticks.sort(key=lambda x: x[0])
    # Wall time is only a trustworthy reference when every tick says so. If
    # it is a best-effort driver loop, rate/stall numbers would measure the
    # driver, not the audio clock, so they must stay unmeasured.
    wall_stable = bool(stable_flags) and all(stable_flags)
    rate_ppm = None
    span = None
    max_stall_ms = None
    stall_at = None
    if wall_stable:
        if len(ticks) >= 2:
            t_first, w_first = ticks[0][0], ticks[0][1]
            t_last, w_last = ticks[-1][0], ticks[-1][1]
            if (w_first is not None and w_last is not None
                    and w_last - w_first > 0 and t_first is not None):
                span = (w_last - w_first, t_last - t_first)
                rate_ppm = ((t_last - t_first) / (w_last - w_first) - 1.0) * 1e6
        # A driver stall shows up as a wall-time gap the node clock did not
        # follow: audio was not rendered for that span.
        prev = None
        for t, w, _f, _sr, _d in ticks:
            if prev is not None and w is not None and prev[1] is not None:
                gap = (w - prev[1]) - (t - prev[0])
                if max_stall_ms is None or gap * 1000.0 > max_stall_ms:
                    max_stall_ms = gap * 1000.0
                    stall_at = t
            prev = (t, w)
    findings = []
    if max_stall_ms is not None and \
            max_stall_ms > maps.clock("stall_ms"):
        findings.append({
            "category": "clock_stall", "stage": maps.stage("clock_stall"),
            "label": maps.label("clock_stall"), "severity": "warn",
            "at": stall_at, "value_ms": max_stall_ms,
            "detail": "节点时钟未跟上墙钟（音频渲染停顿）"})
    return {"ticks": len(ticks), "rate_ppm": rate_ppm,
            "node_to_wall_span": span, "max_stall_ms": max_stall_ms,
            "wall_stable": wall_stable, "findings": findings}


def _analyze_phase(filter_events: list, map_events: list, maps: Maps) -> dict:
    series = []
    for e in filter_events:
        if e.get("kind") != "estimate":
            continue
        t = _num(e, "t")
        err = _num(e, "err_beats")
        if t is None or err is None:
            continue
        series.append({"t": t, "err_beats": err,
                       "adj": _num(e, "adj"),
                       "base_tempo": _num(e, "base_tempo"),
                       "delay_s": _num(e, "link_delay_s")})
    series.sort(key=lambda s: s["t"])
    findings = []
    slope = None
    total_change = None
    steps = []
    tempo_span = None
    if len(series) >= 2:
        xs = [s["t"] for s in series]
        ys = [s["err_beats"] for s in series]
        slope = _slope(xs, ys)
        total_change = ys[-1] - ys[0]
        tempo_span = xs[-1] - xs[0]
        jump_beats = maps.phase("jump_beats")
        for a, b in zip(series, series[1:]):
            d = b["err_beats"] - a["err_beats"]
            dt = b["t"] - a["t"]
            if abs(d) >= jump_beats and dt > 0:
                steps.append({"t": b["t"], "delta_beats": d, "dt": dt})
    # Explicit hard jumps (deliberate phase nudge or re-lock). The very first
    # lock is normal startup, not a phase anomaly, so it is excluded.
    map_jumps = [e for e in map_events if e.get("kind") == "jump"
                 and e.get("reason") not in ("initial_lock", "adopt_relock")]
    # Median tempo for converting beats to ms, only if tempo evidence exists.
    tempos = [s["base_tempo"] for s in series if s["base_tempo"]]
    tempo = sorted(tempos)[len(tempos) // 2] if tempos else None

    # A gradual drift is sustained one-directional movement of the phase
    # error that the loop keeps correcting. Transient zero-crossings of a
    # settled PLL must not count, so require the error to accumulate
    # monotonically across the window and to stay outside the band for the
    # minimum duration. A single-step discontinuity is handled as a jump.
    drift_ppm = maps.phase("drift_ppm")
    gradual = False
    one_sided_beats = 0.0
    if series and tempo_span and tempo_span >= maps.phase("ramp_min_duration_s"):
        band = maps.phase("drift_band_beats")
        prev = series[0]["err_beats"]
        for s in series[1:]:
            d = s["err_beats"] - prev
            if abs(prev) >= band and (d * prev) > 0:
                one_sided_beats += abs(d)
            prev = s["err_beats"]
        if one_sided_beats >= maps.phase("drift_min_beats"):
            gradual = True
    if slope is not None and tempo and \
            abs(slope) * 60.0 / tempo * 1e6 >= drift_ppm and \
            total_change is not None and abs(total_change) >= \
            maps.phase("drift_min_beats"):
        gradual = True
    # A sustained slope beyond the configured rate is also a gradual drift,
    # even if the one-sided accumulator stayed under its gate.
    if slope is not None and abs(slope) >= maps.phase(
            "err_slope_beats_per_s") and total_change is not None and \
            abs(total_change) >= maps.phase("drift_min_beats"):
        gradual = True
    if gradual and not map_jumps:
        findings.append({
            "category": "tempo_ramp", "stage": maps.stage("tempo_ramp"),
            "label": maps.label("tempo_ramp"), "severity": "info",
            "at": series[-1]["t"] if series else None,
            "value_beats": total_change, "slope_beats_per_s": slope,
            "detail": "相位误差缓慢累积（渐变速度/时钟漂移）"})
    if map_jumps or steps:
        steps.sort(key=lambda s: abs(s["delta_beats"]), reverse=True)
        findings.append({
            "category": "phase_jump", "stage": maps.stage("phase_jump"),
            "label": maps.label("phase_jump"), "severity": "info",
            "at": steps[0]["t"] if steps else map_jumps[0].get("t"),
            "value_beats": steps[0]["delta_beats"] if steps else
            map_jumps[0].get("beats"),
            "hard_jumps": len(map_jumps),
            "detail": "节拍相位单次突跳（重锁/相位跳变）"})
    trace = []
    for s in _downsample(series, 32):
        ms = (s["err_beats"] * 60000.0 / tempo) if tempo else None
        trace.append({"t": s["t"], "err_beats": s["err_beats"],
                      "err_ms": ms})
    return {"samples": len(series), "slope_beats_per_s": slope,
            "total_change_beats": total_change, "tempo": tempo,
            "steps": steps, "hard_jumps": len(map_jumps),
            "hard_jump_events": map_jumps,
            "trace": trace, "findings": findings}


def _analyze_tempo(tempo_events: list, maps: Maps) -> dict:
    """Distinguish a gradual tempo ramp from a one-off change. Several tempo
    changes that move in one direction are a ramp; a single change is a step.
    Ramping is a cause even when the PLL hides it from the audible grid."""
    changes = []
    for e in tempo_events:
        if e.get("kind") != "change":
            continue
        t = _num(e, "t")
        old = _num(e, "old")
        new = _num(e, "new")
        if t is None or old is None or new is None:
            continue
        changes.append({"t": t, "old": old, "new": new,
                        "adopted": bool(e.get("adopted"))})
    changes.sort(key=lambda c: c["t"])
    findings = []
    ramp = False
    if len(changes) >= 2:
        deltas = [(c["new"] - c["old"], c["t"])
                  for c in changes]
        signs = {1 if d > 0 else -1 for d, _ in deltas if d != 0}
        span = changes[-1]["t"] - changes[0]["t"]
        ramp = len(signs) == 1 and span >= maps.phase("ramp_min_duration_s")
    if ramp:
        findings.append({
            "category": "tempo_ramp", "stage": maps.stage("tempo_ramp"),
            "label": maps.label("tempo_ramp"), "severity": "info",
            "at": changes[-1]["t"],
            "value_bpm": changes[-1]["new"] - changes[0]["old"],
            "changes": len(changes),
            "detail": "速度沿同一方向连续变化（渐变速度）"})
    return {"changes": len(changes), "ramp": ramp, "findings": findings}


def _hit_components(trig: dict, maps: Maps):
    """Additive decomposition of one trigger's timing, or absent fields."""
    tempo = _num(trig, "tempo")
    due = _num(trig, "due")
    buf0 = _num(trig, "buf_t0")
    placement = _num(trig, "placement_t")
    map_err = _num(trig, "map_err_beats")
    map_ms = None
    if map_err is not None and tempo:
        map_ms = map_err * 60000.0 / tempo
    sched_ms = None
    if due is not None and buf0 is not None:
        sched_ms = max(0.0, buf0 - due) * 1000.0
    place_ms = None
    if due is not None and placement is not None:
        place_ms = (placement - due) * 1000.0
    return {"tempo": tempo, "map_ms": map_ms, "sched_ms": sched_ms,
            "place_ms": place_ms}


def _analyze_hits(hit_events: list, output_events: list,
                  maps: Maps, play_event: dict | None = None,
                  jump_times: list | None = None) -> dict:
    trigs = [e for e in hit_events if e.get("kind") == "trigger"]
    # Evidence before playback started is pre-roll: the sequencer was already
    # being scanned but nothing was supposed to sound yet. Exclude it so a
    # normal startup is not reported as late/missed.
    start_t = None
    if play_event is not None:
        start_t = _num(play_event, "t")
    if start_t is not None:
        trigs = [e for e in trigs
                 if _num(e, "buf_t0") is not None and _num(e, "buf_t0") >= start_t]
        missed_events = [e for e in hit_events if e.get("kind") == "missed"
                         and _num(e, "buf_t0") is not None
                         and _num(e, "buf_t0") >= start_t]
    else:
        missed_events = [e for e in hit_events if e.get("kind") == "missed"]
    missed = missed_events
    # A lateness that lands on the same buffer as a deliberate phase jump is
    # a consequence of the jump, not an independent scheduler fault: the jump
    # moves due times retroactively. Exclude those from schedule_late so the
    # cause is attributed once, to phase_jump.
    jump_times = jump_times or []

    def _near_jump(t):
        return t is not None and any(abs(t - jt) < 0.05 for jt in jump_times)

    map_ms, sched_ms, place_ms = [], [], []
    for trig in trigs:
        c = _hit_components(trig, maps)
        if c["map_ms"] is not None:
            map_ms.append(abs(c["map_ms"]))
        buf_t0 = _num(trig, "buf_t0")
        if c["sched_ms"] is not None and not _near_jump(buf_t0):
            sched_ms.append(abs(c["sched_ms"]))
        if c["place_ms"] is not None:
            place_ms.append(abs(c["place_ms"]))
    # Output-stage latency: measured audible onset vs expected hit time.
    latencies = []
    for ev in output_events:
        measured = _num(ev, "measured_t")
        expected = _num(ev, "expected_t")
        if measured is None or expected is None:
            continue
        latencies.append((measured - expected) * 1000.0)
    findings = []
    n = len(trigs)
    if n == 0:
        findings.append({
            "category": "unmeasured", "stage": maps.stage("unmeasured"),
            "label": maps.label("unmeasured"), "severity": "info",
            "at": None, "metric": "hit",
            "detail": "没有实际触发时刻证据，无法分解输出偏差"})
    if map_ms:
        p99 = _percentile([abs(v) for v in map_ms], 0.99)
        if p99 is not None and p99 > maps.ms("map_error"):
            findings.append({
                "category": "map_error", "stage": maps.stage("map_error"),
                "label": maps.label("map_error"), "severity": "warn",
                "at": None, "value_ms": p99, "n": len(map_ms),
                "detail": "节拍到输出时刻的映射误差超过阈值"})
    if sched_ms:
        p99 = _percentile([abs(v) for v in sched_ms], 0.99)
        if p99 is not None and p99 > maps.ms("schedule_late"):
            findings.append({
                "category": "schedule_late",
                "stage": maps.stage("schedule_late"),
                "label": maps.label("schedule_late"), "severity": "warn",
                "at": None, "value_ms": p99, "n": len(sched_ms),
                "detail": "调度迟到：缓冲区已晚于事件应触发时刻"})
    if missed:
        findings.append({
            "category": "missed_trigger",
            "stage": maps.stage("missed_trigger"),
            "label": maps.label("missed_trigger"), "severity": "warn",
            "at": None, "n": len(missed),
            "detail": "事件超出迟到容忍窗口，计入漏触发"})
    if latencies:
        p99 = _percentile([abs(v) for v in latencies], 0.99)
        if p99 is not None and p99 > maps.ms("audio_latency"):
            findings.append({
                "category": "audio_latency",
                "stage": maps.stage("audio_latency"),
                "label": maps.label("audio_latency"), "severity": "warn",
                "at": None, "value_ms": p99, "n": len(latencies),
                "detail": "实测音频输出起音相对预期时刻延迟"})
    else:
        findings.append({
            "category": "unmeasured", "stage": maps.stage("unmeasured"),
            "label": maps.label("unmeasured"), "severity": "info",
            "at": None, "metric": "output",
            "detail": "没有实测输出起音证据，音频输出延迟未测量"})
    def stat(vals):
        if not vals:
            return None
        return {"n": len(vals), "mean_ms": sum(vals) / len(vals),
                "p99_ms": _percentile([abs(v) for v in vals], 0.99),
                "max_ms": _percentile([abs(v) for v in vals], 1.0)}
    return {"triggers": n, "missed": len(missed),
            "map_error": stat(map_ms), "schedule_late": stat(sched_ms),
            "placement": stat(place_ms), "audio_latency": stat(latencies),
            "findings": findings}


def _analyze_status(events: list) -> dict:
    status = {"playing": None, "master_id": None, "is_master": None,
              "tempo": None, "last_t": None, "locked": None}
    for e in events:
        stage, kind = e.get("stage"), e.get("kind")
        t = _num(e, "t")
        if t is not None:
            status["last_t"] = t
        if stage == "play" and kind == "state":
            status["playing"] = bool(e.get("playing"))
            status["play_beat"] = _num(e, "beat")
        elif stage == "master" and kind == "change":
            status["master_id"] = e.get("master_id")
            status["is_master"] = bool(e.get("is_master"))
        elif stage == "tempo" and kind == "change":
            status["tempo"] = _num(e, "new")
        elif stage == "filter" and kind == "estimate":
            status["locked"] = bool(e.get("locked"))
    return status


def _analyze_node(node: str, snap: dict, maps: Maps) -> dict:
    events = snap.get("events", [])
    clock_events = _sorted_events(events, "clock")
    filter_events = _sorted_events(events, "filter")
    map_events = _sorted_events(events, "map")
    hit_events = _sorted_events(events, "hit")
    output_events = _sorted_events(events, "output")

    clock = _analyze_clock(clock_events, maps)
    phase = _analyze_phase(filter_events, map_events, maps)
    tempo = _analyze_tempo(_sorted_events(events, "tempo"), maps)
    play_events = _sorted_events(events, "play")
    play_start = next((e for e in play_events
                       if e.get("kind") == "state" and e.get("playing")),
                      None)
    hits = _analyze_hits(hit_events, output_events, maps, play_start,
                         [j.get("t") for j in phase["hard_jump_events"]])
    findings = list(clock["findings"]) + list(phase["findings"]) + \
        list(tempo["findings"]) + list(hits["findings"])

    # Clock-rate drift is a cause of gradual phase movement, localise it even
    # when the PLL hides it from the audible grid. Only trust it when the
    # wall reference is stable; otherwise it measures the driver, not audio.
    rate = clock["rate_ppm"]
    if clock["wall_stable"] and rate is not None and \
            abs(rate) >= maps.phase("drift_ppm"):
        findings.append({
            "category": "tempo_ramp", "stage": maps.stage("tempo_ramp"),
            "label": maps.label("tempo_ramp"), "severity": "info",
            "at": None, "value_ppm": rate,
            "detail": "节点时钟速率相对参考持续偏移（渐变漂移）"})

    # Severity is per-cause from its own threshold; one unrelated warning
    # must not silently escalate gradual-drift/jump findings.

    # Multiple evidence sources can flag the same cause (e.g. a tempo ramp is
    # visible in both the tempo stream and the clock rate). Collapse to one
    # finding per (category, stage) so an anomaly is reported once, with all
    # of its evidence sources listed.
    merged: dict = {}
    for f in findings:
        key = (f["category"], f["stage"])
        if key not in merged:
            f = dict(f)
            f["sources"] = [f.get("detail", f["category"])]
            merged[key] = f
        else:
            base = merged[key]
            base["sources"].append(f.get("detail", f["category"]))
            if f["severity"] == "warn":
                base["severity"] = "warn"
            for k, v in f.items():
                if k not in ("category", "stage", "label", "severity",
                             "detail", "sources") and base.get(k) is None:
                    base[k] = v
    findings = list(merged.values())

    attribution: dict = {}
    for f in findings:
        key = (f["category"], f["stage"])
        slot = attribution.setdefault(
            key, {"category": f["category"], "stage": f["stage"],
                  "label": f["label"], "count": 0,
                  "worst_ms": 0.0, "severity": "info"})
        slot["count"] += 1
        val = f.get("value_ms")
        if isinstance(val, (int, float)) and abs(val) > abs(slot["worst_ms"]):
            slot["worst_ms"] = float(val)
        if f["severity"] == "warn":
            slot["severity"] = "warn"
    attribution = sorted(attribution.values(),
                         key=lambda a: (a["severity"] != "warn", -a["count"]))

    return {"node": node, "status": _analyze_status(events),
            "clock": {k: clock[k] for k in
                      ("ticks", "rate_ppm", "max_stall_ms",
                       "node_to_wall_span")},
            "phase": {k: phase[k] for k in
                      ("samples", "slope_beats_per_s", "total_change_beats",
                       "tempo", "steps", "hard_jumps", "trace")},
            "tempo": {"changes": tempo["changes"], "ramp": tempo["ramp"]},
            "hits": {k: hits[k] for k in
                     ("triggers", "missed", "map_error", "schedule_late",
                      "placement", "audio_latency")},
            "findings": findings, "attribution": attribution,
            "evidence": {"recorded": snap.get("recorded"),
                         "counts": snap.get("counts"),
                         "evicted": snap.get("evicted"),
                         "skipped_by_sampling":
                             snap.get("skipped_by_sampling"),
                         "internal_errors": snap.get("internal_errors")}}


def diagnose(doc: dict, maps: Maps) -> dict:
    """Reconstruct the evidence chain and attribute anomalies to stages."""
    if not isinstance(doc, dict) or "nodes" not in doc:
        raise ValueError("不是可诊断的证据文档")
    nodes = {}
    for node, snap in doc["nodes"].items():
        nodes[node] = _analyze_node(node, snap, maps)

    categories: dict = {}
    stages: dict = {}
    alerts = []
    for node, res in nodes.items():
        for f in res["findings"]:
            if f["severity"] == "warn":
                categories[f["category"]] = categories.get(f["category"], 0) + 1
                stages[f["stage"]] = stages.get(f["stage"], 0) + 1
            alerts.append({
                "node": node, "kind": f["category"], "stage": f["stage"],
                "label": f["label"], "severity": f["severity"],
                "text": f"{node} {f['label']} → 阶段 {f['stage']}"})
        # Surface incomplete evidence: internal recorder errors are a fault;
        # rolling eviction is informational.
        ev = res["evidence"]
        if ev.get("internal_errors"):
            alerts.append({"node": node, "kind": "trace_error",
                           "stage": "trace", "label": "记录器内部错误",
                           "severity": "warn",
                           "text": f"{node} 证据记录器内部错误 "
                                   f"{ev['internal_errors']} 次"})
        # Eviction is normal for a bounded rolling window: it limits accuracy
        # of long-horizon replay but is not a fault. It is still surfaced so
        # a consumer knows the trace is not the whole session.
        if ev.get("evicted") and any(ev["evicted"].values()):
            alerts.append({"node": node, "kind": "evidence_evicted",
                           "stage": "trace", "label": "证据滚动淘汰",
                           "severity": "info",
                           "text": f"{node} 证据超出容量滚动淘汰，超出窗口的部分不可重放"})
    integrity = evidence_integrity(doc)
    cross_end = any(a["kind"] in CROSS_END_CATEGORIES
                    and a.get("severity", "warn") == "warn" for a in alerts)
    monitor = any(a["kind"] in MONITOR_CATEGORIES
                  and a.get("severity", "warn") == "warn" for a in alerts)
    return {
        "ok": not any(a.get("severity", "warn") == "warn" for a in alerts),
        "maps_source": maps.source,
        "nodes": nodes,
        "summary": {
            "warnings": sum(1 for a in alerts
                            if a.get("severity", "warn") == "warn"),
            "infos": sum(1 for a in alerts
                         if a.get("severity") == "info"),
            "categories": categories,
            "stages": stages,
            "cross_end_rhythm_anomaly": cross_end,
            "monitor_sound_anomaly": monitor,
            "evidence_complete": integrity["complete"],
            "alerts": alerts,
        },
    }


def diagnose_file(path: str, maps_path: str | None = None) -> dict:
    return diagnose(read_evidence(path), load_maps(maps_path))


# ---------------------------------------------------- constructed scenarios
def _emit_output_evidence(app, rec, buffers: list, sr: int) -> None:
    """Pair rendered-audio onsets with engine placements and write output
    evidence. Runs after rendering, observes only."""
    import numpy as np

    from .measure import find_onsets

    if not buffers:
        return
    x = np.concatenate(buffers)
    onsets = find_onsets(x, sr)
    placements = list(app.engine.placed)
    if not onsets.size or not placements:
        return
    eng = app.engine
    scale = (1.0 + eng.drift) / sr
    onset_t = eng.t0 + onsets * scale
    pr_t = np.array([p[3] for p in placements])
    used = set()
    for ot in onset_t:
        k = int(np.argmin(np.abs(pr_t - ot)))
        if abs(pr_t[k] - ot) > 0.06 or k in used:
            continue
        used.add(k)
        beat = placements[k][0]
        rec.record("output", "onset", beat=beat,
                   expected_t=float(pr_t[k]), measured_t=float(ot),
                   latency_ms=float((ot - pr_t[k]) * 1000.0))


def run_scenario(seconds: float = 12.0, sr: int = 4000, dt: float = 0.01,
                 tempo: float = 120.0, drift_ppm: float = 0.0,
                 jump_beats: float = 0.0, jump_at: float | None = None,
                 tempo_end: float | None = None, tempo_steps: int = 0,
                 seed: int = 7, play_at: float = 0.5,
                 jitter: float = 0.0, loss: float = 0.0,
                 name: str = "scenario") -> dict:
    """Run a deterministic in-process scenario and capture its evidence.

    This is used to construct controlled slow-drift, gradual-tempo and
    phase-jump inputs. It uses fresh engine instances (never the original
    process) and the virtual network, so the emitted evidence is standalone
    and reproducible.
    """
    from .app import BeatApp
    from .measure import find_onsets
    from .trace import TraceRecorder, build_evidence
    from .transport import VirtualNetwork

    net = VirtualNetwork(seed=seed)
    net.set_all_links(delay=0.001, jitter=jitter, loss=loss)
    apps: dict = {}
    traces: dict = {}
    caps = {"clock": 2048, "filter": 2048, "map": 256, "tempo": 128,
            "master": 64, "play": 32, "hit": 2048, "output": 512}
    every = {"clock": 10, "filter": 5}
    for i, nid in enumerate(("A", "B", "C")):
        tr = net.connect(nid)
        drift = 0.0 if nid == "A" else drift_ppm
        app = BeatApp(nid, nid, tr, node_index=i, sr=sr, drift_ppm=drift,
                      t0=1000.0 * i, quantum=4.0)
        rec = TraceRecorder(nid, capacities=caps, sample_every=every)
        app.enable_trace(rec)
        app.engine.wall_clock = (lambda n=net: n.t)
        tr.clock = app.engine
        apps[nid], traces[nid] = app, rec
    audio: dict = {nid: [] for nid in apps}
    n = int(round(dt * sr))
    steps = int(round(seconds / dt))
    if jump_beats and jump_at is None:
        jump_at = seconds / 2.0
    played = jumped = t_changed = False
    master = apps["A"]
    ramp = tempo_end is not None and tempo_steps > 0
    ramp_i = 0
    ramp_dt = (seconds / tempo_steps) if ramp else None
    for _ in range(steps):
        net.advance(dt)
        if not t_changed and master.session.is_master:
            master.session.set_tempo(tempo)
            t_changed = True
        if (ramp and t_changed and ramp_i < tempo_steps
                and net.t >= play_at + (ramp_i + 1) * ramp_dt):
            ramp_i += 1
            frac = ramp_i / float(tempo_steps)
            master.session.set_tempo(tempo + (tempo_end - tempo) * frac)
        if not played and net.t >= play_at and master.session.is_master:
            master.session.request_play(True)
            played = True
        if (jump_beats and jump_at is not None and not jumped
                and net.t >= jump_at and master.session.is_master):
            master.session.nudge_phase(jump_beats)
            jumped = True
        for nid, app in apps.items():
            audio[nid].append(app.tick(n))
    for nid, app in apps.items():
        _emit_output_evidence(app, traces[nid], audio[nid], sr)
    meta = {
        "name": name, "seconds": seconds, "sr": sr, "dt": dt, "tempo": tempo,
        "drift_ppm": drift_ppm, "jump_beats": jump_beats,
        "jump_at": jump_at, "tempo_end": tempo_end,
        "tempo_steps": tempo_steps, "seed": seed, "play_at": play_at,
        "jitter": jitter, "loss": loss,
        "generator": "interbeat.diagnostics.run_scenario",
    }
    return build_evidence(meta, {nid: rec.snapshot()
                                 for nid, rec in traces.items()})


def concise(result: dict) -> str:
    """One-screen human summary of a diagnosis."""
    lines = [f"证据映射: {result.get('maps_source')}",
             f"整体: {'正常' if result.get('ok') else '存在异常'}"]
    s = result.get("summary", {})
    lines.append(f"告警 {s.get('warnings', 0)} 条，提示 {s.get('infos', 0)} 条；"
                 f"跨端节奏异常={s.get('cross_end_rhythm_anomaly')}；"
                 f"监听声音异常={s.get('monitor_sound_anomaly')}")
    for node, res in result.get("nodes", {}).items():
        st = res["status"]
        hits = res["hits"]
        phase = res["phase"]
        lines.append(
            f"[{node}] playing={st.get('playing')} master={st.get('is_master')} "
            f"clock={res['clock'].get('rate_ppm')}ppm "
            f"斜率={phase.get('slope_beats_per_s')} beats/s "
            f"触发={hits.get('triggers')} 漏={hits.get('missed')}")
        for a in res.get("attribution", []):
            lines.append(f"    - {a['label']} @{a['stage']} "
                         f"{a['severity']} x{a['count']}")
    for alert in s.get("alerts", []):
        if alert.get("severity", "warn") == "warn":
            lines.append(f"  ! {alert['text']}")
    for alert in s.get("alerts", []):
        if alert.get("severity") == "info":
            lines.append(f"  · {alert['text']}")
    return "\n".join(lines)
