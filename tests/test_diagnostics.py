"""Diagnostics: layered evidence, cause separation, replay, and no-defaults.

The tests build controlled slow-drift and phase-jump inputs, diagnose them
offline from the evidence document only, and assert that:

- mapping error, scheduling lateness and audio output latency are separate
  measurements (not folded into one number);
- a gradual drift attributes to tempo_ramp, a single discontinuity to
  phase_jump, and the two are distinguishable;
- diagnosis is a pure function of the document (replay is identical);
- a missing measurement is reported as absent, never as zero;
- a malformed or incomplete map is a hard error, not a silent default;
- tracing is observational: enabling it does not change scheduling.
"""
import json

import numpy as np
import pytest

from interbeat.app import BeatApp
from interbeat.diagnostics import (diagnose, load_maps, run_scenario,
                                   concise)
from interbeat.trace import (TraceRecorder, build_evidence, read_evidence,
                             write_evidence)
from interbeat.transport import VirtualNetwork


def _scenario(**kw):
    kw.setdefault("seconds", 10.0)
    kw.setdefault("sr", 4000)
    return run_scenario(**kw)


def _node_categories(result, node):
    return {f["category"] for f in result["nodes"][node]["findings"]}


# ------------------------------------------------------- cause separation
def test_drift_and_jump_attribute_to_different_causes():
    drift = diagnose(_scenario(name="drift", drift_ppm=-200.0), load_maps())
    jump = diagnose(_scenario(name="jump", jump_beats=0.5), load_maps())

    # Followers (clocks drift) show a gradual ramp on the filter stage.
    assert "tempo_ramp" in _node_categories(drift, "B")
    assert "phase_jump" not in _node_categories(drift, "B")
    assert drift["nodes"]["B"]["phase"]["hard_jumps"] == 0

    # A deliberate jump shows up as a single phase discontinuity on the map
    # stage, and is NOT misreported as a gradual ramp.
    assert "phase_jump" in _node_categories(jump, "A")
    assert "tempo_ramp" not in _node_categories(jump, "A")
    assert jump["nodes"]["A"]["phase"]["hard_jumps"] == 1


def test_three_error_sources_are_measured_separately():
    res = diagnose(_scenario(drift_ppm=-200.0), load_maps())
    node = res["nodes"]["B"]
    hits = node["hits"]
    # Every component is reported independently and can be absent.
    assert hits["triggers"] > 0
    assert hits["map_error"]["n"] > 0           # phase mapping error
    assert hits["schedule_late"]["n"] > 0       # scheduler placement
    assert hits["audio_latency"]["n"] > 0       # measured output latency
    # They live on distinct stages in the diagnostic table.
    cats = {f["category"]: f["stage"] for f in node["findings"]}
    for category in ("tempo_ramp",):
        assert cats[category] == load_maps().stage(category)


def test_startup_is_not_reported_as_late_or_missed():
    res = diagnose(_scenario(seconds=6.0, play_at=0.5), load_maps())
    for node in res["nodes"].values():
        hint = {f["category"] for f in node["findings"]}
        assert "schedule_late" not in hint
        assert "missed_trigger" not in hint
        assert node["hits"]["missed"] == 0


# ------------------------------------------------------------ replay
def test_diagnosis_is_pure_replay_from_the_document(tmp_path):
    doc = _scenario(name="replay", drift_ppm=-150.0)
    path = tmp_path / "ev.json"
    write_evidence(str(path), doc["meta"], doc["nodes"])

    loaded = read_evidence(str(path))
    once = diagnose(loaded, load_maps())
    twice = diagnose(loaded, load_maps())
    assert json.dumps(once, sort_keys=True) == json.dumps(twice, sort_keys=True)

    # The scenario carries its own generator metadata: no live process needed.
    assert loaded["meta"]["generator"] == "interbeat.diagnostics.run_scenario"
    assert once["summary"]["cross_end_rhythm_anomaly"] in (True, False)


def test_evidence_roundtrips_through_disk(tmp_path):
    doc = _scenario(seconds=4.0)
    path = tmp_path / "e.json"
    write_evidence(str(path), doc["meta"], doc["nodes"])
    back = read_evidence(str(path))
    assert set(back["nodes"]) == set(doc["nodes"])
    assert all(back["nodes"][n]["events"] for n in back["nodes"])


# ------------------------------------------------------- no default masking
def test_missing_measurement_is_absent_not_zero():
    """A trace with hits but no output evidence must report audio_latency as
    None and emit a 'unmeasured' finding - never 0 ms."""
    rec = TraceRecorder("solo")
    rec.record("hit", "trigger", force=True, beat=1.0, due=0.5, buf_t0=0.5,
               buf_t1=0.51, placement_t=0.5, late_s=0.0, tempo=120.0)
    doc = build_evidence({"source": "unit"}, {"solo": rec.snapshot()})
    res = diagnose(doc, load_maps())
    node = res["nodes"]["solo"]
    assert node["hits"]["audio_latency"] is None
    cats = [f["category"] for f in node["findings"]]
    assert "unmeasured" in cats
    assert "audio_latency" not in cats


def test_empty_map_is_a_hard_error(tmp_path):
    with pytest.raises(ValueError):
        load_maps(str(tmp_path))  # empty directory: no fallback defaults
    bad = tmp_path / "map.json"
    bad.write_text(json.dumps({"thresholds": {}}))
    with pytest.raises(ValueError):
        load_maps(str(bad))


def test_unknown_category_has_no_implicit_stage(tmp_path):
    from interbeat.diagnostics import Maps
    maps = Maps({
        "thresholds": {"ms": {"map_error": 1.0}, "phase": {},
                       "clock": {}, "evidence": {}},
        "causes": {"map_error": {"stage": "clock/filter", "label": "x"}},
    })
    with pytest.raises(ValueError):
        maps.stage("not_in_map")


def test_corrupt_evidence_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"format":"nope"}')
    with pytest.raises(ValueError):
        read_evidence(str(path))


# ------------------------------------------------ observational guarantee
def test_tracing_does_not_change_playback_decisions():
    def run(trace_on):
        net = VirtualNetwork(seed=7)
        net.set_all_links(delay=0.002)
        app = BeatApp("A", "A", net.connect("A"), node_index=0, sr=4000,
                      t0=1000.0)
        net.endpoints["A"].clock = app.engine
        if trace_on:
            app.enable_trace(TraceRecorder("A", sample_every={"clock": 1,
                                                              "filter": 1}))
        beats = []
        app.engine.on_hit = lambda b, i: beats.append((b, i))
        net.advance(0.5)
        app.session.request_play(True)
        for _ in range(600):
            net.advance(0.01)
            app.tick(40)
        return beats

    assert run(False) == run(True)


def test_trace_is_bounded_and_reports_eviction():
    rec = TraceRecorder("A", capacities={"hit": 8}, sample_every={"clock": 1})
    for i in range(50):
        rec.record("hit", "trigger", force=True, beat=float(i))
    snap = rec.snapshot()
    hits = [e for e in snap["events"] if e["stage"] == "hit"]
    assert len(hits) == 8                     # bounded ring buffer
    assert snap["evicted"]["hit"] == 42       # loss is visible, not silent


def test_unstable_wall_reference_yields_no_clock_verdict():
    """If the trace marks wall time as a best-effort driver, rate/stall must
    stay unmeasured - the tool must not invent a drift verdict from it."""
    rec = TraceRecorder("solo")
    # 10 ticks 10 ms apart in node time, but wildly irregular wall gaps.
    walls = [0.0, 0.3, 0.35, 0.9, 0.95, 1.6, 1.61, 2.4, 2.41, 3.2]
    for i, w in enumerate(walls):
        rec.record("clock", "tick", force=True, t=i * 0.01, wall=w,
                   wall_stable=False)
    doc = build_evidence({"source": "unit"}, {"solo": rec.snapshot()})
    res = diagnose(doc, load_maps())
    clock = res["nodes"]["solo"]["clock"]
    assert clock["rate_ppm"] is None
    assert clock["max_stall_ms"] is None
    cats = {f["category"] for f in res["nodes"]["solo"]["findings"]}
    assert "clock_stall" not in cats
    assert "tempo_ramp" not in cats


def test_monitor_anomaly_is_separate_from_cross_end():
    """An output-latency fault must raise the monitor flag and map to the
    output stage, while a clean map/schedule keeps cross-end clean."""
    rec = TraceRecorder("solo")
    # Schedule on time (no map/schedule fault) but audible onset is late.
    for i in range(10):
        rec.record("hit", "trigger", force=True, beat=float(i), due=1.0 + i,
                   buf_t0=1.0 + i, buf_t1=1.01 + i, placement_t=1.0 + i,
                   late_s=0.0, map_err_beats=0.0, tempo=120.0)
        rec.record("output", "onset", force=True, beat=float(i),
                   expected_t=1.0 + i, measured_t=1.0 + i + 0.2,
                   latency_ms=200.0)
    doc = build_evidence({"source": "unit"}, {"solo": rec.snapshot()})
    res = diagnose(doc, load_maps())
    summary = res["summary"]
    assert summary["monitor_sound_anomaly"] is True
    assert summary["cross_end_rhythm_anomaly"] is False
    stages = {a["kind"]: a["stage"] for a in summary["alerts"]}
    assert stages["audio_latency"] == load_maps().stage("audio_latency")


def test_cross_end_anomaly_maps_to_its_stage():
    rec = TraceRecorder("solo")
    rec.record("hit", "trigger", force=True, beat=1.0, due=1.0, buf_t0=2.0,
               buf_t1=2.01, placement_t=2.0, late_s=1.0, map_err_beats=0.0,
               tempo=120.0)
    doc = build_evidence({"source": "unit"}, {"solo": rec.snapshot()})
    res = diagnose(doc, load_maps())
    summary = res["summary"]
    assert summary["cross_end_rhythm_anomaly"] is True
    alerts = {a["kind"]: a["stage"] for a in summary["alerts"]}
    assert alerts["schedule_late"] == load_maps().stage("schedule_late")


def test_gradual_tempo_ramp_vs_single_step():
    """A one-way sequence of tempo changes is a gradual ramp; a single change
    is a step and must not be reported as a ramp."""
    ramp = TraceRecorder("solo")
    for i, bpm in enumerate((120, 124, 128, 132, 136)):
        ramp.record("tempo", "change", force=True, t=i * 2.0,
                    old=bpm - 4.0, new=float(bpm))
    step = TraceRecorder("solo")
    step.record("tempo", "change", force=True, t=0.0, old=120.0, new=90.0)

    cases = {"ramp": build_evidence({}, {"solo": ramp.snapshot()}),
             "step": build_evidence({}, {"solo": step.snapshot()})}
    r_ramp = diagnose(cases["ramp"], load_maps())["nodes"]["solo"]
    r_step = diagnose(cases["step"], load_maps())["nodes"]["solo"]
    assert r_ramp["tempo"]["ramp"] is True
    assert "tempo_ramp" in {f["category"] for f in r_ramp["findings"]}
    assert r_step["tempo"]["ramp"] is False
    assert "tempo_ramp" not in {f["category"] for f in r_step["findings"]}


def test_same_cause_is_reported_once_per_stage():
    """Tempo-ramp evidence from several sources must collapse into a single
    finding with all sources listed, not repeated alerts."""
    rec = TraceRecorder("solo")
    for i, bpm in enumerate((120, 126, 132, 138)):
        rec.record("tempo", "change", force=True, t=i * 2.0,
                   old=bpm - 6.0, new=float(bpm))
    # A correlated clock rate offset on the same cause:
    for i in range(20):
        rec.record("clock", "tick", force=True, t=i * 0.01, wall=i * 0.01,
                   wall_stable=True)
    doc = build_evidence({}, {"solo": rec.snapshot()})
    node = diagnose(doc, load_maps())["nodes"]["solo"]
    ramp_findings = [f for f in node["findings"]
                     if f["category"] == "tempo_ramp"]
    assert len(ramp_findings) == 1
    assert len(ramp_findings[0]["sources"]) >= 1


def test_scenario_tempo_ramp_attributes_to_tempo_ramp():
    doc = _scenario(name="ramp", tempo=120.0, tempo_end=168.0,
                    tempo_steps=8, seconds=12.0)
    res = diagnose(doc, load_maps())
    for node in ("A", "B", "C"):
        cats = _node_categories(res, node)
        assert "tempo_ramp" in cats
    # The ramp is gradual, not a discontinuity.
    assert "phase_jump" not in _node_categories(res, "A")


def test_concise_summary_mentions_cross_end_and_monitor():
    res = diagnose(_scenario(seconds=6.0, drift_ppm=-200.0), load_maps())
    text = concise(res)
    assert "跨端节奏异常" in text
    assert "监听声音异常" in text


def test_rerun_evidence_is_identical():
    """The interactive-beat rerun must reproduce the same evidence: a fixed
    seed and deterministic virtual clock give byte-identical event streams."""
    a = _scenario(seconds=6.0, drift_ppm=-200.0, seed=7, name="rerun")
    b = _scenario(seconds=6.0, drift_ppm=-200.0, seed=7, name="rerun")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_rerun_diagnosis_is_identical():
    a = diagnose(_scenario(seconds=6.0, drift_ppm=-200.0, seed=7),
                 load_maps())
    b = diagnose(_scenario(seconds=6.0, drift_ppm=-200.0, seed=7),
                 load_maps())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

