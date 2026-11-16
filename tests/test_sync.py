"""End-to-end sync scenarios on the deterministic virtual network.

Timing model: every world step advances true time by dt=10ms; each app
renders dt*sr frames per step, so sample s of an app's concatenated output
corresponds to true time s/sr. Cross-app onset deviations are measured on
the synthesized audio, not on internal state.
"""
import math

import numpy as np
import pytest

from conftest import World

SR = 8000
TOL_P99_MS = 25.0
TOL_MAX_MS = 60.0


def make_trio(w):
    a = w.add_app("A", 0, t0=0.0)
    b = w.add_app("B", 1, t0=1000.0)
    c = w.add_app("C", 2, t0=2000.0)
    return a, b, c


def test_late_join_enters_on_quantized_boundary():
    w = World(sr=SR, delay=0.002)
    a, b, _ = make_trio(w)
    w.run(1.0)
    qb = a.session.request_play(True)
    assert qb == 4.0
    w.run(14.3)  # true t ~= 15.3 s -> shared beat ~30.6 at 120 bpm
    join_beat = a.session.beatmap.beat_at(a.engine.now())

    c2 = w.add_app("D", 2, t0=5000.0)  # late joiner, unsynced local clock
    w.run(8.0)

    first_beat = c2.engine.hit_log[0][0]
    # entered at an explicit quantization boundary after the join beat,
    # not at beat 1, and not immediately on packet receipt
    assert first_beat % 4.0 == 0.0
    assert first_beat > join_beat
    assert first_beat >= 4.0  # session was already past beat 30

    d = w.deviation("A", "D", t_from=w.t - 5.0)
    assert d["n"] > 5
    assert d["p99_ms"] < TOL_P99_MS


def test_same_phase_different_bar_positions():
    w = World(sr=SR, delay=0.002)
    a, b, c = make_trio(w)
    w.run(1.0)
    a.session.request_play(True)
    w.run(25.0)

    # shared beat grid: all apps agree on the beat position itself
    for app in (a, b, c):
        beat = app.session.beatmap.beat_at(app.engine.now())
        assert 51.5 < beat < 52.5
    # but bar positions differ because phrase lengths differ (4/6/8)
    for app, ph in ((a, 4), (b, 6), (c, 8)):
        beat = app.session.beatmap.beat_at(app.engine.now())
        bar = math.floor(beat / ph)
        pos = beat % ph
        app._check = (bar, pos)
    bars = {app._check[0] for app in (a, b, c)}
    assert len(bars) == 3  # three different bar indices on one shared beat

    # coincident hits still land together
    d = w.deviation("A", "B", t_from=w.t - 10.0)
    assert d["p99_ms"] < TOL_P99_MS
    d = w.deviation("A", "C", t_from=w.t - 10.0)
    assert d["p99_ms"] < TOL_P99_MS


def test_jitter_loss_reorder_keeps_grid():
    w = World(sr=SR, delay=0.002, jitter=0.030, loss=0.05, reorder=0.15)
    a, b, c = make_trio(w)
    w.run(1.0)
    a.session.request_play(True)
    w.run(60.0)

    for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
        d = w.deviation(x, y, t_from=w.t - 30.0)
        assert d["n"] > 20, (x, y, d)
        assert d["p99_ms"] < TOL_P99_MS, (x, y, d)
        assert d["max_ms"] < TOL_MAX_MS, (x, y, d)

    # no dropped beats: app A pattern has 8 distinct onsets per 2 s
    on = w.onsets("A", t_from=w.t - 10.0)
    assert 36 <= len(on) <= 44


def test_clock_drift_stays_bounded_closed_loop():
    w = World(sr=SR, delay=0.002, jitter=0.010)
    w.add_app("A", 0, drift_ppm=-100.0)
    w.add_app("B", 1, drift_ppm=0.0, t0=1000.0)
    w.add_app("C", 2, drift_ppm=100.0, t0=2000.0)
    w.run(1.0)
    w.apps["A"].session.request_play(True)
    w.run(150.0)

    for x, y in (("A", "B"), ("A", "C")):
        d = w.deviation(x, y, t_from=w.t - 30.0)
        assert d["p99_ms"] < TOL_P99_MS, (x, y, d)


def test_simultaneous_start_alone_cannot_hold_tempo():
    """Control experiment: same drift, but the correction loop is disabled.
    Starting together does not keep the grid together - deviation grows
    linearly with the clock drift."""
    w = World(sr=SR, delay=0.002)
    w.add_app("A", 0, drift_ppm=-100.0)
    w.add_app("B", 1, drift_ppm=100.0, t0=1000.0)
    w.run(1.0)
    for app in w.apps.values():
        app.session.filter.kp = 0.0
        app.session.filter.ki = 0.0
        app.session.filter.max_adj = 0.0
    w.apps["A"].session.request_play(True)
    w.run(120.0)

    d = w.deviation("A", "B", t_from=w.t - 20.0)
    # 200 ppm relative drift over ~2 min -> tens of ms of divergence
    assert d["max_ms"] > 15.0


def test_master_switch_and_tempo_change():
    w = World(sr=SR, delay=0.002, jitter=0.010)
    a, b, c = make_trio(w)
    w.run(1.0)
    a.session.request_play(True)
    w.run(4.0)
    assert a.session.is_master

    # follower may not set tempo: rejection feeds back as an event
    assert not c.session.set_tempo(90.0)
    assert any(k == "rejected" for k, _ in c.events)

    b.session.request_master("B")
    w.run(2.0)
    for app in (a, b, c):
        assert app.session.master_id == "B"
    assert b.session.is_master and not a.session.is_master

    assert b.session.set_tempo(96.0)
    w.run(8.0)
    for app in (a, b, c):
        assert 90.0 < app.session.beatmap.tempo < 102.0

    # grid continuity across the handover: A's hits keep coming ~4/s
    on = w.onsets("A", t_from=4.0)
    on = on[on <= 7.0]
    assert 10 <= len(on) <= 14

    for x, y in (("A", "B"), ("A", "C")):
        d = w.deviation(x, y, t_from=w.t - 5.0)
        assert d["p99_ms"] < TOL_P99_MS, (x, y, d)


def test_master_crash_reelection():
    w = World(sr=SR, delay=0.002)
    a, b, c = make_trio(w)
    w.run(1.0)
    a.session.request_play(True)
    w.run(4.0)
    w.remove_app("A")  # master vanishes (crash)
    w.run(4.0)         # MASTER_TIMEOUT is 2 s
    assert b.session.master_id == "B"
    assert c.session.master_id == "B"
    assert b.session.is_master
    assert any(k == "master_timeout" for k, _ in b.events)
    w.run(6.0)
    d = w.deviation("B", "C", t_from=w.t - 4.0)
    assert d["p99_ms"] < TOL_P99_MS


def test_start_stop_quantized_on_shared_grid():
    w = World(sr=SR, delay=0.002)
    a, b, c = make_trio(w)
    w.run(1.0)
    a.session.request_play(True)
    w.run(9.3)  # t = 10.3 -> shared beat ~20.6
    stop_beat = a.session.request_play(False)
    assert stop_beat == 24.0
    w.run(4.0)

    # beat 24 at 120 bpm ~= true t=12 s; nothing may sound after the tail
    for app_id in ("A", "B", "C"):
        on = w.onsets(app_id, t_from=12.6)
        assert len(on) == 0, app_id

    w.run(1.0)  # t ~ 15.3
    start_beat = b.session.request_play(True)
    assert start_beat == 32.0
    w.run(6.0)

    # all apps restart at the same shared beat
    for app in (a, b, c):
        beats = [bb for bb, _ in app.engine.hit_log if bb >= start_beat - 1e-9]
        assert beats and beats[0] == start_beat

    d = w.deviation("A", "C", t_from=w.t - 4.0)
    assert d["p99_ms"] < TOL_P99_MS


def test_reordered_play_messages_converge():
    w = World(sr=SR, delay=0.001, jitter=0.02, reorder=0.6)
    a, b, c = make_trio(w)
    w.run(1.0)
    a.session.request_play(True)
    w.run(3.0)
    a.session.request_play(False)
    w.run(3.0)
    b.session.request_play(True)
    w.run(5.0)
    # despite heavy reordering, Lamport play_seq makes all nodes converge
    for app in (a, b, c):
        assert app.session.playing
        assert app.session.play_beat == b.session.play_beat
        assert app.session.play_seq == b.session.play_seq


def test_bounded_work_and_queues():
    import time
    w = World(sr=SR, delay=0.002, jitter=0.02)
    make_trio(w)
    w.run(1.0)
    w.apps["A"].session.request_play(True)
    w.run(5.0)
    for app in w.apps.values():
        assert len(app.engine.voices) <= 32
        assert len(app.session.filter.samples) <= 64
        assert app.engine.overflows == 0
    # render cost per 10 ms buffer stays far below the 10 ms budget
    app = w.apps["A"]
    t0 = time.perf_counter()
    for _ in range(500):
        app.tick(80)
    per_call_ms = (time.perf_counter() - t0) / 500 * 1000.0
    assert per_call_ms < 5.0
