import math

from interbeat.timeline import BeatMap, Timeline, TimelineFilter


def test_timeline_roundtrip():
    tl = Timeline(b0=4.0, t0=2.0, tempo=120.0)  # 0.5 s per beat
    assert tl.beat_at(2.0) == 4.0
    assert tl.beat_at(3.0) == 6.0
    assert tl.time_at(6.0) == 3.0
    assert abs(tl.time_at(tl.beat_at(7.77)) - 7.77) < 1e-12


def test_beatmap_steer_is_continuous():
    bm = BeatMap(120.0)
    before = bm.beat_at(10.0)
    bm.steer_to(10.0, 130.0)
    assert abs(bm.beat_at(10.0) - before) < 1e-12
    assert bm.tempo == 130.0


def test_filter_hard_locks_on_first_sample():
    bm = BeatMap(120.0)
    f = TimelineFilter()
    f.add_sample(tempo=100.0, beat=20.0, master_send_t=50.0, recv_local_t=50.01)
    f.update(bm, now=50.01, dt=0.01)
    assert f.locked
    assert abs(bm.beat_at(50.01) - 20.0) < 1e-9
    assert abs(bm.tempo - 100.0) < 1e-9


def test_filter_ignores_reordered_samples():
    f = TimelineFilter()
    f.add_sample(120.0, 10.0, 5.00, 5.01)
    f.add_sample(120.0, 10.2, 5.10, 5.11)
    # a stale copy of the first sample arriving late must be dropped
    f.add_sample(120.0, 10.0, 5.00, 5.12)
    assert len(f.samples) == 2


def test_filter_min_delay_selects_least_delayed():
    bm = BeatMap(120.0)
    f = TimelineFilter()
    # true offset: master clock = local clock + 100 s; min delay 2 ms
    for i in range(20):
        send = 10.0 + i * 0.1
        delay = 0.002 + (0.03 if i % 3 else 0.0)  # jittery delay
        beat = 8.0 + i * 0.2  # 120 bpm -> 0.2 beat per 0.1 s
        f.add_sample(120.0, beat, send, send - 100.0 + delay)
    f.update(bm, now=10.0 - 100.0 + 0.002 + 2.0, dt=0.01)
    f.update(bm, now=10.0 - 100.0 + 0.002 + 2.0, dt=0.01)
    # after lock, local beat should track master beat within a few ms
    master_beat_now = 8.0 + (12.0 - 10.0) * 2.0
    err_beats = abs(bm.beat_at(10.0 - 100.0 + 0.002 + 2.0) - master_beat_now)
    assert err_beats * 0.5 < 0.010  # < 10 ms at 120 bpm


def test_filter_slew_is_bounded():
    bm = BeatMap(120.0)
    f = TimelineFilter(max_adj=0.05)
    f.add_sample(120.0, 0.0, 0.0, 0.001)
    f.update(bm, now=0.001, dt=0.01)
    # master suddenly far ahead (e.g. burst of delayed packets arriving):
    for i in range(50):
        send = 1.0 + i * 0.1
        f.add_sample(120.0, 2.0 + i * 0.2, send, send + 0.001)
        now = send + 0.001
        before = bm.tempo
        f.update(bm, now=now, dt=0.01)
        # tempo adjustment never exceeds max_adj = 5%
        assert abs(bm.tempo - 120.0) <= 120.0 * 0.05 + 1e-9
