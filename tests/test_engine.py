import numpy as np

from interbeat.app import BeatApp
from interbeat.measure import find_onsets
from interbeat.sequencer import Pattern, Sequencer, Step
from interbeat.transport import VirtualNetwork


def make_app(pattern=None, sr=8000, quantum=4.0):
    net = VirtualNetwork()
    app = BeatApp("solo", "solo", net.connect("solo"), sr=sr, quantum=quantum)
    if pattern is not None:
        app.sequencer = Sequencer(pattern)
        app.engine.sequencer = app.sequencer
    return app


def test_sample_accurate_placement():
    sr = 8000
    app = make_app(Pattern(4, [Step(2.0, "kick")]), sr=sr)
    app.session.request_play(True)
    # render until the first hit has passed
    bufs = [app.tick(80) for _ in range(400)]  # 4 s
    x = np.concatenate(bufs)
    # engine placement is sample-accurate: the kick's first sample lands
    # exactly at the scheduled frame (detector envelopes have their own
    # constant bias, so verify placement on the raw signal here)
    first = int(np.flatnonzero(x != 0.0)[0])
    # play starts at quantized beat 4.0; the pattern's step at phrase
    # position 2.0 first fires at shared beat 6.0
    tl = app.session.beatmap
    t_hit = tl.time_at(6.0)
    assert abs(first / sr - t_hit) < 2.0 / sr  # within 2 samples


def test_no_double_trigger_across_buffers():
    sr = 8000
    app = make_app(Pattern(4, [Step(0.0, "kick"), Step(1.0, "kick"),
                               Step(2.0, "kick"), Step(3.0, "kick")]), sr=sr)
    app.session.request_play(True)
    for _ in range(400):
        app.tick(80)
    beats = [b for b, _ in app.engine.hit_log]
    assert len(beats) == len(set(beats))  # each beat scheduled exactly once


def test_voice_cap_and_overflow_counter():
    sr = 8000
    dense = Pattern(1, [Step(i / 64.0, "kick") for i in range(64)])
    app = make_app(dense, sr=sr, quantum=1.0)
    app.session.request_play(True)
    for _ in range(300):  # 3 s: playback starts at beat 1 (t=0.5 s)
        app.tick(80)
    assert len(app.engine.voices) <= 32
    assert app.engine.overflows > 0


def test_quantized_start_boundary():
    sr = 8000
    app = make_app(Pattern(4, [Step(0.0, "kick")]), sr=sr, quantum=4.0)
    for _ in range(137):  # 1.37 s -> beat ~2.74 at 120 bpm
        app.tick(80)
    qb = app.session.request_play(True)
    assert qb is not None and qb % 4.0 == 0.0
    for _ in range(400):
        app.tick(80)
    first_beat = app.engine.hit_log[0][0]
    assert first_beat == qb  # first hit lands exactly on the quantized beat
