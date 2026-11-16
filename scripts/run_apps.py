#!/usr/bin/env python3
"""Headless demo: three sequencer apps sync over REAL UDP sockets
(loopback mesh with fault injection), render 30 s of audio to out/*.wav,
and print the inter-app hit deviation measured on the synthesized output.

    python3 scripts/run_apps.py [--seconds 30] [--loss 0.05] [--jitter 0.03]
"""
import argparse
import os
import sys
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, ".pylocal"), ROOT):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from interbeat.app import BeatApp
from interbeat.measure import find_onsets
from interbeat.transport import FaultyUDPMeshTransport

SR = 44100
IDS = ["A", "B", "C"]
BASE_PORT = 22101


def write_wav(path, x, sr):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--loss", type=float, default=0.05)
    ap.add_argument("--jitter", type=float, default=0.03)
    ap.add_argument("--reorder", type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    apps, transports = {}, {}
    for i, nid in enumerate(IDS):
        peers = [("127.0.0.1", BASE_PORT + j) for j in range(len(IDS))
                 if j != i]
        tr = FaultyUDPMeshTransport("127.0.0.1", BASE_PORT + i, peers=peers,
                                    loss=args.loss, jitter=args.jitter,
                                    reorder=args.reorder, seed=i + 1)
        app = BeatApp(nid, nid, tr, node_index=i, sr=SR, t0=1000.0 * i)
        tr.set_clock(app.engine)
        apps[nid], transports[nid] = app, tr

    n = int(SR * 0.01)
    audio = {nid: [] for nid in IDS}
    print(f"3 个应用经 UDP 端口 {BASE_PORT}-{BASE_PORT+2} 互联，"
          f"丢包={args.loss:.0%} 抖动=±{args.jitter*1000:.0f}ms "
          f"重排={args.reorder:.0%}")
    t_end = time.monotonic() + args.seconds
    next_t = time.monotonic()
    started = False
    while time.monotonic() < t_end:
        next_t += 0.01
        if not started and time.monotonic() > t_end - args.seconds + 1.0:
            qb = apps["A"].session.request_play(True)
            print(f"[1.0s] 播放 @ 量化边界 {qb:.1f} 拍")
            started = True
        for nid in IDS:
            audio[nid].append(apps[nid].tick(n))
        d = next_t - time.monotonic()
        if d > 0:
            time.sleep(d)
        else:
            next_t = time.monotonic()
    for tr in transports.values():
        tr.close()

    # ---- measure inter-app hit deviation on the synthesized output ----
    slots = {}
    for nid in IDS:
        x = np.concatenate(audio[nid])
        write_wav(os.path.join(ROOT, "out", f"app_{nid}.wav"), x, SR)
        on = find_onsets(x, SR)
        eng = apps[nid].engine
        t = eng.t0 + on / SR * (1.0 + eng.drift)
        beats = apps[nid].session.beatmap.beat_at(t)
        s = {}
        for slot, b in zip(np.round(beats * 4.0) / 4.0, beats):
            s.setdefault(float(slot), float(b))
        slots[nid] = s
    mix = np.sum([np.concatenate(audio[nid]) for nid in IDS], axis=0) / 3
    write_wav(os.path.join(ROOT, "out", "mix.wav"), np.tanh(mix), SR)

    tempo = apps["A"].session.session_tempo
    print("\n跨端击打偏差（由合成输出实测，共享拍槽配对）:")
    worst = 0.0
    for i in range(len(IDS)):
        for j in range(i + 1, len(IDS)):
            a, b = slots[IDS[i]], slots[IDS[j]]
            devs = np.array([(b[s] - a[s]) * 60000.0 / tempo
                             for s in set(a) & set(b)])
            ms = np.abs(devs)
            worst = max(worst, float(np.percentile(ms, 99)))
            print(f"  {IDS[i]}-{IDS[j]}: n={len(ms)} 均值={ms.mean():.2f}ms "
                  f"p99={np.percentile(ms, 99):.2f}ms 最大={ms.max():.2f}ms")
    print(f"\nWAV 已写入 out/（app_A/B/C.wav 与 mix.wav），"
          f"最劣 p99 = {worst:.2f}ms")


if __name__ == "__main__":
    main()
