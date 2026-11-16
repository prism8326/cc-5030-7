"""Multi-end production panel for the interactive beat session.

Hosts three BeatApps talking over REAL UDP sockets (loopback mesh with
fault injection), renders their audio in a real-time driver loop, and
serves a web UI:

  GET /         the panel page
  GET /events   SSE stream of kernel state (10 Hz)
  GET /monitor  the kernel's actual rendered mix, s16le 44.1 kHz
  POST /control control actions routed into the kernel

Everything on the page is driven by the kernel: lane playheads and hit
flashes render kernel state, controls call the kernel API, the deviation
table is measured from the rendered audio, and kernel failures (rejections,
packet loss, resyncs, overflows) are reported back into the page log.
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .app import BeatApp
from .measure import find_onsets
from .page import PAGE
from .sequencer import Step
from .transport import FaultyUDPMeshTransport

APP_IDS = ["A", "B", "C"]
BASE_PORT = 21101
TICK = 0.01
DEFAULT_INST = {"A": "kick", "B": "snare", "C": "blip"}


class InterpolatedClock:
    """Engine-time clock that interpolates between driver ticks, so network
    arrival stamps are not quantized to the tick grid."""

    def __init__(self, engine):
        self.engine = engine
        self._wall0 = time.monotonic()
        self._eng0 = engine.now()

    def sync(self):
        self._wall0 = time.monotonic()
        self._eng0 = self.engine.now()

    def now(self):
        return self._eng0 + (time.monotonic() - self._wall0)


class AppHolder:
    def __init__(self, panel, node_id):
        self.panel = panel
        self.id = node_id
        self.alive = True
        idx = APP_IDS.index(node_id)
        peers = [("127.0.0.1", BASE_PORT + j) for j in range(len(APP_IDS))
                 if j != idx]
        self.transport = FaultyUDPMeshTransport(
            "127.0.0.1", BASE_PORT + idx, peers=peers, seed=idx + 1)
        self.app = BeatApp(node_id, node_id, self.transport, node_index=idx,
                           sr=panel.sr, on_event=panel.on_kernel_event)
        self.iclock = InterpolatedClock(self.app.engine)
        self.transport.set_clock(self.iclock)
        self.app.engine.on_hit = self.on_hit

    def on_hit(self, beat, inst):
        self.panel.recent_hits[self.id].append((beat, inst))

    def close(self):
        self.alive = False
        self.transport.close()


class Panel:
    def __init__(self, sr=44100):
        self.sr = sr
        self.lock = threading.Lock()
        self.apps: dict[str, AppHolder] = {}
        for nid in APP_IDS:
            self.apps[nid] = AppHolder(self, nid)
        self.log: deque = deque(maxlen=80)
        self.recent_hits: dict[str, list] = {nid: [] for nid in APP_IDS}
        self.recent_audio: dict[str, deque] = {nid: deque(maxlen=200)
                                               for nid in APP_IDS}
        self.dev_samples: dict[str, deque] = {}
        self.dev_table: dict[str, dict] = {}
        self.audio_subs: list[queue.Queue] = []
        self.state_subs: list[queue.Queue] = []
        self.running = False
        self.add_log("面板启动：3 个应用经 UDP 网格互联")

    # ------------------------------------------------------------ logging
    def add_log(self, text, bad=False):
        t = time.strftime("%H:%M:%S") + f".{int(time.monotonic()*10)%10}"
        self.log.append({"t": t, "text": text, "bad": bad})

    def on_kernel_event(self, kind, **kw):
        bad = kind in ("rejected", "peer_lost", "master_timeout",
                       "resync_saturated")
        detail = kw.get("detail", "")
        self.add_log(f"{kind} {detail}".strip(), bad=bad)

    # --------------------------------------------------------- driver loop
    def start(self):
        self.running = True
        threading.Thread(target=self._driver, daemon=True).start()

    def _driver(self):
        n = int(self.sr * TICK)
        next_t = time.monotonic()
        last_measure = last_state = 0.0
        while self.running:
            next_t += TICK
            bufs = {}
            with self.lock:
                holders = [h for h in self.apps.values() if h.alive]
            for h in holders:
                h.iclock.sync()
                bufs[h.id] = h.app.tick(n)
            if bufs:
                mix = np.sum(list(bufs.values()), axis=0) / len(bufs)
                mix = np.tanh(mix)
                self._push_audio((mix * 32767).astype(np.int16).tobytes())
                for nid, b in bufs.items():
                    self.recent_audio[nid].append(b)
            now = time.monotonic()
            if now - last_measure > 0.5:
                last_measure = now
                self._measure()
            if now - last_state > 0.1:
                last_state = now
                self._push_state()
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    # -------------------------------------------------- live measurement
    def _measure(self):
        """Inter-app hit deviation measured from the rendered audio: onsets
        are mapped to shared-beat slots via each app's own beat map and
        paired per slot (same method as the test harness)."""
        with self.lock:
            holders = [h for h in self.apps.values() if h.alive]
        slots = {}
        for h in holders:
            if not h.app.session.playing:
                continue
            audio = np.concatenate(self.recent_audio[h.id])
            if len(audio) < self.sr:
                continue
            on = find_onsets(audio, self.sr)
            if len(on) < 3:
                continue
            f0 = h.app.engine.frame - len(audio)
            t = (h.app.engine.t0 + (f0 + on) / self.sr
                 * (1.0 + h.app.engine.drift))
            beats = h.app.session.beatmap.beat_at(t)
            s = {}
            for slot, b in zip(np.round(beats * 4.0) / 4.0, beats):
                s.setdefault(float(slot), float(b))
            slots[h.id] = s
        ids = sorted(slots)
        tempo = self._master_tempo()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair = f"{ids[i]}-{ids[j]}"
                dq = self.dev_samples.setdefault(pair, deque(maxlen=300))
                a, b = slots[ids[i]], slots[ids[j]]
                for s in set(a) & set(b):
                    dq.append((b[s] - a[s]) * 60000.0 / tempo)
        self.dev_table = {}
        for pair, dq in self.dev_samples.items():
            if not dq:
                continue
            ms = np.abs(np.array(dq))
            self.dev_table[pair] = {
                "n": int(len(ms)), "mean_ms": float(ms.mean()),
                "p99_ms": float(np.percentile(ms, 99)),
                "max_ms": float(ms.max())}

    def _master_tempo(self):
        with self.lock:
            for h in self.apps.values():
                if h.alive and h.app.session.is_master:
                    return h.app.session.session_tempo
        return 120.0

    # ------------------------------------------------------------ state
    def _state(self):
        with self.lock:
            holders = list(self.apps.values())
        master = next((h for h in holders if h.alive and
                       h.app.session.is_master), None)
        ref = master or next((h for h in holders if h.alive), None)
        beat = 0.0
        quantum = 4.0
        playing = False
        if ref is not None:
            ses = ref.app.session
            beat = ses.beatmap.beat_at(ref.app.engine.now())
            quantum = ses.quantum
            playing = ses.playing
        apps = []
        for h in holders:
            ses = h.app.session
            eng = h.app.engine
            b = ses.beatmap.beat_at(eng.now())
            ph = h.app.sequencer.pattern.phrase_beats
            hits, self.recent_hits[h.id] = self.recent_hits[h.id], []
            apps.append({
                "id": h.id, "alive": h.alive,
                "beat": b, "bar": int(b // ph) + 1,
                "phrase_pos": b % ph, "phrase_beats": ph,
                "phrase_base": float(int(b // ph) * ph),
                "is_master": ses.is_master, "locked": ses.map_ready,
                "steps": [{"beat": s.beat, "inst": s.inst, "vel": s.vel,
                           "note": s.note}
                          for s in h.app.sequencer.pattern.steps],
                "hits": [{"beat": bb, "inst": ii} for bb, ii in hits],
            })
        nb = math.ceil((beat + 0.25 - 1e-9) / quantum) * quantum
        net = {}
        for h in holders:
            st = h.transport.stats
            net[h.id] = f"发{st['sent']} 丢{st['dropped']} 缓{st['delayed']}"
        return {
            "beat": beat, "tempo": self._master_tempo(), "playing": playing,
            "quantum": quantum, "next_boundary": f"{nb:.1f}",
            "master": master.id if master else None,
            "term": ref.app.session.term if ref else 0,
            "apps": apps, "dev": self.dev_table,
            "log": list(self.log)[-30:],
            "netstats": "  ".join(f"{k}:{v}" for k, v in net.items()),
        }

    def _push_state(self):
        data = json.dumps(self._state()).encode()
        for q in list(self.state_subs):
            try:
                q.put_nowait(data)
            except queue.Full:
                pass

    def _push_audio(self, chunk: bytes):
        for q in list(self.audio_subs):
            try:
                q.put_nowait(chunk)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(chunk)
                except (queue.Empty, queue.Full):
                    pass

    # ----------------------------------------------------------- control
    def control(self, msg: dict) -> dict:
        action = msg.get("action")
        with self.lock:
            holders = {nid: h for nid, h in self.apps.items() if h.alive}
        if not holders and action != "revive":
            return {"ok": False, "error": "没有存活的应用"}
        try:
            if action == "play" or action == "stop":
                h = self._master_or_any(holders)
                qb = h.app.session.request_play(action == "play")
                if qb is None:
                    return {"ok": False, "error": "状态未变化"}
                self.add_log(f"{'播放' if action=='play' else '停止'} @ 量化边界 {qb:.1f} 拍")
                return {"ok": True, "beat": qb}
            if action == "set_tempo":
                h = self._master_or_any(holders)
                if h.app.session.set_tempo(float(msg["bpm"])):
                    self.add_log(f"速度 → {float(msg['bpm']):.1f} BPM")
                    return {"ok": True}
                return {"ok": False, "error": "被拒绝：不是主时钟"}
            if action == "set_master":
                node = msg["node"]
                h = holders.get(node) or self._master_or_any(holders)
                h.app.session.request_master(node)
                self.add_log(f"主时钟 → {node}")
                return {"ok": True}
            if action == "toggle_step":
                h = holders[msg["app"]]
                self._toggle_step(h, int(msg["cell"]))
                return {"ok": True}
            if action == "set_phrase":
                h = holders[msg["app"]]
                self._set_phrase(h, float(msg["beats"]))
                return {"ok": True}
            if action == "inject":
                for h in holders.values():
                    h.transport.configure(loss=msg.get("loss"),
                                          jitter=msg.get("jitter"),
                                          reorder=msg.get("reorder"))
                self.add_log(f"故障注入 丢包={msg.get('loss',0):.0%} "
                             f"抖动={msg.get('jitter',0)*1000:.0f}ms "
                             f"重排={msg.get('reorder',0):.0%}")
                return {"ok": True}
            if action == "kill":
                h = self._master_or_any(holders)
                with self.lock:
                    h.close()
                self.add_log(f"应用 {h.id}（主时钟）已宕掉", bad=True)
                return {"ok": True}
            if action == "revive":
                with self.lock:
                    dead = [nid for nid, h in self.apps.items() if not h.alive]
                    for nid in dead:
                        self.apps[nid].close()
                        self.apps[nid] = AppHolder(self, nid)
                        self.recent_audio[nid].clear()
                        self.add_log(f"应用 {nid} 复活（迟加入）")
                if not dead:
                    return {"ok": False, "error": "没有离线的应用"}
                return {"ok": True}
            return {"ok": False, "error": f"未知动作 {action}"}
        except KeyError as e:
            return {"ok": False, "error": f"参数错误: {e}"}
        except Exception as e:  # surface kernel failures back to the page
            self.add_log(f"控制异常: {e}", bad=True)
            return {"ok": False, "error": str(e)}

    def _master_or_any(self, holders):
        for h in holders.values():
            if h.app.session.is_master:
                return h
        return next(iter(holders.values()))

    def _toggle_step(self, h, cell):
        beat = cell / 2.0
        steps = h.app.sequencer.pattern.steps
        for i, s in enumerate(steps):
            if abs(s.beat - beat) < 1e-9:
                del steps[i]
                self.add_log(f"{h.id}: 删除 {beat} 拍处的 {s.inst}")
                return
        steps.append(Step(beat, DEFAULT_INST.get(h.id, "kick"), 0.9))
        steps.sort(key=lambda s: s.beat)
        self.add_log(f"{h.id}: 在 {beat} 拍添加 {DEFAULT_INST.get(h.id)}")

    def _set_phrase(self, h, beats):
        pat = h.app.sequencer.pattern
        pat.phrase_beats = beats
        pat.steps = [s for s in pat.steps if s.beat < beats]
        self.add_log(f"{h.id}: 乐句长度 → {beats:g} 拍")


# --------------------------------------------------------------- HTTP API
class Handler(BaseHTTPRequestHandler):
    panel: Panel = None  # set by serve()
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = queue.Queue(maxsize=50)
            self.panel.state_subs.append(q)
            try:
                while True:
                    try:
                        data = q.get(timeout=15)
                        self.wfile.write(b"data: " + data + b"\n\n")
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                if q in self.panel.state_subs:
                    self.panel.state_subs.remove(q)
        elif self.path == "/monitor":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            q = queue.Queue(maxsize=100)
            self.panel.audio_subs.append(q)
            try:
                while True:
                    chunk = q.get()
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                if q in self.panel.audio_subs:
                    self.panel.audio_subs.remove(q)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/control":
            n = int(self.headers.get("Content-Length", 0))
            try:
                msg = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                msg = {}
            out = self.panel.control(msg)
            body = json.dumps(out).encode()
            self.send_response(200 if out.get("ok") else 400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


def serve(host="127.0.0.1", port=8410):
    panel = Panel()
    Handler.panel = panel
    panel.start()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"互动节拍面板: http://{host}:{port}/  （3 个应用经 UDP 端口 "
          f"{BASE_PORT}-{BASE_PORT+2} 互联）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        panel.running = False


if __name__ == "__main__":
    serve()
