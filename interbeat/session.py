"""Session: membership, mastership, and the shared (tempo, phase, play) state.

The three components are modelled and ordered independently:

  tempo   - ordered by (term, tempo_seq); adopted continuously on change
  phase   - ordered by (term, phase_seq); tracked via TimelineFilter
  play    - ordered by a Lamport play_seq; takes effect at a quantized
            shared beat, never immediately on packet receipt

Mastership is a switchable role ordered by a Lamport term. On handover the
new master adopts the session grid it was already tracking (its beat map is
already locked to it), so switching the master clock does not glitch the
shared beat.
"""
from __future__ import annotations

import math

from . import protocol
from .timeline import BeatMap, Clock, TimelineFilter
from .transport import Transport

HELLO_PERIOD = 0.5
BEACON_PERIOD = 0.1
MASTER_TIMEOUT = 2.0
PEER_TIMEOUT = 3.0
DEFAULT_QUANTUM = 4.0  # beats


class Session:
    def __init__(self, node_id: str, name: str, clock: Clock,
                 transport: Transport, quantum: float = DEFAULT_QUANTUM,
                 on_event=None):
        self.node_id = node_id
        self.name = name
        self.clock = clock
        self.transport = transport
        self.quantum = quantum
        self.on_event = on_event or (lambda kind, **kw: None)

        self.beatmap = BeatMap()
        self.filter = TimelineFilter()

        # shared state components (independently versioned)
        self.session_tempo = 120.0
        self.tempo_seq = 0
        self.phase_seq = 0
        self.term = 0
        self.master_id: str | None = None
        self.is_master = False
        self.play_seq = 0
        self.playing = False
        self.play_beat = 0.0  # shared beat at which current play state began

        self.peers: dict[str, dict] = {}  # node -> {name, last_seen}
        self._last_hello = -1e9
        self._last_beacon = -1e9
        self._last_tick: float | None = None
        self._sat_reported = False
        self.on_map_jump = None  # called when the map makes a hard jump

    @property
    def map_ready(self) -> bool:
        return self.is_master or self.filter.locked

    # -------------------------------------------------------------- control
    def set_tempo(self, bpm: float) -> bool:
        """Change the session tempo (master only). A follower request is
        rejected and reported - failure feedback for the UI."""
        if not self.is_master:
            self.on_event("rejected", detail="set_tempo: not master")
            return False
        bpm = max(30.0, min(300.0, float(bpm)))
        if bpm != self.session_tempo:
            self.session_tempo = bpm
            self.tempo_seq += 1
        return True

    def request_play(self, playing: bool) -> float | None:
        """Quantized start/stop at the next quantum boundary on the shared
        grid. A received packet never restarts anyone immediately."""
        if playing == self.playing:
            return None
        now = self.clock.now()
        b = self.beatmap.beat_at(now)
        qb = math.ceil((b + 1e-9) / self.quantum) * self.quantum
        if qb - b < 0.25:  # scheduling margin
            qb += self.quantum
        self.play_seq += 1
        self._apply_play(self.play_seq, playing, qb)
        self.transport.send(protocol.play(self.node_id, self.play_seq,
                                          playing, qb))
        return qb

    def request_master(self, node_id: str) -> None:
        self.term += 1
        self._apply_master(self.term, node_id)
        self.transport.send(protocol.master(self.node_id, self.term, node_id))

    def nudge_phase(self, beats: float) -> None:
        """Deliberate session-wide phase jump (master only)."""
        if not self.is_master:
            self.on_event("rejected", detail="nudge_phase: not master")
            return
        self.phase_seq += 1
        now = self.clock.now()
        self.beatmap.jump(now, self.beatmap.beat_at(now) + beats,
                          self.beatmap.tempo)
        self._beacon(now, hard=True)

    # ----------------------------------------------------------------- tick
    def tick(self) -> None:
        now = self.clock.now()
        dt = 0.01 if self._last_tick is None else min(0.1, now - self._last_tick)
        self._last_tick = now

        for msg, recv_stamp in self.transport.poll():
            self._handle(msg, now, recv_stamp)

        for node, info in list(self.peers.items()):
            if now - info["last_seen"] > PEER_TIMEOUT:
                del self.peers[node]
                self.on_event("peer_lost", node=node)
        if self.master_id is not None and self.master_id != self.node_id:
            m = self.peers.get(self.master_id)
            if m is None or now - m["last_seen"] > MASTER_TIMEOUT:
                self.on_event("master_timeout", node=self.master_id)
                self.master_id = None
        if self.master_id is None:
            self._elect()

        if now - self._last_hello >= HELLO_PERIOD:
            self._last_hello = now
            self.transport.send(protocol.hello(
                self.node_id, self.name, self.term, self.master_id,
                {"seq": self.play_seq, "playing": self.playing,
                 "beat": self.play_beat}))

        if self.is_master:
            self.beatmap.steer_to(now, self.session_tempo)
            if now - self._last_beacon >= BEACON_PERIOD:
                self._last_beacon = now
                self._beacon(now)
        else:
            jumped = self.filter.update(self.beatmap, now, dt)
            if jumped and self.on_map_jump:
                self.on_map_jump()
            if self.filter.saturated:
                if not self._sat_reported:
                    self._sat_reported = True
                    self.on_event("resync_saturated", err=self.filter.last_err)
            else:
                self._sat_reported = False

    def _beacon(self, now: float, hard: bool = False) -> None:
        self.transport.send(protocol.timeline(
            self.node_id, self.term, self.tempo_seq, self.phase_seq,
            self.session_tempo, self.beatmap.beat_at(now), now, hard=hard))

    # ------------------------------------------------------------- handlers
    def _handle(self, msg: dict, now: float, recv_stamp: float) -> None:
        t = msg["type"]
        node = msg["node"]
        if node == self.node_id:
            return
        if t == protocol.HELLO:
            is_new = node not in self.peers
            self.peers[node] = {"name": msg.get("name", node), "last_seen": now}
            if is_new:
                self.on_event("peer_joined", node=node)
            term = msg.get("term", 0)
            mid = msg.get("master_id")
            if mid and (term > self.term or
                        (term == self.term and self.master_id is not None
                         and mid < self.master_id)):
                self._apply_master(term, mid)
            p = msg.get("play") or {}
            if p.get("seq", 0) > self.play_seq:
                self._apply_play(p["seq"], bool(p["playing"]), float(p["beat"]))
        elif t == protocol.TIMELINE:
            self.peers.setdefault(node, {"name": node, "last_seen": now})
            self.peers[node]["last_seen"] = now
            term = msg.get("term", 0)
            if term > self.term:
                self._apply_master(term, node)
            elif term == self.term and node != self.master_id:
                if self.master_id is None or node < self.master_id:
                    self._apply_master(term, node)
            if node != self.master_id or term < self.term or self.is_master:
                return
            tempo_seq = msg.get("tempo_seq", 0)
            phase_seq = msg.get("phase_seq", 0)
            if tempo_seq > self.tempo_seq or phase_seq > self.phase_seq:
                # The timeline changed: samples of the old timeline
                # extrapolate wrongly, drop them from the min-delay window.
                self.filter.samples.clear()
            if tempo_seq > self.tempo_seq:
                self.session_tempo = float(msg["tempo"])
            self.tempo_seq = max(self.tempo_seq, tempo_seq)
            if msg.get("hard") and phase_seq > self.phase_seq:
                self.filter.reset()  # deliberate jump: re-lock hard
            self.phase_seq = max(self.phase_seq, phase_seq)
            if (tempo_seq, phase_seq) == (self.tempo_seq, self.phase_seq):
                # sample describes the current timeline; reordered samples of
                # superseded timelines are dropped here. recv_stamp is the
                # arrival time in our own clock, stamped by the transport.
                self.filter.add_sample(float(msg["tempo"]), float(msg["beat"]),
                                       float(msg["at"]), recv_stamp)
        elif t == protocol.PLAY:
            seq = msg.get("seq", 0)
            if seq > self.play_seq:
                self._apply_play(seq, bool(msg["playing"]), float(msg["beat"]))
        elif t == protocol.MASTER:
            term = msg.get("term", 0)
            mid = msg.get("master_id")
            if mid and (term > self.term or
                        (term == self.term and self.master_id is not None
                         and mid < self.master_id)):
                self._apply_master(term, mid)

    def _apply_play(self, seq: int, playing: bool, beat: float) -> None:
        self.play_seq = max(self.play_seq, seq)
        if playing != self.playing:
            self.on_event("play_state", playing=playing, beat=beat)
        self.playing = playing
        self.play_beat = beat

    def _apply_master(self, term: int, master_id: str) -> None:
        was_master = self.is_master
        self.term = max(self.term, term)
        self.master_id = master_id
        self.is_master = (master_id == self.node_id)
        self.tempo_seq = 0
        self.phase_seq = 0
        if self.is_master and not was_master:
            # Adopt the grid we were tracking: mastership handover is
            # continuous by construction (session_tempo and the beat map
            # already describe the shared grid).
            self.on_event("became_master", term=self.term)
        elif not self.is_master:
            if was_master:
                # We were the grid: adopt it as the lock state and steer
                # toward the new master - no re-lock jump, no audio gap.
                self.filter.adopt_current(self.beatmap)
                self.on_event("lost_master", term=self.term)
            else:
                # Re-key the filter onto the new master's sample stream;
                # keep the current map so the handover does not glitch.
                self.filter.reset(keep_lock=True)

    def _elect(self) -> None:
        winner = min([self.node_id] + list(self.peers))
        self._apply_master(self.term + 1, winner)
        self.transport.send(protocol.master(self.node_id, self.term, winner))

    # --------------------------------------------------------------- status
    def status(self) -> dict:
        now = self.clock.now()
        return {
            "node": self.node_id,
            "name": self.name,
            "beat": self.beatmap.beat_at(now),
            "tempo": self.session_tempo,
            "map_tempo": self.beatmap.tempo,
            "is_master": self.is_master,
            "master_id": self.master_id,
            "term": self.term,
            "playing": self.playing,
            "play_beat": self.play_beat,
            "peers": sorted(self.peers),
            "locked": True if self.is_master else self.filter.locked,
            "filter_err": 0.0 if self.is_master else self.filter.last_err,
        }
