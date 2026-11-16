"""Transports: real UDP mesh, fault-injecting UDP, and a deterministic
virtual network for tests.

All transports share one interface:

    send(msg)        -> queue a datagram (dict) to all known peers
    poll()           -> list of (msg, recv_local_time) received since last
                        call (bounded); recv_local_time is stamped on ARRIVAL
                        in the receiver's own clock, not at processing time,
                        so control-loop polling intervals do not quantize the
                        sync measurements
"""
from __future__ import annotations

import heapq
import random
import socket
import threading
import time
from collections import deque

from . import protocol
from .timeline import SystemClock

MAX_DGRAM = 4096
POLL_BUDGET = 64  # bounded work per tick


class Transport:
    def send(self, msg: dict) -> None:
        raise NotImplementedError

    def poll(self) -> list[tuple[dict, float]]:
        raise NotImplementedError

    def set_clock(self, clock) -> None:
        pass

    def close(self) -> None:
        pass


class UDPMeshTransport(Transport):
    """Real UDP sockets. Peers are explicit (host, port) pairs; an optional
    broadcast address covers LAN discovery between machines. A receiver
    thread stamps datagrams on arrival."""

    def __init__(self, bind_host: str, bind_port: int,
                 peers: list[tuple[str, int]] | None = None,
                 broadcast: tuple[str, int] | None = None, clock=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind_host, bind_port))
        self.sock.setblocking(False)
        self.peers = list(peers or [])
        self.broadcast = broadcast
        if broadcast is not None:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._clock = clock or SystemClock()
        self._rx: deque = deque(maxlen=1024)
        self._closed = False
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def set_clock(self, clock) -> None:
        self._clock = clock

    def _recv_loop(self) -> None:
        self.sock.setblocking(True)
        while not self._closed:
            try:
                data, _addr = self.sock.recvfrom(MAX_DGRAM)
            except OSError:
                return
            msg = protocol.decode(data)
            if msg is not None:
                self._rx.append((msg, self._clock.now()))

    def add_peer(self, addr: tuple[str, int]) -> None:
        if addr not in self.peers:
            self.peers.append(addr)

    def _emit(self, data: bytes) -> None:
        for addr in self.peers:
            try:
                self.sock.sendto(data, addr)
            except OSError:
                pass
        if self.broadcast is not None:
            try:
                self.sock.sendto(data, self.broadcast)
            except OSError:
                pass

    def send(self, msg: dict) -> None:
        self._emit(protocol.encode(msg))

    def poll(self) -> list[tuple[dict, float]]:
        out = []
        for _ in range(POLL_BUDGET):
            try:
                out.append(self._rx.popleft())
            except IndexError:
                break
        return out

    def close(self) -> None:
        self._closed = True
        try:
            self.sock.close()
        except OSError:
            pass


class FaultyUDPMeshTransport(UDPMeshTransport):
    """Real UDP transport with controllable loss / delay / reordering.

    Used by the panel's fault-injection controls and the real-socket demo:
    packets genuinely traverse the kernel UDP stack, but a configurable
    fraction is dropped or held back (hold-and-release creates reordering).
    """

    def __init__(self, *args, loss: float = 0.0, jitter: float = 0.0,
                 reorder: float = 0.0, seed: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = loss
        self.jitter = jitter
        self.reorder = reorder
        self.rng = random.Random(seed)
        self._held: list[tuple[float, bytes]] = []  # heap of (due_mono, data)
        self.stats = {"sent": 0, "dropped": 0, "delayed": 0}

    def configure(self, loss: float | None = None, jitter: float | None = None,
                  reorder: float | None = None) -> None:
        if loss is not None:
            self.loss = max(0.0, min(1.0, loss))
        if jitter is not None:
            self.jitter = max(0.0, jitter)
        if reorder is not None:
            self.reorder = max(0.0, min(1.0, reorder))

    def send(self, msg: dict) -> None:
        data = protocol.encode(msg)
        if self.rng.random() < self.loss:
            self.stats["dropped"] += 1
            return
        delay = self.rng.uniform(0.0, self.jitter)
        if self.rng.random() < self.reorder:
            delay += 0.05 + self.rng.uniform(0.0, 0.05)
        if delay <= 0.0:
            self.stats["sent"] += 1
            self._emit(data)
        else:
            self.stats["delayed"] += 1
            heapq.heappush(self._held, (time.monotonic() + delay, data))

    def poll(self) -> list[tuple[dict, float]]:
        now = time.monotonic()
        while self._held and self._held[0][0] <= now:
            _, data = heapq.heappop(self._held)
            self.stats["sent"] += 1
            self._emit(data)
        return super().poll()


class VirtualNetwork:
    """Deterministic in-process network for tests.

    True time is advanced explicitly by the test; per-link delay, jitter,
    loss and reorder are configurable, with a seeded RNG for reproducibility.
    Delivery records the true delivery instant so the receiver can stamp it
    in its own (possibly drifting) clock exactly.
    """

    def __init__(self, seed: int = 7):
        self.t = 0.0
        self.rng = random.Random(seed)
        self.links: dict[tuple[str, str], dict] = {}
        self.default_link = {"delay": 0.001, "jitter": 0.0, "loss": 0.0,
                             "reorder": 0.0}
        self._queue: list[tuple[float, int, str, dict]] = []
        self._seq = 0
        self.endpoints: dict[str, "VirtualTransport"] = {}

    def connect(self, node_id: str) -> "VirtualTransport":
        ep = VirtualTransport(self, node_id)
        self.endpoints[node_id] = ep
        return ep

    def set_all_links(self, **kw) -> None:
        self.default_link.update(kw)

    def transmit(self, src: str, msg: dict) -> None:
        for dst, ep in self.endpoints.items():
            if dst == src:
                continue
            cfg = self.links.get((src, dst), self.default_link)
            if self.rng.random() < cfg["loss"]:
                continue
            delay = cfg["delay"] + self.rng.uniform(0.0, cfg["jitter"])
            if self.rng.random() < cfg["reorder"]:
                delay += 0.05 + self.rng.uniform(0.0, 0.05)
            self._seq += 1
            heapq.heappush(self._queue, (self.t + delay, self._seq, dst, msg))

    def advance(self, dt: float) -> None:
        self.t += dt
        while self._queue and self._queue[0][0] <= self.t:
            deliver_t, _, dst, msg = heapq.heappop(self._queue)
            ep = self.endpoints.get(dst)
            if ep is not None:
                ep._inbox.append((msg, deliver_t))


class VirtualTransport(Transport):
    def __init__(self, net: VirtualNetwork, node_id: str):
        self.net = net
        self.node_id = node_id
        self.clock = None  # set by the harness to the node's audio clock
        self._inbox: list[tuple[dict, float]] = []

    def send(self, msg: dict) -> None:
        self.net.transmit(self.node_id, msg)

    def poll(self) -> list[tuple[dict, float]]:
        batch, self._inbox = self._inbox[:POLL_BUDGET], self._inbox[POLL_BUDGET:]
        if self.clock is None:
            return [(m, self.net.t) for m, _ in batch]
        now = self.clock.now()
        rate = 1.0 + getattr(self.clock, "drift", 0.0)
        return [(m, now - (self.net.t - d) * rate) for m, d in batch]
