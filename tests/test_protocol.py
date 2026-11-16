from interbeat import protocol


def test_roundtrip_all_types():
    msgs = [
        protocol.hello("a", "alpha", 3, "a",
                       {"seq": 2, "playing": True, "beat": 8.0}),
        protocol.timeline("a", 3, 1, 0, 120.0, 16.5, 100.25),
        protocol.play("a", 2, True, 8.0),
        protocol.master("a", 3, "a"),
    ]
    for m in msgs:
        out = protocol.decode(protocol.encode(m))
        assert out == m


def test_decode_rejects_garbage():
    assert protocol.decode(b"") is None
    assert protocol.decode(b"not json") is None
    assert protocol.decode(b"{}") is None
    assert protocol.decode(b'{"type":"nope","node":"a"}') is None
    assert protocol.decode(b'{"type":"hello"}') is None  # missing node
