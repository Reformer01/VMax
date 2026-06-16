from __future__ import annotations

from speechsum.audio.recorder import list_devices


def test_list_devices():
    devices = list_devices()
    assert isinstance(devices, list)
    if devices:
        dev = devices[0]
        assert "index" in dev
        assert "name" in dev
        assert "channels" in dev
        assert dev["channels"] > 0
