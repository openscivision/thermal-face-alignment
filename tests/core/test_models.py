from pathlib import Path

import numpy as np
import pytest

from tfan.core import models


def test_model_downloader_uses_gdown(monkeypatch, tmp_path):
    calls = {}

    def fake_download(url, output, quiet=False):
        calls["url"] = url
        calls["output"] = output
        Path(output).write_bytes(b"weights")
        return output

    monkeypatch.setattr(models.gdown, "download", fake_download)

    downloader = models._ModelDownloader("478", save_dir=tmp_path)
    model_path = downloader.download_model()

    assert model_path.exists()
    assert models._file_id_map["478"] in calls["url"]
    assert Path(calls["output"]) == model_path


def test_model_downloader_skips_existing_file(monkeypatch, tmp_path):
    existing = tmp_path / "70.pt"
    existing.write_bytes(b"preexisting")

    def fail_download(*args, **kwargs):  # pragma: no cover - would signal a bug
        raise AssertionError("gdown should not be called when file exists")

    monkeypatch.setattr(models.gdown, "download", fail_download)

    downloader = models._ModelDownloader("70", save_dir=tmp_path)
    model_path = downloader.download_model()

    assert model_path == existing
    assert model_path.read_bytes() == b"preexisting"


def test_thermal_landmarks_init_loads_model(monkeypatch, tmp_path):
    class DummyModel:
        def __init__(self, n_landmarks, use_depth=False):
            self.n_landmarks = n_landmarks
            self.loaded = None
            self.device = None

        def load_state_dict(self, state, strict=False):
            self.loaded = (state, strict)

        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

    class DummyLandmarker:
        def __init__(self):
            self.detect_called = False

        def detect(self, _img):
            self.detect_called = True
            return []

    model_path = tmp_path / "dummy.pt"
    model_path.write_bytes(b"stub")

    calls = {}

    monkeypatch.setattr(models, "DMMv2", DummyModel)
    monkeypatch.setattr(
        models.ThermalLandmarks, "_landmarker_cls", DummyLandmarker, raising=False
    )

    def fake_get_model(model_name):
        calls["model"] = model_name
        return model_path

    monkeypatch.setattr(models, "_get_model", fake_get_model)

    def fake_torch_load(path, weights_only=False):
        calls["torch_load"] = {"path": Path(path), "weights_only": weights_only}
        return {"state": "ok"}

    monkeypatch.setattr(models.torch, "load", fake_torch_load)

    tl = models.ThermalLandmarks(device="cpu", n_landmarks=478, normalize=False)

    assert calls["model"] == "478"
    assert calls["torch_load"]["path"] == model_path
    assert calls["torch_load"]["weights_only"] is True
    assert isinstance(tl.dmm, DummyModel)
    assert tl.dmm.loaded == ({"state": "ok"}, False)
    assert tl.device == "cpu"
    assert tl.n_landmarks == 478


def _make_process_only_landmarker(monkeypatch):
    tl = models.ThermalLandmarks.__new__(models.ThermalLandmarks)
    tl.img_shape = None
    tl.eta = 0.75
    tl.max_lvl = 0
    tl.stride = 100
    tl.last_sparse_lm = None

    class DummyTracker:
        def __init__(self):
            self.last_input = None

        def detect(self, img2d):
            self.last_input = img2d
            return [{"landmarks": [], "box": []}]

    tracker = DummyTracker()
    tl.face_tracker = tracker

    captured = {}

    def capture(img3d):
        captured["img3d"] = img3d
        return "lm", "conf"

    monkeypatch.setattr(tl, "get_landmarks_multi", capture)
    monkeypatch.setattr(tl, "get_landmarks_single", capture)
    return tl, tracker, captured


def test_process_rejects_invalid_mode(monkeypatch):
    tl, _tracker, _captured = _make_process_only_landmarker(monkeypatch)
    with pytest.raises(ValueError, match="mode must be one of"):
        tl.process(np.zeros((2, 2), dtype=np.float32), mode="rgb")


def test_process_auto_integer_chooses_pixel(monkeypatch):
    tl, tracker, captured = _make_process_only_landmarker(monkeypatch)
    img = np.array([[0, 10], [255, 300]], dtype=np.uint16)

    tl.process(img, mode="auto", multi=True)

    assert tracker.last_input is not None
    assert tracker.last_input.dtype == np.float32
    assert tracker.last_input.shape == img.shape
    assert np.array_equal(tracker.last_input, img.astype(np.float32))

    out = captured["img3d"]
    expected2d = np.clip(img.astype(np.float32), 0.0, 255.0)
    expected3d = np.repeat(expected2d[..., None], 3, axis=2).astype(np.float32)
    assert out.shape == expected3d.shape
    assert out.dtype == np.float32
    assert np.array_equal(out, expected3d)


def test_process_auto_float_0_1_chooses_pixel(monkeypatch):
    tl, tracker, captured = _make_process_only_landmarker(monkeypatch)
    img = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)

    tl.process(img, mode="auto", multi=True)

    assert tracker.last_input is not None
    assert np.array_equal(tracker.last_input, img.astype(np.float32))

    out = captured["img3d"]
    expected2d = np.clip(img, 0.0, 1.0) * 255.0
    expected3d = np.repeat(expected2d[..., None], 3, axis=2).astype(np.float32)
    assert np.allclose(out, expected3d, atol=1e-5)


def test_process_auto_temperature_range_chooses_temperature(monkeypatch):
    tl, tracker, captured = _make_process_only_landmarker(monkeypatch)
    img = np.array([[20.0, 30.0], [40.0, 25.0]], dtype=np.float32)

    tl.process(img, mode="auto", multi=True)

    assert tracker.last_input is not None
    assert np.array_equal(tracker.last_input, img.astype(np.float32))

    out = captured["img3d"]
    expected2d = (img - 20.0) / (40.0 - 20.0) * 255.0
    expected3d = np.repeat(expected2d[..., None], 3, axis=2).astype(np.float32)
    assert np.allclose(out, expected3d, atol=1e-5)


def test_process_auto_float_high_range_chooses_pixel(monkeypatch):
    tl, tracker, captured = _make_process_only_landmarker(monkeypatch)
    img = np.array([[0.0, 100.0], [200.0, 255.0]], dtype=np.float32)

    tl.process(img, mode="auto", multi=True)

    assert tracker.last_input is not None
    assert np.array_equal(tracker.last_input, img.astype(np.float32))

    out = captured["img3d"]
    expected2d = np.clip(img, 0.0, 255.0)
    expected3d = np.repeat(expected2d[..., None], 3, axis=2).astype(np.float32)
    assert np.array_equal(out, expected3d)


def test_process_rgb_warns_and_is_averaged(monkeypatch):
    tl, tracker, captured = _make_process_only_landmarker(monkeypatch)
    temp = np.array([[20.0, 30.0], [40.0, 25.0]], dtype=np.float32)
    rgb = np.stack([temp, temp, temp], axis=2)

    with pytest.warns(RuntimeWarning, match="got a 3-channel image"):
        tl.process(rgb, mode="auto", multi=True)

    assert tracker.last_input is not None
    assert tracker.last_input.ndim == 2
    assert np.allclose(tracker.last_input, temp, atol=1e-6)

    out = captured["img3d"]
    expected2d = (temp - 20.0) / (40.0 - 20.0) * 255.0
    expected3d = np.repeat(expected2d[..., None], 3, axis=2).astype(np.float32)
    assert np.allclose(out, expected3d, atol=1e-5)


def test_process_rejects_non_2d_frame(monkeypatch):
    tl, _tracker, _captured = _make_process_only_landmarker(monkeypatch)
    bad = np.zeros((2, 2, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected 2D thermal frame"):
        tl.process(bad, mode="auto", multi=True)


def test_process_temperature_mode_constant_frame(monkeypatch):
    tl, tracker, captured = _make_process_only_landmarker(monkeypatch)
    img = np.full((2, 2), 30.0, dtype=np.float32)

    tl.process(img, mode="temperature", multi=True)

    assert tracker.last_input is not None
    assert np.array_equal(tracker.last_input, img.astype(np.float32))

    out = captured["img3d"]
    assert out.shape == (2, 2, 3)
    assert out.dtype == np.float32
    assert np.array_equal(out, np.zeros((2, 2, 3), dtype=np.float32))


def test_process_pixel_mode_does_not_temperature_normalize(monkeypatch):
    tl, tracker, captured = _make_process_only_landmarker(monkeypatch)
    img = np.array([[20.0, 30.0], [40.0, 25.0]], dtype=np.float32)

    tl.process(img, mode="pixel", multi=True)

    assert tracker.last_input is not None
    assert np.array_equal(tracker.last_input, img.astype(np.float32))

    out = captured["img3d"]
    expected3d = np.repeat(img[..., None], 3, axis=2).astype(np.float32)
    assert np.array_equal(out, expected3d)
