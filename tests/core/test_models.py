from pathlib import Path

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
