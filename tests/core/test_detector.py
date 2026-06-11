import sys
from pathlib import Path

import numpy as np
import pytest
from tfan.core import detector, models
from tfan.core.detector import OnnxFaceDetector


class FakeSession:
    def __init__(self, raw):
        self.raw = raw
        self.feeds = []

    def get_inputs(self):
        class _Input:
            name = "images"

        return [_Input()]

    def run(self, _outputs, feeds):
        self.feeds.append(feeds)
        return [self.raw]


def _raw_output(detections, n_anchors=8400):
    raw = np.zeros((1, 5, n_anchors), dtype=np.float32)
    for column, (cx, cy, w, h, conf) in enumerate(detections):
        raw[0, :, column] = (cx, cy, w, h, conf)
    return raw


def test_robust_normalize_affine_invariance():
    rng = np.random.default_rng(0)
    pattern = rng.random((16, 16)).astype(np.float32)
    as_celsius = 20.0 + pattern * 20.0
    as_intensity = pattern * 255.0

    np.testing.assert_allclose(
        detector.robust_normalize(as_celsius),
        detector.robust_normalize(as_intensity),
        atol=1e-5,
    )


def test_robust_normalize_constant_and_nonfinite():
    constant = np.full((4, 4), 31.5, dtype=np.float32)
    assert np.array_equal(detector.robust_normalize(constant), np.zeros((4, 4), dtype=np.float32))

    all_nan = np.full((4, 4), np.nan, dtype=np.float32)
    assert np.array_equal(detector.robust_normalize(all_nan), np.zeros((4, 4), dtype=np.float32))

    partial = np.linspace(0.0, 100.0, 16, dtype=np.float32).reshape(4, 4)
    partial[0, 0] = np.nan
    out = detector.robust_normalize(partial)
    finite = np.isfinite(out)
    assert not finite[0, 0]
    assert finite[1:].all()
    assert out[finite].min() >= 0.0
    assert out[finite].max() <= 1.0


def test_letterbox_geometry():
    img = np.zeros((240, 320, 3), dtype=np.uint8)

    out, ratio, (left, top) = detector.letterbox(img, new_size=640)

    assert out.shape == (640, 640, 3)
    assert ratio == pytest.approx(2.0)
    assert (left, top) == (0, 80)
    assert (out[:80] == 114).all()
    assert (out[-80:] == 114).all()
    assert (out[80:560] == 0).all()


def test_scale_boxes_round_trip_and_clipping():
    original_hw = (240, 320)
    ratio, pad = 2.0, (0, 80)
    boxes_original = np.array([[10.0, 20.0, 100.0, 200.0]], dtype=np.float32)
    boxes_letterbox = boxes_original * ratio + np.array([pad[0], pad[1]] * 2)

    recovered = detector.scale_boxes_to_original(boxes_letterbox, ratio, pad, original_hw)
    np.testing.assert_allclose(recovered, boxes_original, atol=1e-5)

    out_of_frame = np.array([[-100.0, 0.0, 1000.0, 700.0]], dtype=np.float32)
    clipped = detector.scale_boxes_to_original(out_of_frame, ratio, pad, original_hw)
    assert clipped[0, 0] == 0.0
    assert clipped[0, 2] == 320.0
    assert clipped[0, 3] == 240.0


def test_decode_predictions_synthetic():
    raw = _raw_output([(100.0, 200.0, 40.0, 60.0, 0.9), (50.0, 50.0, 10.0, 10.0, 0.1)])

    boxes, scores = detector.decode_predictions(raw, conf_threshold=0.25)

    np.testing.assert_allclose(boxes, [[80.0, 170.0, 120.0, 230.0]])
    np.testing.assert_allclose(scores, [0.9])


def test_decode_predictions_rejects_end2end_output():
    with pytest.raises(ValueError, match="embedded NMS"):
        detector.decode_predictions(np.zeros((1, 300, 6), dtype=np.float32), 0.25)


def test_nms_numpy_suppresses_and_sorts():
    boxes = np.array(
        [
            [10.0, 10.0, 110.0, 110.0],  # overlaps the best box
            [200.0, 200.0, 300.0, 300.0],
            [0.0, 0.0, 100.0, 100.0],  # best
        ],
        dtype=np.float32,
    )
    scores = np.array([0.8, 0.7, 0.9], dtype=np.float32)

    keep = detector.nms_numpy(boxes, scores, iou_threshold=0.45)

    assert keep == [2, 1]


def test_detect_with_fake_session_end_to_end():
    # 240x320 frame -> letterbox ratio 2.0, pad (0, 80)
    frame = np.linspace(20.0, 40.0, 240 * 320, dtype=np.float32).reshape(240, 320)
    raw = _raw_output(
        [
            (100.0, 300.0, 60.0, 60.0, 0.5),
            (320.0, 320.0, 100.0, 100.0, 0.9),
        ]
    )
    session = FakeSession(raw)
    det = OnnxFaceDetector(session=session)

    results = det.detect(frame)

    blob = session.feeds[0]["images"]
    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert blob.min() >= 0.0
    assert blob.max() <= 1.0

    assert len(results) == 2
    assert results[0]["confidence"] == pytest.approx(0.9)
    assert results[1]["confidence"] == pytest.approx(0.5)
    np.testing.assert_allclose(results[0]["box"], [[135.0, 95.0], [185.0, 145.0]], atol=1e-4)
    np.testing.assert_allclose(results[1]["box"], [[35.0, 95.0], [65.0, 125.0]], atol=1e-4)
    for result in results:
        np.testing.assert_array_equal(result["landmarks"], result["box"])
        assert np.asarray(result["box"]).reshape((-1, 2)).shape == (2, 2)
        assert (np.asarray(result["box"]) >= 0).all()


def test_detect_preprocessing_matches_training_reference():
    rng = np.random.default_rng(1)
    frame = (20.0 + 20.0 * rng.random((640, 640))).astype(np.float32)
    frame[5, 7] = np.nan

    session = FakeSession(_raw_output([]))
    det = OnnxFaceDetector(session=session, percentile_samples=None)
    det.detect(frame)

    blob = session.feeds[0]["images"]
    # 640x640 input -> letterbox is the identity, so the blob is u8 / 255
    u8 = np.round(blob[0, 0] * 255.0).astype(np.uint8)
    reference = np.clip(np.nan_to_num(detector.robust_normalize(frame), nan=0.0) * 255.0, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(u8, reference)


def test_detect_returns_empty_list_without_detections():
    det = OnnxFaceDetector(session=FakeSession(_raw_output([])))

    assert det.detect(np.zeros((64, 64), dtype=np.float32)) == []


def test_detect_box_expand_expands_about_center_and_clips():
    raw = _raw_output([(320.0, 320.0, 100.0, 100.0, 0.9)])
    det = OnnxFaceDetector(session=FakeSession(raw), box_expand=2.0)

    frame = np.zeros((640, 640), dtype=np.float32)
    results = det.detect(frame)

    # 100x100 box centered at (320, 320) doubled -> 200x200, same center
    np.testing.assert_allclose(results[0]["box"], [[220.0, 220.0], [420.0, 420.0]], atol=1e-4)

    det_clipped = OnnxFaceDetector(session=FakeSession(raw), box_expand=20.0)
    clipped = det_clipped.detect(frame)[0]["box"]
    assert clipped[0][0] == 0.0
    assert clipped[1][0] == 640.0


def test_constructor_is_lazy_first_detect_resolves_model(monkeypatch):
    class Boom(Exception):
        pass

    def no_download(*_args, **_kwargs):
        raise Boom("model resolution must not happen at construction")

    monkeypatch.setattr(models, "_ModelDownloader", no_download)

    det = OnnxFaceDetector()
    assert det.session is None

    with pytest.raises(Boom):
        det.detect(np.zeros((8, 8), dtype=np.float32))


def test_missing_onnxruntime_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(RuntimeError, match="onnxruntime"):
        OnnxFaceDetector()


def test_select_providers_cuda_fallback_warns():
    class FakeOrtCpuOnly:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    with pytest.warns(RuntimeWarning, match="CUDAExecutionProvider"):
        providers = OnnxFaceDetector._select_providers("cuda", FakeOrtCpuOnly)
    assert providers == ["CPUExecutionProvider"]

    class FakeOrtCuda:
        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    assert OnnxFaceDetector._select_providers("cuda", FakeOrtCuda) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert OnnxFaceDetector._select_providers("cpu", FakeOrtCuda) == ["CPUExecutionProvider"]


def test_ensure_session_warns_when_cuda_provider_fails_to_load(monkeypatch, tmp_path):
    class CpuOnlySession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            class _Input:
                name = "images"

            return [_Input()]

    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(_path, providers=None):
            return CpuOnlySession()

    det = OnnxFaceDetector(model_path=tmp_path / "d.onnx", device="cuda")
    monkeypatch.setattr(det, "_import_onnxruntime", lambda: FakeOrt)

    with pytest.warns(RuntimeWarning, match="failed to initialize"):
        det._ensure_session()

    assert isinstance(det.session, CpuOnlySession)


def test_model_downloader_onnx_suffix(monkeypatch, tmp_path):
    calls = {}

    def fake_download(url, output, quiet=False):
        calls["url"] = url
        calls["output"] = output
        Path(output).write_bytes(b"onnx-weights")
        return output

    monkeypatch.setattr(models.gdown, "download", fake_download)

    downloader = models._ModelDownloader("detector_v2", save_dir=tmp_path, suffix=".onnx")
    model_path = downloader.download_model()

    assert model_path.name == "detector_v2.onnx"
    assert model_path.exists()
    assert models._file_id_map["detector_v2"] in calls["url"]
