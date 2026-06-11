# ------------------------------------------------------------------------------------
# Author: Philipp Flotho (philipp.flotho[at]uni-saarland.de)
# ------------------------------------------------------------------------------------
"""ONNX Runtime backend for the T-FAKE-trained YOLO face detector (the "v2" tracker)."""

import warnings
from pathlib import Path

import cv2
import numpy as np

DETECTOR_MODEL_NAME = "detector_v2"
DETECTOR_INPUT_SIZE = 640
PAD_VALUE = 114


def _robust_range(x, lower=1.0, upper=99.0, max_samples=None):
    """Return the (lower, upper) percentile range of the finite values of `x`.

    For frames larger than `max_samples` pixels the percentiles are estimated
    on a strided subsample; `max_samples=None` computes exact percentiles.
    Returns (0.0, 1.0) when no finite values exist (yields an all-zero frame
    downstream).
    """
    flat = np.asarray(x, dtype=np.float32).reshape(-1)
    if max_samples is not None and flat.size > max_samples:
        flat = flat[:: flat.size // max_samples]
    finite_mask = np.isfinite(flat)
    if not finite_mask.all():
        if not finite_mask.any():
            # the subsample may have missed the finite pixels; retry in full
            flat = np.asarray(x, dtype=np.float32).reshape(-1)
            finite_mask = np.isfinite(flat)
            if not finite_mask.any():
                return 0.0, 1.0
        flat = flat[finite_mask]
    lo, hi = np.percentile(flat, [lower, upper])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def robust_normalize(image, lower=1.0, upper=99.0, max_samples=None):
    """Percentile-normalize a frame to [0, 1].

    Mirrors the preprocessing the detector was trained with, so it behaves
    identically for temperature frames (°C), 0-255 intensities, or raw camera
    counts. Non-finite values are excluded from the percentile estimate.
    """
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if not np.isfinite(x).any():
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = _robust_range(x, lower, upper, max_samples)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def letterbox(img, new_size=DETECTOR_INPUT_SIZE, pad_value=PAD_VALUE):
    """Resize keeping aspect ratio and center-pad to a square (ultralytics-compatible).

    Returns the padded image, the resize ratio, and the integer (left, top)
    padding offsets actually applied, so the inverse mapping is exact.
    """
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    new_unpad = (max(1, int(round(w * r))), max(1, int(round(h * r))))
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw = (new_size - new_unpad[0]) / 2
    dh = (new_size - new_unpad[1]) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    return img, r, (left, top)


def decode_predictions(raw, conf_threshold):
    """Decode a raw single-class YOLO head output (1, 5, N) into xyxy boxes and scores."""
    p = np.asarray(raw, dtype=np.float32)
    if p.ndim == 3 and p.shape[0] == 1:
        p = p[0]
    if p.ndim != 2 or p.shape[0] != 5:
        raise ValueError(
            f"Unexpected detector output shape {p.shape}; expected (1, 5, N) raw "
            "single-class YOLO output (cx, cy, w, h, conf). Models exported with "
            "embedded NMS (nms=True / end2end) are not supported; re-export with "
            "nms=False."
        )
    keep = p[4] >= conf_threshold
    cx, cy, w, h, conf = p[:, keep]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes, conf


def scale_boxes_to_original(boxes, ratio, pad_lt, original_hw):
    """Map xyxy boxes from letterbox space back to original image coordinates."""
    left, top = pad_lt
    h, w = original_hw[:2]
    out = np.array(boxes, dtype=np.float32, copy=True)
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - left) / ratio, 0.0, float(w))
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - top) / ratio, 0.0, float(h))
    return out


def nms_numpy(boxes, scores, iou_threshold):
    """Greedy non-maximum suppression; returns kept indices sorted by score descending."""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


class OnnxFaceDetector:
    """YOLO face detector trained on T-FAKE, run via ONNX Runtime.

    Drop-in tracker backend for `ThermalLandmarks`: `detect(img2d)` takes the raw
    2D thermal/intensity frame, applies the same robust percentile normalization
    used during training, and returns detections sorted by confidence (best face
    first) as dicts with "landmarks" (the two box corners), "box" ([[x1, y1],
    [x2, y2]]) and "confidence".

    The ONNX model is downloaded (or loaded from `model_path`) lazily on the
    first `detect()` call; constructing the detector never touches the network.

    Parameters
    ----------
    model_path : str or Path, optional
        Local path to the detector `.onnx` file. If omitted, the model is
        downloaded on first use via the package model downloader.
    device : str, default "cpu"
        "cuda" selects the CUDA execution provider when available (requires
        onnxruntime-gpu), otherwise falls back to CPU with a warning.
    conf_threshold : float, default 0.25
        Minimum detection confidence.
    iou_threshold : float, default 0.45
        IoU threshold for non-maximum suppression.
    box_expand : float, default 1.0
        Expansion factor applied to each box about its center (clipped to the
        frame). Useful to calibrate the downstream crop against the legacy TFW
        box extents.
    input_size : int, default 640
        Detector input resolution (letterboxed).
    percentile_samples : int or None, default None
        None computes the normalization percentiles exactly, matching the
        training-time preprocessing bit for bit. Set to e.g. 50_000 to
        estimate them on a strided subsample instead — ~0.5 ms/frame faster
        on VGA-sized frames and growing with resolution, at < 1% box IoU
        deviation.
    providers : list, optional
        Explicit ONNX Runtime provider list; overrides the `device` mapping.
    session : object, optional
        Pre-built InferenceSession-like object (testing / advanced use).
    """

    def __init__(
        self,
        model_path=None,
        device="cpu",
        conf_threshold=0.25,
        iou_threshold=0.45,
        box_expand=1.0,
        input_size=DETECTOR_INPUT_SIZE,
        percentile_samples=None,
        providers=None,
        session=None,
    ):
        if session is None:
            # fail fast on a missing dependency so _ensure_face_tracker degrades
            # as gracefully as it does for the legacy neurovc backend
            self._import_onnxruntime()
        self.model_path = Path(model_path) if model_path is not None else None
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.box_expand = box_expand
        self.input_size = input_size
        self.percentile_samples = percentile_samples
        self.providers = providers
        self.session = session
        self._input_name = None

    @staticmethod
    def _import_onnxruntime():
        try:
            import onnxruntime
        except Exception as exc:
            raise RuntimeError(
                "The v2 face detector requires onnxruntime; install it with " "'pip install onnxruntime' (or onnxruntime-gpu for CUDA)."
            ) from exc
        return onnxruntime

    @staticmethod
    def _select_providers(device, ort):
        available = ort.get_available_providers()
        if device == "cuda":
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            warnings.warn(
                "device='cuda' requested but CUDAExecutionProvider is not available "
                "(install onnxruntime-gpu); running face detection on CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
        return ["CPUExecutionProvider"]

    def _ensure_session(self):
        if self.session is None:
            ort = self._import_onnxruntime()
            model_path = self.model_path
            if model_path is None:
                from tfan.core.models import _ModelDownloader

                model_path = _ModelDownloader(DETECTOR_MODEL_NAME, suffix=".onnx").download_model()
            providers = self.providers or self._select_providers(self.device, ort)
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in self.session.get_providers():
                # ORT silently drops providers that fail to initialize
                # (e.g. CUDA/cuDNN version mismatch)
                warnings.warn(
                    "CUDAExecutionProvider failed to initialize (CUDA/cuDNN version " "mismatch?); face detection is running on CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if self._input_name is None:
            self._input_name = self.session.get_inputs()[0].name
        return self.session

    def detect(self, img2d):
        session = self._ensure_session()

        img2d = np.asarray(img2d)
        # allocation-light, bit-exact equivalent of the training-time preprocessing:
        # clip(robust_normalize(x) * 255, 0, 255).astype(uint8), replicated to 3ch
        # (same float32 operation order as robust_normalize, so truncation matches)
        img_f = img2d.astype(np.float32, copy=False)
        lo, hi = _robust_range(img_f, max_samples=self.percentile_samples)
        scaled = (img_f - lo) / (hi - lo)
        np.nan_to_num(scaled, copy=False, nan=0.0)
        np.clip(scaled, 0.0, 1.0, out=scaled)
        scaled *= 255.0
        u8 = cv2.cvtColor(scaled.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        img_in, ratio, pad_lt = letterbox(u8, self.input_size)
        blob = (img_in.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        raw = session.run(None, {self._input_name: blob})[0]
        boxes, scores = decode_predictions(raw, self.conf_threshold)
        if boxes.shape[0] == 0:
            return []

        boxes = scale_boxes_to_original(boxes, ratio, pad_lt, img2d.shape[:2])
        keep = nms_numpy(boxes, scores, self.iou_threshold)

        results = []
        for idx in keep:
            corners = self._expand_box(boxes[idx], img2d.shape[:2])
            results.append(
                {
                    "landmarks": corners.copy(),
                    "box": corners,
                    "confidence": float(scores[idx]),
                }
            )
        return results

    def _expand_box(self, box_xyxy, original_hw):
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
        if self.box_expand != 1.0:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            half_w = (x2 - x1) / 2.0 * self.box_expand
            half_h = (y2 - y1) / 2.0 * self.box_expand
            x1, x2 = cx - half_w, cx + half_w
            y1, y2 = cy - half_h, cy + half_h
        h, w = original_hw[:2]
        x1, x2 = np.clip([x1, x2], 0.0, float(w))
        y1, y2 = np.clip([y1, y2], 0.0, float(h))
        return np.array([[x1, y1], [x2, y2]], dtype=np.float32)
