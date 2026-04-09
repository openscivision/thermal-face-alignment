# ------------------------------------------------------------------------------------
# Author: Philipp Flotho (philipp.flotho[at]uni-saarland.de)
# ------------------------------------------------------------------------------------

import timm
import torch
import torch.nn as nn
import cv2
from torchvision.transforms import Normalize
import warnings

import numpy as np

from os.path import join

import gdown
import os
from pathlib import Path

MOBILENET = "mobilenetv2_100"
RESNET = "resnet101"

DEVICE = "cuda"


class _ModelDownloader:
    def __init__(self, model_name, save_dir="~/.tfan/models"):
        self.model_name = model_name
        self.save_dir = Path(os.path.expanduser(save_dir))
        self.file_id = _file_id_map.get(model_name)
        if not self.file_id:
            raise ValueError(
                f"Model name '{model_name}' is not valid. Check the file_id_map."
            )
        self.model_url = (
            f"https://drive.google.com/uc?export=download&id={self.file_id}"
        )
        self.model_path = self.save_dir / f"{model_name}.pt"

    def download_model(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.model_path.exists():
            return self.model_path

        print(f"Downloading {self.model_name} with gdown...")
        url = f"https://drive.google.com/uc?export=download&id={self.file_id}"
        output = gdown.download(url, str(self.model_path), quiet=False)
        if output is None or not self.model_path.exists():
            raise RuntimeError(f"Failed to download model '{self.model_name}'")
        return self.model_path


def _get_model(model_name="478"):
    downloader = _ModelDownloader(model_name)
    weights_path = downloader.download_model()
    model_path_final_joint = join(os.path.dirname(weights_path), "joint_converted.pt")
    convert_model(
        weights_path, model_path_final_joint, n_landmarks=int(model_name), mode="JOINT"
    )
    return model_path_final_joint


def _warping_depth(eta, levels, m, n):
    min_dim = min(m, n)
    warping_depth = 0
    d = warping_depth

    for i in range(levels):
        warping_depth += 1
        min_dim = min_dim * eta
        if round(min_dim) < 224:
            break
        d = warping_depth
    return d


def convert_model(model_path_in, model_path_out, mode="", n_landmarks=70):
    model = DMMv2(n_landmarks=n_landmarks)
    dmm = nn.DataParallel(model, device_ids=[0])
    dmm.load_state_dict(
        torch.load(model_path_in, map_location=torch.device("cpu"), weights_only=True),
        strict=False,
    )
    torch.save(dmm.module.state_dict(), model_path_out)


def create_model(model):
    model = timm.create_model(model, pretrained=True)
    model.classifier.requires_grad = False

    def forward_features(x):
        x = model.forward_features(x)
        return model.global_pool(x)

    return model, forward_features


class DMMv2(nn.Module):
    def __init__(self, n_landmarks, use_depth=False):
        super().__init__()
        self.use_depth = False
        self.feature_network = timm.create_model(
            "mobilenetv2_100", pretrained=True, num_classes=0
        )
        self.feature_network.classifier.requires_grad = False

        self.n_landmarks = n_landmarks
        self.fc = nn.Linear(1280, n_landmarks * (self.use_depth + 3))
        nn.init.xavier_uniform_(self.fc.weight)

        self.fc.requires_grad_(True)
        self.act = nn.ReLU()
        self.act_leaky = nn.LeakyReLU()

    def forward(self, x):
        x = self.feature_network.forward_features(x)
        x = self.feature_network.global_pool(x)
        x = self.fc(x)
        x = x.reshape((x.shape[0], self.n_landmarks, self.use_depth + 3))
        x = self.act_leaky(x)

        return x


class ThermalLandmarks:
    """
    Dense face landmark refinement for thermal (and optionally intensity/RGB) inputs.

    This class wraps a two-stage pipeline:
      1) A sparse face detector/tracker (`TFWLandmarker`) produces a face box (and/or sparse landmarks).
      2) A lightweight regression network (`DMMv2`, MobileNetV2 backbone) refines to `n_landmarks`
         dense 2D landmarks plus a per-landmark uncertainty.

    The network operates on 224×224 RGB-like crops. For thermal inputs, a single-channel frame is
    converted into a 3-channel image by clipping to a temperature window and rescaling.

    Parameters
    ----------
    model_path : str or pathlib.Path, optional
        Path to a saved `state_dict` for `DMMv2`. If omitted, weights are downloaded (via gdown)
        for the requested `n_landmarks` and converted to a plain state_dict.
    device : {"cpu", "cuda"}, default "cpu"
        Torch device string. If "cuda", the model may be wrapped with `nn.DataParallel`.
    gpus : list[int], default [0, 1]
        GPU device ids passed to `nn.DataParallel` when `device="cuda"`.
    eta : float, default 0.75
        Pyramid scale factor used by the optional sliding-window mode.
    max_lvl : int, default 0
        Maximum pyramid level (used when `sliding_window=True`).
    stride : int, default 100
        Stride (pixels) for the sliding-window scan when enabled.
    n_landmarks : int, default 478
        Number of landmarks predicted per face.
    normalize : bool, default True
        If True, apply ImageNet normalization to crops before inference. By default, the
        normalization assumes crops are in the 0–255 range (internally divided by 255).
        If you standardize inputs to 0–1 earlier, adjust the normalization accordingly.

    Attributes
    ----------
    face_tracker : object
        Instance of the sparse detector/tracker (defaults to `TFWLandmarker`).
        Created eagerly when available and lazily retried on demand otherwise.
    dmm : torch.nn.Module
        Landmark refinement network (`DMMv2`), possibly wrapped in `nn.DataParallel`.
    last_sparse_lm : object
        Cached sparse detection output from the most recent detector run.

    Notes
    -----
    * `process()` accepts 2D (thermal/intensity) or 3D (color) numpy arrays.
    * For 3D inputs, the current implementation relies on `last_sparse_lm` already being populated
      (i.e., the detector is not rerun on the 3-channel image).
    """

    def __init__(
        self,
        model_path=None,
        device="cpu",
        gpus=[0, 1],
        eta=0.75,
        max_lvl=0,
        stride=100,
        n_landmarks=478,
        normalize=True,
    ):
        self.device = device
        self.n_landmarks = n_landmarks
        self.face_tracker = None
        self._tracker_init_error = None
        self._ensure_face_tracker(raise_on_error=False)

        dmm = DMMv2(n_landmarks=n_landmarks)

        model_path = _get_model(str(n_landmarks)) if not model_path else model_path
        print(model_path)
        dmm.load_state_dict(torch.load(model_path, weights_only=True), strict=False)

        if device == "cuda":
            dmm = nn.DataParallel(dmm, device_ids=gpus)
        dmm.eval()
        dmm.to(device)
        self.dmm = dmm
        self.img_shape = None
        self.eta = eta
        self.max_lvl = max_lvl
        self.stride = stride
        self.last_sparse_lm = None
        if normalize:
            tmp = Normalize(
                mean=torch.tensor([0.4850, 0.4560, 0.4060]),
                std=torch.tensor([0.2290, 0.2240, 0.2250]),
            )
            self.transform = lambda x: tmp(x / 255.0)
        else:
            self.transform = lambda x: x

    def _ensure_face_tracker(self, raise_on_error=True):
        tracker = getattr(self, "face_tracker", None)
        if tracker is not None:
            return tracker

        landmarker_cls = getattr(self, "_landmarker_cls", None)
        try:
            if landmarker_cls is None:
                from neurovc.thermal_landmarks import TFWLandmarker as landmarker_cls
            tracker = landmarker_cls(device=self.device)
        except Exception as exc:
            self._tracker_init_error = exc
            if raise_on_error:
                raise RuntimeError(
                    "Tracker-backed inference requires TFWLandmarker; "
                    "use process(..., sliding_window=True) for the tracker-free path."
                ) from exc
            return None

        self.face_tracker = tracker
        self._tracker_init_error = None
        return tracker

    def process(self, image, sliding_window=False, multi=False, mode="auto"):
        """
        Detect faces (sparse) and predict dense landmarks (refined) for an input frame.

        This method updates `self.last_sparse_lm` when the sparse detector is run (typically for 2D
        thermal/intensity inputs), then refines landmarks by cropping a square face patch and running
        the DMMv2 network.

        Parameters
        ----------
        image : numpy.ndarray
            Input frame as either:
              - H×W (2D): thermal temperatures or grayscale intensities.
              - H×W×3 (3D): color image (assumed OpenCV BGR unless your upstream is RGB).
            Supported value conventions depend on `mode`:
              - temperatures in °C (thermal),
              - pixel intensities in [0, 255],
              - pixel intensities in [0, 1].
        multi : bool, default False
            If True, return results for all detected faces. If False, return only the first face
            (current behavior still returns single-element lists).
        sliding_window : bool, default False
            If True, run a multi-scale sliding-window search to find the lowest-uncertainty crop.
            This path does not run the YOLO/TFW face tracker.
            With `multi=True`, overlapping candidates are merged via NMS.
        mode : {"auto", "temperature", "pixel"}, default "auto"
            How to interpret the numeric range of `image`.

            - "temperature":
                Interpret a 2D input as temperatures in °C. The detector runs on the original 2D frame.
                For the CNN crop, values are clipped to a face-relevant window (e.g. 20–40 °C),
                linearly rescaled, replicated to 3 channels, and mapped to an RGB-like 0–255 image.
            - "pixel":
                Interpret input as pixel intensities. Float inputs in [0, 1] are mapped to [0, 255],
                everything else is clipped to [0, 255].
            - "auto":
                Infer from dtype/shape and robust statistics

        Returns
        -------
        landmarks : list[numpy.ndarray] or tuple[numpy.ndarray, numpy.ndarray]
            If `sliding_window=False`:
                A list of per-face landmark arrays, each of shape (n_landmarks, 2), in pixel coordinates
                of the original input image.
            If `sliding_window=True`:
                If `multi=False`, `(lm, scores)` where `lm` has shape (n_landmarks, 2).
                If `multi=True`, a list of per-face landmark arrays.
        confidences : list[numpy.ndarray] or numpy.ndarray
            If `sliding_window=False`:
                A list of per-face confidence vectors, each of shape (n_landmarks,).
            If `sliding_window=True`:
                If `multi=False`, `scores` is an uncertainty vector of shape (n_landmarks,).
                If `multi=True`, a list of per-face uncertainty vectors.

        Raises
        ------
        ValueError
            If the requested mode is incompatible with the input shape (e.g. "temperature" with 3D input),
            or if inference cannot proceed (e.g. missing cached detections for 3D inputs).
        """
        if mode not in {"auto", "temperature", "pixel"}:
            raise ValueError(
                f"mode must be one of {{'auto','temperature','pixel'}}, got {mode!r}"
            )

        img = np.asarray(image)

        def _robust_p1_p99(a, max_samples=200_000):
            flat = np.asarray(a).reshape(-1)
            if flat.size > max_samples:
                step = max(1, flat.size // max_samples)
                flat = flat[::step]
            p1, p99 = np.nanpercentile(flat, [1, 99])
            return float(p1), float(p99)

        def _to_0_255(a):
            a = np.asarray(a)
            if np.issubdtype(a.dtype, np.integer):
                return np.clip(a, 0, 255).astype(np.float32)
            a = a.astype(np.float32)
            p1, p99 = _robust_p1_p99(a)
            if (p99 <= 1.0 + 1e-6) and (p1 >= -1e-6):
                return np.clip(a, 0.0, 1.0) * 255.0
            return np.clip(a, 0.0, 255.0)

        if img.ndim == 3 and img.shape[2] == 3:
            warnings.warn(
                "process(): got a 3-channel image; interpreting it as thermal by averaging channels "
                "(this is a temporary behavior).",
                RuntimeWarning,
                stacklevel=2,
            )
            img = img.astype(np.float32).mean(axis=2)

        if mode == "auto":
            if img.ndim == 2:
                if np.issubdtype(img.dtype, np.integer):
                    mode = "pixel"
                else:
                    p1, p99 = _robust_p1_p99(img)
                    if (p99 <= 1.0 + 1e-6) and (p1 >= -1e-6):
                        mode = "pixel"
                    elif (p99 > 1.0 + 1e-6) and (p99 <= 80.0) and (p1 >= -10.0):
                        mode = "temperature"
                    else:
                        mode = "pixel"
            else:
                raise ValueError(
                    f"Expected 2D thermal frame after preprocessing, got shape {img.shape!r}"
                )

        if img.ndim != 2:
            raise ValueError(
                f"Expected 2D thermal frame after preprocessing, got shape {img.shape!r}"
            )

        img2d = img.astype(np.float32)

        if mode == "temperature":
            x = np.clip(img2d, 20.0, 40.0)
            denom = x.max() - x.min()
            if denom < 1e-6:
                x = np.zeros_like(x, dtype=np.float32)
            else:
                x = (x - x.min()) / denom
            image = np.repeat((x * 255.0)[..., None], 3, axis=2).astype(np.float32)
        else:  # "pixel"
            x = _to_0_255(img2d)
            image = np.repeat(x[..., None], 3, axis=2).astype(np.float32)

        if sliding_window:
            candidates = self._sliding_window_candidates(image)
            if multi:
                if self.stride > 112:
                    warnings.warn(
                        "sliding_window=True with multi=True is suboptimal for stride > 112; "
                        "reduce the stride for more reliable overlap before NMS.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                kept = self._nms_sliding_candidates(candidates)
                return (
                    [candidate["landmarks"] for candidate in kept],
                    [candidate["uncertainties"] for candidate in kept],
                )
            best = min(candidates, key=lambda candidate: candidate["mean_uncertainty"])
            return best["landmarks"], best["uncertainties"]

        tracker = self._ensure_face_tracker()

        # sparse detector runs on the 2D frame (temperature or intensity)
        self.last_sparse_lm = tracker.detect(img2d)

        if not sliding_window:
            if multi:
                return self.get_landmarks_multi(image)
            return self.get_landmarks_single(image)

        raise AssertionError("unreachable")

    def _refine_landmarks(self, img, lm_scaled):
        x_coords = lm_scaled[:, 0]
        y_coords = lm_scaled[:, 1]

        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)

        largest_side = max(y_max - y_min, x_max - x_min) * 1.25
        y_center = (y_max + y_min) / 2
        x_center = (x_max + x_min) / 2

        padding = int(largest_side // 2)

        padded_img = cv2.copyMakeBorder(
            img,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

        x_start = int(x_center - largest_side / 2 + padding)
        y_start = int(y_center - largest_side / 2 + padding)
        x_end = int(x_center + largest_side / 2 + padding)
        y_end = int(y_center + largest_side / 2 + padding)

        patch = padded_img[y_start:y_end, x_start:x_end]

        x = cv2.resize(patch, (224, 224))
        x = (
            torch.from_numpy(x)
            .to(torch.float32)
            .to(self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        with torch.no_grad():
            cropped_transformed = self.transform(x)
            refined_lm = self.dmm(cropped_transformed)

        refined_lm_scaled = (
            (refined_lm[..., :-1] * largest_side).cpu().detach().squeeze().numpy()
        )
        refined_lm_scaled += np.array([[x_start - padding, y_start - padding]])
        confidences = refined_lm[..., -1].cpu().detach().squeeze().numpy()
        return refined_lm_scaled, confidences

    def get_landmarks_single(self, img):
        results = self.last_sparse_lm
        if len(results) == 0:
            return -np.ones((self.n_landmarks, 2)), np.zeros(self.n_landmarks)
        lm_scaled = results[0]["landmarks"]
        lm_scaled = np.array(lm_scaled).reshape((-1, 2))

        bbox = results[0]["box"]
        bbox = np.array(bbox).reshape((-1, 2))
        if lm_scaled is None:
            raise ValueError("Use mediapipe for dense RGB landmarks!")
        if -1 in lm_scaled:
            return lm_scaled, np.zeros(lm_scaled.shape[0])
        lm_scaled, confidences = self._refine_landmarks(img, bbox)

        return [lm_scaled], [confidences]

    def get_landmarks_multi(self, img):
        results = self.last_sparse_lm
        if len(results) == 0:
            return [], []
        landmarks = []
        confidences_all = []
        for result in results:
            lm_scaled = result["landmarks"]
            lm_scaled = np.array(lm_scaled).reshape((-1, 2))
            bbox = result["box"]
            bbox = np.array(bbox).reshape((-1, 2))
            if lm_scaled is None:
                raise ValueError("Use mediapipe for dense RGB landmarks!")
            if -1 in lm_scaled:
                landmarks.append(-np.ones((self.n_landmarks, 2)))
                confidences_all.append(np.zeros(self.n_landmarks))
                continue
            refined, confidences = self._refine_landmarks(img, bbox)
            landmarks.append(refined)
            confidences_all.append(confidences)
        return landmarks, confidences_all

    def _sliding_window_level_candidates(self, img, stride=50):
        img_dims = img.shape
        y_pad = 224 - img_dims[0] % 224
        x_pad = 224 - img_dims[1] % 224

        y_pad_l = y_pad // 2
        y_pad_r = y_pad // 2 + y_pad % 2

        x_pad_l = x_pad // 2
        x_pad_r = x_pad // 2 + y_pad % 2

        pad = (0, 0, x_pad_l, x_pad_r, y_pad_l, y_pad_r)

        with torch.no_grad():
            x = torch.nn.functional.pad(
                torch.from_numpy(img).to(torch.float32), pad
            ).to(self.device)
            img_unfold = x.unfold(0, 224, stride).unfold(1, 224, stride)
            s = img_unfold.shape
            img_unfold = img_unfold.reshape((s[0] * s[1],) + s[2:])
            lm = self.dmm(self.transform(img_unfold))

            mean_uncertainties = lm[..., -1].mean(1)
            y_idx = torch.arange(s[0], device=self.device).repeat_interleave(s[1])
            x_idx = torch.arange(s[1], device=self.device).repeat(s[0])
            offsets = torch.stack([stride * x_idx, stride * y_idx], dim=1).to(
                torch.float32
            )
            pad_offset = torch.tensor(
                [x_pad_l, y_pad_l], device=self.device, dtype=torch.float32
            )
            lm_out = lm[..., :-1] * 224 + offsets[:, None, :]
            lm_out = lm_out - pad_offset.view(1, 1, 2)
            boxes = torch.cat(
                (
                    offsets - pad_offset.view(1, 2),
                    offsets - pad_offset.view(1, 2) + 224,
                ),
                dim=1,
            )

        candidates = []
        for idx in range(lm.shape[0]):
            candidates.append(
                {
                    "landmarks": lm_out[idx].cpu().detach().numpy(),
                    "uncertainties": lm[idx, :, -1].cpu().detach().numpy(),
                    "mean_uncertainty": mean_uncertainties[idx].item(),
                    "box": boxes[idx].cpu().detach().numpy(),
                }
            )
        return candidates

    def _sliding_window_candidates(self, image):
        img_shape = image.shape[:2]
        wp = _warping_depth(self.eta, 100, *img_shape)
        all_candidates = []
        scale = np.array([1.0, 1.0], dtype=np.float32)

        for i in range(wp, self.max_lvl - 1, -1):
            lvl_factor = self.eta**i
            size = (
                int(round(img_shape[1] * lvl_factor)),
                int(round(img_shape[0] * lvl_factor)),
            )
            img = cv2.resize(image, size)
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)
            scale[0] = img_shape[1] / size[0]
            scale[1] = img_shape[0] / size[1]
            level_candidates = self._sliding_window_level_candidates(
                img.astype(np.float32), stride=self.stride
            )
            for candidate in level_candidates:
                landmarks = candidate["landmarks"] * scale
                box = candidate["box"].copy()
                box[[0, 2]] *= scale[0]
                box[[1, 3]] *= scale[1]
                box[[0, 2]] = np.clip(box[[0, 2]], 0.0, img_shape[1])
                box[[1, 3]] = np.clip(box[[1, 3]], 0.0, img_shape[0])
                all_candidates.append(
                    {
                        "landmarks": landmarks,
                        "uncertainties": candidate["uncertainties"],
                        "mean_uncertainty": candidate["mean_uncertainty"],
                        "box": box,
                    }
                )

        return all_candidates

    def _nms_sliding_candidates(self, candidates, iou_threshold=0.5):
        if len(candidates) <= 1:
            return candidates

        nms = getattr(self, "_torchvision_nms", None)
        if nms is None:
            from torchvision.ops import nms

            self._torchvision_nms = nms

        boxes = torch.as_tensor(
            np.stack([candidate["box"] for candidate in candidates], axis=0),
            dtype=torch.float32,
        )
        scores = torch.as_tensor(
            [-candidate["mean_uncertainty"] for candidate in candidates],
            dtype=torch.float32,
        )
        keep = self._torchvision_nms(boxes, scores, iou_threshold)
        keep_idx = keep.cpu().tolist()
        return [candidates[idx] for idx in keep_idx]

    def get_landmarks(self, img, stride=50, refine=True):
        """Sliding window implementation for the landmarks"""
        candidates = self._sliding_window_level_candidates(img, stride=stride)
        best = min(candidates, key=lambda candidate: candidate["mean_uncertainty"])
        return best["landmarks"], best["mean_uncertainty"], best["uncertainties"]


_file_id_map = {
    "478": "1DZU3OOACp8gqxCxotZGwe3_gyNJCoY1p",
    "70": "1DqBVVmw9NscDELsnCxB4Pt9ltVUuNoWR",
}
