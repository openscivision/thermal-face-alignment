# ------------------------------------------------------------------------------------
# AUTHOR STATEMENT
#
# Author: Philipp Flotho (philipp.flotho[at]uni-saarland.de)
#
# For citation, please refer to the project README.
# If you believe any confidential or proprietary content is included, please notify me.
#
# Copyright (c) 2025, Philipp Flotho
# ------------------------------------------------------------------------------------

import timm
import torch
import torch.nn as nn
import cv2
from torchvision.transforms import Normalize

import numpy as np

from os.path import join

import gdown
import os
from pathlib import Path

MOBILENET = "mobilenetv2_100"
RESNET = "resnet101"

DEVICE = "cuda"


class _ModelDownloader:
    def __init__(self, model_name, save_dir="~/.neurovc/models"):
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
        landmarker_cls = getattr(self, "_landmarker_cls", None)
        if landmarker_cls is None:
            from neurovc.thermal_landmarks import TFWLandmarker as landmarker_cls
        self.face_tracker = landmarker_cls()

        dmm = DMMv2(n_landmarks=n_landmarks)
        self.n_landmarks = n_landmarks

        model_path = _get_model(str(n_landmarks)) if not model_path else model_path
        print(model_path)
        dmm.load_state_dict(torch.load(model_path, weights_only=True), strict=False)

        if device == "cuda":
            dmm = nn.DataParallel(dmm, device_ids=gpus)
        dmm.eval()
        dmm.to(device)
        self.device = device
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

    def process(self, image, sliding_window=False, multi=False):
        if self.img_shape is None:
            img_shape = image.shape[:2]
            wp = _warping_depth(self.eta, 100, *img_shape)

        if len(image.shape) == 2:
            image = image.astype(float)
            self.last_sparse_lm = self.face_tracker.detect(image)
            image = np.clip(image, 20, 40)
            image -= image.min()
            image /= image.max()
            image = np.stack([image, image, image], 2) * 255
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not sliding_window:
            if multi:
                return self.get_landmarks_multi(image)
            return self.get_landmarks_single(image)

        best_score = np.inf
        lm = None
        best_scores = None
        for i in range(wp, self.max_lvl - 1, -1):
            lvl_factor = self.eta**i
            size = (
                int(round(img_shape[1] * lvl_factor)),
                int(round(img_shape[0] * lvl_factor)),
            )
            img = cv2.resize(image, size)
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)
            hx = img_shape[1] / size[0]
            hy = img_shape[0] / size[1]
            lm_lvl, score, scores = self.get_landmarks(
                img.astype(np.float32), stride=self.stride
            )
            if score < best_score:
                best_score = score
                best_scores = scores
                lm = lm_lvl * np.expand_dims(np.array([hx, hy]), 0)

        return lm, best_scores

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

        return lm_scaled, confidences

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

    def get_landmarks(self, img, stride=50, refine=True):
        """Sliding window implementation for the landmarks"""

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

            best_scores = lm[..., -1].mean(1)
            best_score_idx = best_scores.argmin().item()
            best_score = best_scores[best_score_idx].item()
            offset = (
                torch.Tensor(
                    [
                        stride * (best_score_idx % s[1]),
                        stride * (best_score_idx // s[1]),
                    ]
                )
                .unsqueeze(0)
                .to(self.device)
            )
            lm = lm[best_score_idx]
            lm_out = lm[:, :-1] * 224 + offset
            lm_out = lm_out - torch.tensor([x_pad_l, y_pad_l]).unsqueeze(0).to(
                self.device
            )

        lm_scaled = lm_out.cpu().detach().numpy()
        confidences = lm[..., -1].cpu().detach().numpy()

        return lm_scaled, best_score, confidences


_file_id_map = {
    "478": "1DZU3OOACp8gqxCxotZGwe3_gyNJCoY1p",
    "70": "1DqBVVmw9NscDELsnCxB4Pt9ltVUuNoWR",
}
