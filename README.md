[![PyPI - Version](https://img.shields.io/pypi/v/thermal-face-alignment)](https://pypi.org/project/thermal-face-alignment/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/thermal-face-alignment)](https://pypi.org/project/thermal-face-alignment/)
[![PyPI - License](https://img.shields.io/pypi/l/thermal-face-alignment)](LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/thermal-face-alignment)](https://pypistats.org/packages/thermal-face-alignment)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/thermal-face-alignment?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=all+time+downloads)](https://pepy.tech/projects/thermal-face-alignment)


# Thermal-facial-alignment network (TFAN) trained on the T-FAKE dataset

## Using the landmarker

Install and run:

```bash
pip install thermal-face-alignment
```

```python
import cv2
from tfan import ThermalLandmarks

# Read a thermal image (grayscale)
image = cv2.imread("thermal.png", cv2.IMREAD_GRAYSCALE)

# Initialize landmarker (downloads weights on first use)
landmarker = ThermalLandmarks(device="cpu", n_landmarks=70)

landmarks, confidences = landmarker.process(image)
```
![TFW Example Prediction](https://raw.githubusercontent.com/openscivision/thermal-face-alignment/e052eebeb04176177e919360a7fe8b3e65a7d85b/img/tfw-sample_tfan.png)
*Predicted 70 and 478 point landmarks on an example from the [TFW Dataset](https://github.com/IS2AI/TFW).*


![Landmarking example](https://raw.githubusercontent.com/openscivision/thermal-face-alignment/e052eebeb04176177e919360a7fe8b3e65a7d85b/img/tiv_lm70_478_example.gif)
*Predicted 70 and 478 point landmarks on an example from the [BU-TIV Benchmark](https://csr.bu.edu/BU-TIV/BUTIV.html).*


## Practical Usage

The ThermalLandmarks wraps a landmarker trained on T-FAKE either with a tracker-free sliding window selecting the face with lowest uncertainty or via a bbox computed with a smaller model. The default face detector ("v2") is a YOLO11 network trained on T-FAKE, run via ONNX Runtime and downloaded automatically on first use; the legacy TFW detector remains available via `tracker="tfw"`.

> **Breaking change:** `neurovc` is no longer installed by default. To keep using the legacy TFW
> tracker, install the extra: `pip install "thermal-face-alignment[tfw]"`.

Please note that we trained our network with temperature value range of 20°C to 40°C. While our implementation performs an automatic rescaling, please make sure that you adapt our landmarker options based on the input pixel values.

### Initialization options

```python
ThermalLandmarks(
    model_path=None,
    device="cpu",
    gpus=[0, 1],
    eta=0.75,
    max_lvl=0,
    stride=100,
    n_landmarks=478, # 478 or 70 point landmarks are supported
    normalize=True,
    tracker="v2",
    detector_path=None,
)
```

- **`model_path`** (`str` or `Path`, optional)
  Path to a pretrained DMMv2 model (`state_dict`).
  If omitted, pretrained weights matching `n_landmarks` are downloaded automatically.

- **`device`** (`"cpu"` or `"cuda"`, default `"cpu"`)
  Torch device used for inference. When using `"cuda"`, the model may be wrapped in `DataParallel`.

- **`gpus`** (`list[int]`, default `[0, 1]`)
  GPU device IDs used when `device="cuda"`.

- **`n_landmarks`** (`int`, default `478`)
  Number of facial landmarks predicted per face.
  Choices:
  - `70` — sparse landmarks following the Face Synthetic convention of (Wood et al., 2021).
  - `478` — dense landmarks following the [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker?hl=de) face mesh convention.

- **`normalize`** (`bool`, default `True`)
  Apply ImageNet normalization to cropped face patches before inference.
  Assumes inputs are scaled to `[0, 255]`.

- **`eta`** (`float`, default `0.75`)
  Pyramid scale factor used in sliding-window mode.

- **`max_lvl`** (`int`, default `0`)
  Maximum pyramid level for multi-scale sliding-window inference.

- **`stride`** (`int`, default `100`)
  Pixel stride used during sliding-window scanning.

- **`tracker`** (`str` or detector instance, default `"v2"`)
  Face detector backend used for the bbox-based (non-sliding-window) path:
  - `"v2"` — YOLO11 face detector trained on T-FAKE, run via ONNX Runtime (downloaded on first use).
  - `"tfw"` — legacy TFW detector (requires `pip install "thermal-face-alignment[tfw]"`).
  - a detector instance with a `detect(frame)` method, e.g. for custom thresholds:

  ```python
  from tfan import OnnxFaceDetector, ThermalLandmarks

  detector = OnnxFaceDetector(conf_threshold=0.5, iou_threshold=0.45)
  landmarker = ThermalLandmarks(n_landmarks=70, tracker=detector)
  ```

  With `tracker="v2"`, the sparse `"landmarks"` entries in `landmarker.last_sparse_lm` are the two
  bounding-box corners; the 5-point sparse landmarks are only produced by `tracker="tfw"`.
  For GPU detection install `onnxruntime-gpu`; with the default CPU build, `device="cuda"` warns
  and runs the detection on CPU (the landmark network still uses CUDA via torch).

- **`detector_path`** (`str` or `Path`, optional)
  Local path to the v2 detector `.onnx` file, skipping the automatic download (`tracker="v2"` only).

---

### Inference options

```python
landmarks, confidences = landmarker.process(
    image,
    sliding_window=False,
    multi=False,
    mode="auto",
    upsample_factor=1.0,
    nms_iou_threshold=0.5,
    uncertainty_factor=None,
    top_k=None,
)
```

- **`image`** (`numpy.ndarray`)
  Input frame:
  - `H×W`: thermal or grayscale image
  - `H×W×3`: RGB/BGR image

- **`mode`** (`"auto" | "temperature" | "pixel"`, default `"auto"`)
  Controls how numeric values are interpreted:
  - `"temperature"`: 2D thermal image in °C
  - `"pixel"`: pixel intensities in `[0, 255]` or `[0, 1]`
  - `"auto"`: inferred from dtype and value range

- **`multi`** (`bool`, default `False`)
  If `True`, return landmarks for all detected faces.
  If `False`, only the first face is returned.

- **`sliding_window`** (`bool`, default `False`)
  Enable multi-scale sliding-window inference.
  This path does not run the YOLO/TFW face tracker.
  With `multi=True`, overlapping candidates are merged with NMS.
  **Note:** `stride > 112` is suboptimal for `multi=True`.

- **`upsample_factor`** (`float`, default `1.0`)
  Optionally upsample the input before inference. Values larger than `1.0` can help with very
  small faces. Returned landmarks stay in the original image coordinates.

- **`nms_iou_threshold`** (`float`, default `0.5`)
  IoU threshold used by sliding-window NMS when `sliding_window=True` and `multi=True`.

- **`uncertainty_factor`** (`float`, optional)
  Optional post-NMS pruning factor for sliding-window multi-face inference. Keeps detections whose
  mean uncertainty is at most `best_mean_uncertainty * uncertainty_factor`.

- **`top_k`** (`int`, optional)
  Optional maximum number of sliding-window detections to keep after NMS and uncertainty filtering.

---

### Outputs

- **`landmarks`**
  Pixel coordinates in the original image:
  - List of `(n_landmarks, 2)` arrays (multi-face)
  - Single `(n_landmarks, 2)` array (sliding window)

- **`confidences`**
  Per-landmark uncertainty scores of shape `(n_landmarks,)`


# Background

This landmarker is an implementation of our work presented in our [CVPR paper]([https://<paper-url>](https://openaccess.thecvf.com/content/CVPR2025/papers/Flotho_T-FAKE_Synthesizing_Thermal_Images_for_Facial_Landmarking_CVPR_2025_paper.pdf)) on thermal landmarking ([Main GitHub](https://github.com/phflot/tfake)). The default face detection is a YOLO11 network trained on the T-FAKE dataset. The [TFW face detector](https://github.com/IS2AI/TFW), which we originally employed for our initial face detection as it performed very well in our benchmark, remains available as the legacy `tracker="tfw"` backend. Please note that this library is meant for research purposes only.

## Landmarker Performance on our Charlotte Benchmark
![landmarks](https://raw.githubusercontent.com/openscivision/thermal-face-alignment/7221fdc136ac84f2ce5a304b45b04bdd4bc7405b/img/landmarks.jpg)

## Training Dataset

![Image](https://raw.githubusercontent.com/openscivision/thermal-face-alignment/7221fdc136ac84f2ce5a304b45b04bdd4bc7405b/img/fake-thermal.jpg)

We trained our landmarker on our custom-made T-FAKE dataset consisting of synthetic thermal images. To download the original color images, sparse annotations, and segmentation masks for the dataset, please use the links in the [FaceSynthetics repository](https://github.com/microsoft/FaceSynthetics).

Our dataset has been generated for a warm and for a cold condition. Each dataset can be downloaded separately as

- A small sample with 100 images from [here warm](https://drive.google.com/file/d/1-Y40_wqVV5WM1swEpjFTJiB8sGdZtQwR/view?usp=sharing) and [here cold](https://drive.google.com/file/d/1-_-RHg7ZDzFFtoyeXJsyrdtgkcnJ3FMR/view?usp=sharing)
- A medium sample with 1,000 images from [here warm](https://drive.google.com/file/d/1-NcsaNa6dbfmQ0l6UjmwZSJWDsUFM4vW/view?usp=sharing) and [here cold](https://drive.google.com/file/d/1-PqPR86GDj5LB_6PZKlek6o6FNbkf7Fo/view?usp=sharing)
- The full dataset with 100,000 images from [here warm](https://drive.google.com/file/d/1-3-OC-VYL14uyLA4Vi9DpwDlkauuNh7K/view?usp=sharing) and [here cold](https://drive.google.com/file/d/1wh25Yi9sT-0j6qXz0JlHUtIIbLAYUnrZ/view?usp=sharing)
- The dense annotations are available from [here](https://drive.google.com/file/d/1-lMYaok0xbfQyBTxj6dcuxT1iryU7TOs/view?usp=sharing)

## Pre-trained models

The models for the thermalization as well as the landmarkers can be downloaded from [here](https://drive.google.com/drive/folders/1-ppKS4xuBY-EbmGCkvKTLYMXHA3lK8R8?usp=sharing).

## License

Our landmarking methods and the training dataset are licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International](LICENSE.txt) license as it is derived from the [FaceSynthetics dataset](https://github.com/microsoft/FaceSynthetics).

## Citation

If you use this code for your own work, please cite our paper:

> P. Flotho, M. Piening, A. Kukleva and G. Steidl, “T-FAKE: Synthesizing Thermal Images for Facial Landmarking,” Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), 2025. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Flotho_T-FAKE_Synthesizing_Thermal_Images_for_Facial_Landmarking_CVPR_2025_paper.html)

BibTeX entry
```
@InProceedings{tfake2025_CVPR,
    author    = {Flotho, Philipp and Piening, Moritz and Kukleva, Anna and Steidl, Gabriele},
    title     = {T-FAKE: Synthesizing Thermal Images for Facial Landmarking},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {26356-26366}
}
```

When using the legacy `tracker="tfw"` backend, the thermal face bounding box detection uses the TFW landmarker model, please additionally cite:

> Kuzdeuov, A., Aubakirova, D., Koishigarina, D., & Varol, H. A. (2022). TFW: Annotated Thermal Faces in the Wild Dataset. *IEEE Transactions on Information Forensics and Security*, 17, 2084–2094. https://doi.org/10.1109/TIFS.2022.3177949

```bibtex
@article{9781417,
    author={Kuzdeuov, Askat and Aubakirova, Dana and Koishigarina, Darina and Varol, Huseyin Atakan},
    journal={IEEE Transactions on Information Forensics and Security},
    title={TFW: Annotated Thermal Faces in the Wild Dataset},
    year={2022},
    volume={17},
    pages={2084-2094},
    doi={10.1109/TIFS.2022.3177949}
}
```
