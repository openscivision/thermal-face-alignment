[![PyPI - Version](https://img.shields.io/pypi/v/thermal-face-alignment)](https://pypi.org/project/thermal-face-alignment/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/thermal-face-alignment)](https://pypi.org/project/thermal-face-alignment/)
[![PyPI - License](https://img.shields.io/pypi/l/thermal-face-alignment)](LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/thermal-face-alignment)](https://pypistats.org/packages/thermal-face-alignment)


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
landmarker = ThermalLandmarks(device="cpu", n_landmarks=478)

landmarks, confidences = landmarker.process(image)
```

![landmarks](https://raw.githubusercontent.com/openscivision/thermal-face-alignment/7221fdc136ac84f2ce5a304b45b04bdd4bc7405b/img/landmarks.jpg)

## Training dataset

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

The thermal face bounding box detection in this repo uses the TFW landmarker model, please additionally cite:

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
