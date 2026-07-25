# YOLOv8 in PyTorch

[![CI](https://github.com/Samielakkad/AI-Computer-Vision-YOLOv8-Neusoft/actions/workflows/ci.yml/badge.svg)](https://github.com/Samielakkad/AI-Computer-Vision-YOLOv8-Neusoft/actions/workflows/ci.yml)

A readable YOLOv8 implementation study built during a 2024 computer-vision internship at Neusoft. The repository covers model construction, training, inference and VOC/COCO-style evaluation without importing an off-the-shelf YOLO model.

![Example detection output](figures/detection_sample.png)

The image above is a saved inference example. The checkpoint and dataset that produced it are not included, so it should be treated as a pipeline demonstration, not benchmark evidence.

## Evidence at a glance

| Item | Included | What can be checked |
| --- | --- | --- |
| Backbone, PANet neck and decoupled DFL head | Yes | Source and CPU forward-pass tests |
| Box decoding and anchor-grid utilities | Yes | Hand-calculated unit cases |
| Training loop and VOC data loader | Yes | Source review; requires an external dataset to run |
| VOC mAP and optional COCO mAP harness | Yes | Source review; requires data and a checkpoint |
| Example inference image | Yes | Visual pipeline example only |
| Training dataset | No | Not reproducible from this repository |
| Trained checkpoint and training logs | No | Not reproducible from this repository |
| Final mAP result | No | No performance result is claimed here |

## Architecture

![YOLOv8 architecture](figures/yolov8_architecture.png)

The implementation has three explicit stages:

- [`nets/backbone.py`](nets/backbone.py): CSP-style backbone with C2f and SPPF blocks
- [`nets/yolo.py`](nets/yolo.py): top-down and bottom-up PANet feature fusion, followed by separate box and class branches
- [`nets/yolo_training.py`](nets/yolo_training.py): task-aligned assignment with classification, CIoU and distribution focal losses

At 640 x 640, the tested 20-class `s` configuration produces three prediction scales with 8,400 anchor points. The box branch uses 16 DFL bins per side.

## Run the public checks

The CI path uses CPU PyTorch and does not need weights or a dataset:

```bash
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pytest
pytest -q
```

The ten tests construct the network, run a zero-input forward pass, verify output shapes and finite values, and check box/anchor geometry against hand-calculated cases.

## Train or evaluate

For the full toolchain:

```bash
pip install -r requirements.txt
```

Then provide a PASCAL VOC-style dataset under `VOCdevkit/`:

```bash
python voc_annotation.py
python train.py
python predict.py
python get_map.py
```

Configuration is currently edited in the Python entry files. `get_map.py` can generate predictions, extract VOC ground truth, calculate VOC mAP@0.5, or call the optional COCO evaluator. A valid checkpoint and test split are required before any metric is meaningful.

## Repository map

```text
nets/             model, backbone and training losses
utils/            data loading, decoding, NMS, fitting and VOC metrics
utils_coco/       optional COCO conversion and metrics
tests/            CPU architecture and geometry checks
train.py          freeze/unfreeze training entry point
predict.py        image, video, heatmap and ONNX modes
get_map.py        evaluation entry point
voc_annotation.py VOC split/annotation preparation
```

Source comments are mainly in French because that was the working language of the internship project.

## License

All rights reserved. The repository is public for review and reference; see [`LICENSE`](LICENSE).
