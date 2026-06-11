# DeepFaintEdges

DeepFaintEdges is a deep learning framework for detecting faint edges in noisy images using a convolutional neural network (CNN). The method is based on the ICPR 2020 paper:

**Multi-scale Processing of Noisy Images using Edge Preservation Losses**
*Nati Ofir and Yosi Keller*
International Conference on Pattern Recognition (ICPR 2020)

Paper:
https://ieeexplore.ieee.org/document/9413325

The repository contains a pretrained U-Net model, training and testing code, and examples demonstrating edge detection on both synthetic binary images and real-world natural images.

---

# Motivation

Detecting edges in images with low signal-to-noise ratio (SNR) remains a challenging computer vision problem.

Traditional edge detectors such as:

* Sobel
* Prewitt
* Roberts
* Canny

rely primarily on local gradients and often fail when:

* Image noise is high
* Edge contrast is low
* Boundaries are fragmented
* Structures are partially occluded

DeepFaintEdges approaches this problem using a U-Net-based architecture trained to recover clean edge maps directly from noisy observations.

The model learns spatial and contextual information across multiple scales, allowing it to identify edge structures that are difficult to recover using conventional local filtering techniques.

---

# Features

* Deep CNN-based faint edge detector
* U-Net architecture
* Pretrained model included (`keras_moving_trained_model0.h5`)
* Support for noisy binary images
* Support for natural images
* Keras implementation
* PyTorch implementation
* Jupyter notebook demo
* Training and evaluation scripts
* Multi-scale edge-preserving learning framework

---

# Example Results

The network receives a noisy image as input and predicts a clean edge probability map.

Applications include:

* Faint contour extraction
* Biomedical imaging
* Remote sensing
* Industrial inspection
* Low-light vision systems
* Image preprocessing for segmentation

Example pipeline:

```text
Input Image
      ↓
Additive Noise
      ↓
U-Net
      ↓
Predicted Edge Map
      ↓
Thresholding / Post-processing
```

---

# Repository Structure

```text
DeepFaintEdges/
│
├── DeepEdges/
│   ├── train_unet.py
│   ├── test_unet.py
│   ├── test_real_image_unet.py
│   ├── requirements.txt
│   └── ...
│
├── FinalImages/
│
├── keras_moving_trained_model0.h5
│
├── demo.ipynb
│
└── README.md
```

Main components:

| File                             | Description                                       |
| -------------------------------- | ------------------------------------------------- |
| `demo.ipynb`                     | Interactive notebook demonstrating edge detection |
| `keras_moving_trained_model0.h5` | Pretrained U-Net weights                          |
| `train_unet.py`                  | Train a U-Net model                               |
| `test_unet.py`                   | Evaluate on binary datasets                       |
| `test_real_image_unet.py`        | Run inference on natural images                   |
| `requirements.txt`               | Python dependencies                               |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/NatiOfir/DeepFaintEdges.git
cd DeepFaintEdges
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate:

Linux / Mac:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r DeepEdges/requirements.txt
```

---

# Quick Start

## Option 1: Jupyter Notebook Demo

Launch:

```bash
jupyter notebook demo.ipynb
```

The notebook loads the pretrained model and demonstrates edge detection on example images.

---

## Option 2: Test on Binary Images

```bash
python DeepEdges/test_unet.py
```

This script evaluates the network on synthetic binary-image datasets.

---

## Option 3: Test on Real Images

```bash
python DeepEdges/test_real_image_unet.py
```

This script applies the trained network to natural images.

---

# Training

To train a new model:

```bash
python DeepEdges/train_unet.py
```

Training parameters such as:

* learning rate
* batch size
* number of epochs
* augmentation
* noise level

can be modified within the training script.

The network learns to reconstruct edge maps from noisy observations while preserving edge structures across multiple image scales.

---

# Network Architecture

The repository uses a U-Net style encoder-decoder architecture.

```text
Input Image
      ↓
Encoder
      ↓
Bottleneck
      ↓
Decoder
      ↓
Edge Probability Map
```

Characteristics:

* Skip connections
* Multi-scale feature aggregation
* Fully convolutional
* Pixel-wise prediction
* End-to-end training

The architecture is designed to preserve fine edge details while suppressing image noise.

---

# Dataset

The training process uses binary image datasets commonly used in image-processing research. The original repository references the dataset:

http://www.imageprocessingplace.com/root_files_V3/image_databases.htm

The model is trained to detect edges under varying noise levels and image degradations.

---

# Pretrained Model

The repository includes:

```text
keras_moving_trained_model0.h5
```

This checkpoint contains pretrained U-Net weights and can be used directly for inference without retraining.

Example:

```python
from tensorflow.keras.models import load_model

model = load_model("keras_moving_trained_model0.h5")
```

---

# Comparison with Classical Edge Detectors

| Method         | Noise Robustness | Curved Edges | Learned Features |
| -------------- | ---------------- | ------------ | ---------------- |
| Sobel          | Low              | Limited      | No               |
| Prewitt        | Low              | Limited      | No               |
| Canny          | Medium           | Good         | No               |
| DeepFaintEdges | High             | Excellent    | Yes              |

DeepFaintEdges leverages learned representations rather than relying solely on local gradient information.

---

# Related Projects

Classical faint edge detection repositories by the same author:

### FastEdgesCpp

Fast detection of faint curved edges in C++.

https://github.com/NatiOfir/FastEdgesCpp

### FaintCurvedEdgeDetection

MATLAB implementation of curved faint edge detection.

https://github.com/NatiOfir/FaintCurvedEdgeDetection

### TrianglesEdgeDetection

Triangle-based faint edge detection framework.

https://github.com/NatiOfir/TrianglesEdgeDetection

### FaintStraightEdgeDetection

Fast detector for faint straight edges.

https://github.com/NatiOfir/FaintStraightEdgeDetection

---

# Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{ofir2020multiscale,
  title={Multi-scale Processing of Noisy Images using Edge Preservation Losses},
  author={Ofir, Nati and Keller, Yosi},
  booktitle={International Conference on Pattern Recognition (ICPR)},
  year={2020}
}
```

---

# License

No license file is currently included in this repository.

If you intend to use the code in commercial products or redistribute modified versions, please contact the repository owner or add an explicit open-source license.

---

# Acknowledgments

This work was developed as part of research on deep learning approaches for faint edge detection and robust image understanding in noisy environments.

Special thanks to the contributors of:

* TensorFlow / Keras
* PyTorch
* U-Net
* The image-processing research community

Good luck and happy edge detecting!
