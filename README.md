# EPA: Boosting Event-based Video Frame Interpolation with Perceptually Aligned Learning

<!-- TODO: Add badges for a professional look -->
[![Conference](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://nips.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="assets/pipline.pdf" width="80%">
  <br>
  <em><b>Fig 1</b>: Visual comparison of EPA against state-of-the-art methods on challenging high-speed motion and blurry inputs. EPA generates sharper intermediate frames with fewer artifacts.</em>
  <!-- TODO: Create a high-quality GIF or comparison image, place it in a folder (e.g., assets/), and replace the path above. -->
</p>

This repository contains the official PyTorch implementation for the paper **"EPA: Boosting Event-based Video Frame Interpolation with Perceptually Aligned Learning"**, accepted at NeurIPS 2025.

In this work, we introduce **EPA**, a novel framework designed to address a critical challenge in Event-based Video Frame Interpolation (E-VFI): performance degradation in extreme scenarios with high-speed motion and severe keyframe degradation (e.g., blur, noise). The core innovation of EPA is a paradigm shift from conventional pixel-level supervision to learning within a degradation-insensitive, semantic-perceptual feature space.

---

## 🚀 Key Contributions

*   **Perceptually Aligned Learning Paradigm**: By operating in a semantic-perceptual feature space, EPA is significantly more robust to real-world degradations like motion blur and sensor noise, leading to superior generalization.
*   **Bidirectional Event-Guided Alignment (BEGA) Module**: We propose a novel and efficient module that leverages the high temporal resolution of event streams to accurately align and fuse semantic features from keyframes.
*   **State-of-the-Art Performance**: EPA achieves leading performance on multiple synthetic and real-world benchmarks (e.g., GOPRO, Vimeo90k, HS-ERGB), especially in terms of perceptual quality metrics like LPIPS and DISTS.

---

## 🛠️ Installation

### 1. Prerequisites
*   Linux
*   Python 3.8+
*   PyTorch 1.10+
*   CUDA 11.1+

### 2. Setup Environment
We highly recommend using `conda` to create an isolated environment.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/EPA.git
cd EPA

# 2. Create and activate the conda environment
conda create -n epa python=3.8
conda activate epa

# 3. Install dependencies
pip install -r requirements.txt
```
<!-- TODO: Ensure you have a `requirements.txt` file in your project listing all necessary Python packages. -->

---

## 💾 Datasets and Pre-trained Models

### 1. Datasets
Please download the required datasets and organize them as follows.
<!-- TODO: If you have preprocessing scripts, add instructions here. -->
```
EPA
├── data
│   ├── gopro
│   │   ├── train
│   │   └── test
│   ├── vimeo90k
│   │   ├── sequences
│   │   └── sep_trainlist.txt
│   └── hs_ergb
│       └── ...
...
```

### 2. Pre-trained Models
Download our pre-trained models from the link below and place them in the `checkpoints` directory.

[**Download All Pre-trained Models (Google Drive / Hugging Face)**](https://your-download-link.com)
<!-- TODO: Upload your model weights and replace the link above. -->

After downloading, extract the files and place them under `./checkpoints/`.

---

## ⚡️ Quick Start: Inference and Evaluation

### Inference
To run inference on your own sequence of frames, use the following command:
```bash
python test.py --config configs/epa_gopro.yaml \
               --checkpoint checkpoints/epa_gopro.pth \
               --input_dir path/to/your/input_frames \
               --output_dir path/to/save/results
```
<!-- TODO: Verify that your config files and script arguments match this example. -->

### Evaluation
To reproduce the quantitative results from our paper on a standard benchmark (e.g., the GOPRO test set), run:```bash
python evaluate.py --config configs/epa_gopro.yaml \
                   --checkpoint checkpoints/epa_gopro.pth \
                   --dataset_path data/gopro/test
```

---

## 📈 Training

To train the model from scratch, follow our two-stage training procedure:

```bash
# Stage 1: Train the reconstruction generator
python train.py --config configs/epa_stage1.yaml

# Stage 2: Freeze the generator and train the BEGA alignment module
python train.py --config configs/epa_stage2.yaml --weights checkpoints/epa_stage1_best.pth
```
<!-- TODO: Ensure your training script and configs are set up to support this two-stage process. -->

---

## 📊 Results

### Quantitative Comparison
Our method demonstrates superior performance across multiple benchmarks.

**Table 1: Performance on GOPRO and Vimeo90k datasets.**
| Method      | Dataset    | LPIPS ↓ | DISTS ↓ | PSNR ↑  |
|-----------|-----------|---------|---------|---------|
| CBMNet    | GOPRO     | 0.050   | 0.046   | ...     |
| TLXNet    | GOPRO     | 0.021   | 0.019   | ...     |
| **EPA (Ours)** | **GOPRO** | **0.006**   | **0.008**   | **...** |
| ...       | ...       | ...     | ...     | ...     |

<!-- TODO: Copy the main quantitative results table from your paper here. -->

### Qualitative Comparison

[//]: # (<p align="center">)

[//]: # (  <img src="assets/qualitative_comparison.png" width="90%">)

[//]: # (  <br>)

[//]: # (  <em><b>Fig 2</b>: Visual comparison on the real-world HS-ERGB dataset.</em>)

[//]: # (  <!-- TODO: Replace this with a qualitative result image from your paper. -->)

[//]: # (</p>)

---

## 📜 Citation
If you find our work useful for your research, please consider citing our paper:
```bibtex
@inproceedings{liu2025epa,
  title={{EPA}: Boosting Event-based Video Frame Interpolation with Perceptually Aligned Learning},
  author={Liu, Yuhan and Fu, Linghui and Yang, Zhen and Chen, Hao and Li, Youfu and Deng, Yongjian},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## 🙏 Acknowledgements
Our implementation builds upon several excellent open-source projects. We are grateful for their contributions to the community.
*   [repository1_name](https://github.com/user/repo1)
*   [repository2_name](https://github.com/user/repo2)
<!-- TODO: List any repositories you built upon. -->

## 📄 License
This project is licensed under the [MIT License](LICENSE).
<!-- TODO: Add a file named LICENSE to your repository and copy the MIT License text into it. -->
