# Color-Invariant Saree Design Recognition
### DeepLure AI Engineering Case Study (AIE-CASE)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Metric Learning](https://img.shields.io/badge/Loss-ArcFace-blue.svg)](https://arxiv.org/abs/1801.07698)
[![Backbone](https://img.shields.io/badge/Backbone-ConvNeXt--Tiny%20%7C%20EfficientNet-brightgreen.svg)](https://arxiv.org/abs/2201.03545)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 1. Project Overview & Problem Formulation

In textile e-commerce and manufacturing, a single saree surface motif (e.g., *Peacock Butta*, *Temple Borders*, *Ikat Diamonds*, *Paisley Kalka*) is woven or printed across dozens of distinct **colorways** (e.g., Crimson Red & Gold Zari, Royal Blue & Silver, Emerald Green & Orange).

**The Challenge**: Standard Deep Learning vision backbones trained with standard Cross-Entropy rely on color as a dominant low-frequency shortcut, matching garments by background hue rather than motif structure.

**The Solution**: We treat saree recognition as **"Face Recognition for Textiles"**:
1. **Metric Learning on Hypersphere**: Projects images into a 512-dimensional $L_2$-normalized latent space using **ArcFace (Additive Angular Margin Loss)**.
2. **Color Disentanglement**: Aggressive color perturbation (Grayscale dropout, extreme hue/saturation shifting, solarization) during training forces the neural network to anchor entirely on geometric textures, weave contours, and motif topologies.
3. **Dual-Task Formulation**:
   - **Identification (1:N)**: Retrieve and rank gallery references for an unseen query image (Rank-1, Rank-5, mAP).
   - **Verification (1:1)**: Determine whether two images share the same design via cosine similarity thresholding (ROC-AUC, EER).

---

## 🏛️ 2. System Architecture

```
                                  [ Input Saree RGB Image ]
                                              │
                                              ▼
                         [ Color-Invariance Transform Pipeline ]
                         • Grayscale Dropout (p = 0.35)
                         • Extreme Hue / Saturation Jitter (±0.4)
                         • Random Solarization & Lum Inversion (p = 0.15)
                         • Multi-scale Random Resized Crop (224x224)
                                              │
                                              ▼
                           [ ConvNeXt-Tiny Feature Extractor ]
                         • 7x7 Depthwise Convolutions (spatial repeat bias)
                         • Inverted Bottleneck & LayerNorm
                                              │
                                              ▼
                             [ Global Average Pooling 2D ]
                                              │
                                              ▼
                             [ Metric Projection Head ]
                         • BatchNorm1d -> Dropout(0.2) -> Linear(768 -> 512)
                         • L2 Hypersphere Normalization (||e||_2 = 1.0)
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        [ Training: ArcFace Head ]                       [ Inference: Cosine Matching ]
        • Learnable Class Centers W                      • 1:N Gallery Search: s = e_q · E_g^T
        • Angular Margin Penalty: cos(θ + m)             • 1:1 Pair Verification: s >= τ* (0.65)
        • Temperature Scaling: s = 30.0
```

---

## 📁 3. Project Structure

```
Deep Lure/
├── Dockerfile                # Production multi-stage container build
├── docker-compose.yml        # Orchestration for web app, training & benchmarks
├── entrypoint.sh             # Container startup script
├── approach_note.md          # 500-word defensible Approach Note (Deliverable 1)
├── config.py                 # Central configuration dataclass
├── dataset.py                # Dataset loaders, augmentations & mock data synthesis
├── models.py                 # ConvNeXt / EfficientNet backbones + metric projection head
├── losses.py                 # Pure PyTorch ArcFace & SupCon loss implementations
├── train.py                  # PyTorch training pipeline with AMP & Cosine schedule
├── evaluate.py               # Identification (Rank-1/5, mAP) & Verification (ROC-AUC, EER)
├── inference.py              # 1:N Gallery Search & 1:1 Pair Verification CLI and Python API
├── efficiency_benchmark.py   # Parameter count, FLOPs, and latency profiling (Deliverable 4)
├── run_demo.py               # Zero-setup 1-click end-to-end verification script
├── kaggle_notebook.ipynb     # Self-contained Kaggle / Colab ready notebook (Deliverable 2)
└── README.md                 # Project documentation
```

---

## 🐳 4. Containerized Execution (Docker & Docker Compose)

To guarantee 100% environment consistency across developer machines, training clusters, and production servers:

### Option A: Using Docker Compose (Recommended)
```bash
# Build and run the entire interactive dashboard and API
docker compose up --build

# Run training inside container
docker compose run --rm trainer

# Run efficiency benchmark inside container
docker compose run --rm benchmark
```

### Option B: Using Standalone Docker
```bash
# 1. Build image
docker build -t saree-recognition:latest .

# 2. Run web service
docker run -p 5000:5000 -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs saree-recognition:latest

# 3. Run interactive demo inside container
docker run --rm saree-recognition:latest python run_demo.py
```

---

## 🚀 5. Local Quickstart (Without Docker)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run 1-Click Zero-Setup Demo
```bash
python run_demo.py
```

### Step 3: Train on Custom / Kaggle Saree Dataset
```bash
python train.py --data_dir ./data/saree_corpus --backbone convnext_tiny --epochs 15 --batch_size 32
```

### Step 4: 1:1 Saree Pair Verification
```bash
python inference.py verify \
  --img1 ./data/demo_sarees/design_01_peacock_butta/colorway_01.jpg \
  --img2 ./data/demo_sarees/design_01_peacock_butta/colorway_02.jpg \
  --threshold 0.65
```

### Step 5: 1:N Gallery Search
```bash
python inference.py search \
  --query ./query_sample.jpg \
  --gallery_dir ./data/demo_sarees/ \
  --topk 5
```

---

## 📊 6. Efficiency & Model Complexity (Deliverable 4)

| Model Architecture | Parameters | Memory Footprint | GFLOPs (224x224) | CPU Latency | GPU Latency | Embedding Dim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ConvNeXt-Tiny (Ours)** | **28.6 M** | **110 MB** | **4.5** | **~24 ms** | **~3.2 ms** | **512-D (2 KB)** |
| **EfficientNet-B0 (Edge)** | **5.3 M** | **21 MB** | **0.4** | **~9 ms** | **~1.8 ms** | **512-D (2 KB)** |
| ResNet-50 (Baseline) | 25.6 M | 98 MB | 4.1 | ~21 ms | ~2.9 ms | 512-D (2 KB) |

---

## 🧪 7. Dual Evaluation Protocol (Deliverable 3)

| Metric Category | Metric | Purpose & Practical Justification |
| :--- | :--- | :--- |
| **Identification (1:N)** | **CMC Rank-1 & Rank-5** | Measures whether the exact design is returned as the #1 match or in top-5 candidate carousel. |
| **Identification (1:N)** | **Mean Average Precision (mAP)** | Evaluates overall ranking quality when multiple colorways of the same design exist in the gallery. |
| **Verification (1:1)** | **ROC-AUC** | Quantifies design discrimination capability across all possible decision thresholds ($0.0 \to 1.0$). |
| **Verification (1:1)** | **Equal Error Rate (EER)** | Identifies the balanced operating point where False Acceptance (FMR) equals False Rejection (FNMR). |
| **Verification (1:1)** | **Optimal Cosine Threshold ($\tau^* \approx 0.65$)** | Calibrated operational cut-off for automated matching. |

APPLINK: color-invariant-saree-design-recognition.ai.studio
