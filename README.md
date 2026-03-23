# 🩸 Uncertainty-Aware Fingerprint-Based Blood Group Classification Using Bayesian Deep Ensembles

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Research-lightgrey?style=flat-square"/>
  </p>

<p align="center">
  A research-grade deep learning framework that classifies ABO/Rh blood groups from grayscale fingerprint images while quantifying epistemic uncertainty using Monte Carlo Dropout and heterogeneous Bayesian Deep Ensembles.
</p>

<p align="center">
  <b>Ensemble Accuracy: 88.11%</b> &nbsp;|&nbsp; <b>Baseline Accuracy: 60.67%</b> &nbsp;|&nbsp; <b>Improvement: +27.44%</b>
</p>

---

> ⚠️ **Disclaimer:** This work is strictly methodological and non-clinical. No clinically validated fingerprint–blood group relationship is assumed or implied.

---

## 📌 Overview

Traditional blood group identification relies on invasive biochemical assays requiring samples, infrastructure, and trained personnel. This project explores a **non-invasive, fingerprint-based deep learning approach** with a key focus on **uncertainty quantification** — addressing the overconfidence problem common in standard deterministic deep learning models.

The framework combines:
- A **baseline CNN** with Monte Carlo Dropout for comparison
- A **heterogeneous Bayesian Deep Ensemble** (MediumCNN + DeepCNN + MobileNetV2)
- **Epistemic uncertainty estimation** via ensemble aggregation and Monte Carlo sampling

---

## 🏗️ System Pipeline

<p align="center">
  <img src="assets/fig1_pipeline.png" alt="System Pipeline" width="800"/>
  <br/>
  <em>Figure 1. Overview of the proposed Bayesian deep ensemble framework. Grayscale fingerprint images are preprocessed and passed through multiple independently trained convolutional models. Predictions are aggregated using Monte Carlo inference to obtain final class predictions and epistemic uncertainty.</em>
</p>

---

## 🏆 Key Contributions

- Heterogeneous deep ensemble with Monte Carlo inference for 8-class blood group prediction
- **+27.44% accuracy improvement** over single-model Monte Carlo Dropout baseline
- Well-separated uncertainty distributions: entropy ratio improved from **1.002 → 1.534**
- Selective prediction achieving **>95% accuracy** by rejecting lowest-confidence 20% of predictions
- Ablation studies on ensemble size and Monte Carlo inference depth
- Fully reproducible pipeline with fixed seeds, saved splits, and checkpoints

---

## 📊 Results

### Baseline vs. Ensemble Performance

| Metric | Baseline (MC=30) | Ensemble (MC=30) |
|---|---|---|
| **Accuracy (%)** | 60.67 | **88.11** |
| Predictive Entropy (Correct) | 2.022 | **0.723** |
| Predictive Entropy (Incorrect) | 2.026 | **1.109** |
| Entropy Ratio | 1.002 | **1.534** |
| Mean Epistemic Uncertainty | — | 0.070 |

> The ensemble achieves a **27.44 percentage point improvement** in classification accuracy. More critically, the entropy ratio rises from 1.002 to 1.534 — meaning the ensemble effectively distinguishes confident correct predictions from uncertain incorrect ones, a property the baseline entirely lacks.

### Uncertainty Discrimination

<p align="center">
  <img src="assets/fig3_entropy.png" alt="Entropy Histograms" width="750"/>
  <br/>
  <em>Figure 3. Predictive entropy histograms for baseline (left) and ensemble (right). Correct predictions (blue) show lower entropy; incorrect predictions (orange) show higher entropy. The ensemble demonstrates significantly stronger separation.</em>
</p>

### Epistemic Uncertainty Decomposition

<p align="center">
  <img src="assets/fig4_uncertainty.png" alt="Uncertainty Decomposition" width="480"/>
  <br/>
  <em>Figure 4. Predictive entropy vs. epistemic uncertainty for ensemble predictions. High epistemic uncertainty aligns with ambiguous ridge structures and borderline class assignments.</em>
</p>

### Calibration & Selective Prediction

<p align="center">
  <img src="assets/fig5_reliability.png" alt="Reliability Diagram" width="420"/>
  <br/>
  <em>Figure 5. Reliability diagram showing mean predicted confidence vs. empirical accuracy. The ensemble exhibits strong calibration, with high-confidence predictions closely following the ideal diagonal.</em>
</p>

Rejecting the lowest-confidence **20% of predictions** pushes accuracy to **>95%** on retained samples, demonstrating strong practical utility of the uncertainty estimates.

---

## 🧠 Model Architecture

### Bayesian Inference Mechanism

<p align="center">
  <img src="assets/fig2_bayesian.png" alt="Bayesian Aggregation" width="720"/>
  <br/>
  <em>Figure 2. Bayesian aggregation mechanism. An approximate posterior is obtained via ensemble diversity. Monte Carlo sampling across ensemble members estimates predictive distributions and uncertainty.</em>
</p>

### Ensemble Members

| Model | Role |
|---|---|
| **MediumCNN** | Moderately deep CNN for local ridge pattern capture |
| **DeepCNN** | deep CNN for hierarchical feature extraction |
| **MobileNetV2** | Pretrained, grayscale-adapted (single-channel input) |

Heterogeneous architectures encourage diverse representations and robust uncertainty estimation.

### Baseline
Lightweight CNN: `Conv → BatchNorm → ReLU → MaxPool` blocks with MC Dropout and compact classification head.

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-3 (CNNs), 1e-4 (MobileNetV2) |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Loss Function | Categorical Cross-Entropy |
| Epochs | 30 |
| Early Stopping | Validation loss-based |
| MC Passes | 30 per model |

---

## 🔬 Problem Statement

Given a grayscale fingerprint image `x ∈ ℝ^(224×224)`, predict blood group:

```
y ∈ { A+, A−, B+, B−, AB+, AB−, O+, O− }
```

**Monte Carlo Inference:**
```
p(y | x*, D) ≈ (1/K) Σ p(y | x*, θ_k)
```

**Epistemic Uncertainty:**
```
σ² = Var[ p(y | x*) ]
```

**Predictive Entropy:**
```
H = − Σ p(y=c | x*, D) · log p(y=c | x*, D)
```

---

## 📁 Dataset

- ~6,000 grayscale fingerprint images across 8 blood group classes
- Image resolution: 96×103px → resized to 224×224
- Class sizes: 565 (A⁺) to 1,009 (A⁻) — moderately imbalanced
- Stratified split: **~70% train / ~15% val / ~15% test** (~900 test samples)
- Same fixed split used across all experiments for fair comparison

---

## 🗂️ Project Structure

```
blood_group_detection/
│
├── Blood_group_detection.ipynb     # Main experiment notebook
│
├── models/
│   ├── baseline_cnn.py             # Baseline CNN with MC Dropout
│   ├── medium_cnn.py               # MediumCNN ensemble member
│   ├── deep_cnn.py                 # DeepCNN ensemble member
│   └── mobilenetv2_adapted.py      # Grayscale-adapted MobileNetV2
│
├── utils/
│   ├── data_utils.py               # Dataset split and transforms
│   ├── eda_utils.py                # Exploratory data analysis
│   ├── train_utils.py              # Training loop
│   ├── eval_utils.py               # Evaluation metrics
│   ├── inference_uncertainty.py    # MC Dropout inference & uncertainty
│   └── load_seed.py                # Reproducibility seed loading
│
├── train_ensemble.py               # Ensemble training script
│
├── assets/                         # Figures for README
│   ├── fig1_pipeline.png
│   ├── fig2_bayesian.png
│   ├── fig3_entropy.png
│   ├── fig4_uncertainty.png
│   └── fig5_reliability.png
│
├── extra_saves/
│   ├── random_state.pkl            # Saved RNG state
│   └── data_split.pkl              # Saved train/val/test split
│
├── checkpoints/                    # Saved model weights (not tracked)
├── data/                           # Dataset (not included)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/blood-group-detection-bayesian.git
cd blood-group-detection-bayesian
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
Organize your fingerprint images as:
```
data/
├── A+/   ├── A-/   ├── B+/   ├── B-/
├── AB+/  ├── AB-/  ├── O+/   └── O-/
```

### 4. Run on Google Colab
Mount your Drive, set `PROJECT_ROOT` to your Drive path, and open:
```
Blood_group_detection.ipynb
```
Run sequentially: `EDA → Split → Baseline Training → Evaluation → Uncertainty Analysis → Ensemble`

---

## 📦 Requirements

```
torch
torchvision
numpy
matplotlib
scikit-learn
Pillow
tqdm
```

Install: `pip install -r requirements.txt`

---


## ⚠️ Limitations

- No clinically validated fingerprint–blood group dataset exists
- Results are strictly methodological; not intended for medical use
- Limited dataset size; cross-population generalization not guaranteed
- Fingerprint–blood group biological correlation remains scientifically unestablished

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- MobileNetV2 pretrained weights via `torchvision`
- Inspired by Lakshminarayanan et al. (Deep Ensembles) and Gal & Ghahramani (MC Dropout)
- Dermatoglyphics references as cited in the paper
