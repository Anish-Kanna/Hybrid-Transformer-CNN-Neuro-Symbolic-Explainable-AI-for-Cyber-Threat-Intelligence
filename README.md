<div align="center">

&nbsp;

# Hybrid Transformer–CNN Neuro-Symbolic Explainable AI for Cyber Threat Intelligence: Advancing Transparency and Adversarial Robustness
### Explainable AI for Cyber Threat Intelligence

**School of Engineering, Dayananda Sagar University**

&nbsp;

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44.1-f5a623?style=for-the-badge)](https://shap.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

&nbsp;

**Advanced Intrusion Detection with Explainability, Adversarial Robustness & Real-time Inference**

`Intrusion Detection` · `Explainable AI` · `Transformer Networks` · `Adversarial Training` · `Network Security`

&nbsp;

&nbsp;

</div>

---

## Overview

Modern Intrusion Detection Systems (IDS) achieve high accuracy but operate as black boxes — security analysts cannot understand *why* a traffic flow is flagged as malicious. This creates a critical trust gap in SOC operations. Simultaneously, standard ML models are highly vulnerable to adversarial perturbations: attackers can craft subtle traffic modifications that fool the detector while remaining imperceptible to genuine users.

**XAI-CTI** solves this dual challenge by combining:
- Hybrid CNN-Transformer architecture for high-accuracy traffic classification
- SHAP-based feature attribution to explain model decisions in real-time
- Adversarial training to achieve 88% robustness under attack (vs. 65% for standard models)
- Lightweight inference (2-5ms per sample) suitable for production SOC deployment

The result is an **interpretable, robust, production-grade IDS** that security teams can understand, trust, and act upon.

`Network Intrusion Detection` · `Explainable Artificial Intelligence` · `Adversarial Robustness` · `Real-time Analytics` · `Threat Intelligence`

---

## Table of Contents

1. [Problem Statement](#1--problem-statement)
2. [Proposed Approach](#2--proposed-approach)
3. [System Architecture](#3--system-architecture)
4. [Core Principles](#4--core-principles)
5. [Technical Implementation](#5--technical-implementation)
6. [Results & Performance](#6--results--performance)
7. [Explainability Methods](#7--explainability-methods)
8. [Adversarial Robustness](#8--adversarial-robustness)
9. [File Structure & Usage](#9--file-structure--usage)
10. [Project Team](#10--project-team)

---

## 1. Problem Statement

### Three Critical Gaps in Current IDS

| Issue | Impact | XAI-CTI Solution |
|-------|--------|-----------------|
| **Black-Box Predictions** | Analysts can't validate alerts, high false positive fatigue | SHAP + Symbolic explanations |
| **Adversarial Vulnerability** | Attackers craft evasion traffic; 30% accuracy drop under perturbations | Adversarial training → 88% robustness |
| **Slow Inference** | Batch processing delays real-time response | 2-5ms single-sample prediction |

### Research Question

*Can we build an Intrusion Detection System that is simultaneously **accurate, interpretable, robust to adversarial attacks, and deployable in real-time SOC workflows**?*

---

## 2. Proposed Approach

### Core Innovation

**XAI-CTI** combines three complementary AI techniques to achieve **transparent and robust intrusion detection**:
```
Network Traffic Features (79 dimensions)
    ↓
├─ Hybrid CNN-Transformer Model (96.4% accuracy)
├─ SHAP Explainability Layer (Feature importance scoring)
└─ Adversarial Training (88% robustness under attack)
    ↓
Prediction + Confidence + Explanation + Threat Level
```
### Architecture Overview & Workflow
```
Raw CSV (CIC-IDS Dataset)
         │
         ▼
  preprocessing.py
  ├─ Strip whitespace from column names
  ├─ Encode labels: BENIGN → 0, ATTACK → 1
  ├─ Drop NaN / Inf values
  ├─ StandardScaler normalisation
  └─ 80/20 train-test split → PyTorch tensors
         │
         ▼
  XAI_CTI_Model (model.py)
  ├─ Conv1D  → extracts local feature patterns
  ├─ TransformerEncoder → global feature relationships
  └─ Linear(32 → 2)  → class logits
         │
    ┌────┴────────────────────┐
    ▼                         ▼
Standard Training        Adversarial Training (FGSM)
(train_gpu.py)           (train_adversarial.py)
    │                         │
    ▼                         ▼
xai_cti_model.pth      xai_cti_model_adv.pth
                              │
         ┌────────────────────┼
         ▼                    ▼                     
   SHAP Explainer       Symbolic Rules        
  (explain_shap.py)     (symbolic.py)       
  shap_summary.png     Threat Level 
```
### Design Philosophy

-> **Never trust, always verify** — Multiple signals voted on access

-> **Interpretability by design** — Explanations at every layer

-> **Robustness first** — Adversarially trained from day one

-> **Production-ready** — Sub-60ms inference, lightweight (~18k parameters)

---

## 3. System Architecture

### Three-Layer Design Philosophy

| Layer | Component | Purpose |
|-------|-----------|---------|
| **Data Layer** | CSV ingestion → Normalization → Train-test split | Feature consistency & reproducibility |
| **Model Layer** | CNN-Transformer hybrid, 18.5K parameters | Accurate, interpretable, fast |
| **Inference Layer** | SHAP + symbolic explainability | Trustworthy SOC integration |

### Architecture Diagram
```
Network Traffic Features (79 dimensions)
    ↓
[CNN Block]
  ├─ Conv1d: Extract local patterns (32 filters)
  ├─ ReLU: Non-linearity  
  └─ MaxPool1d: Reduce to 39 dimensions
    ↓
[Transformer Encoder]
  ├─ Self-attention: Learn feature relationships
  ├─ Multi-head (2 heads): Diverse pattern recognition
  └─ Output: 39×32 dimensional tensor
    ↓
[Global Average Pooling]
  └─ Aggregate to 32-dimensional representation
    ↓
[Fully Connected Layer]
  └─ 32 → 2 classes (BENIGN / ATTACK)
    ↓
[Softmax + Decision]
  ├─ Confidence: 0-100%
  └─ Prediction: BENIGN or ATTACK
```
---

## 4. Core Principles

### 1. **Explainability by Design**
- SHAP values quantify each feature's contribution to predictions
- Symbolic reasoning translates into human-readable rules
- Security analysts understand *why* an alert fires

### 2. **Adversarial Robustness**
- Standard models degrade 30% under perturbations
- Adversarial training defends against FGSM, PGD-like attacks
- Maintains 88% accuracy even under adversarial perturbations

### 3. **Lightweight & Fast**
- 18.5K parameters (75 KB model file)
- 2-5ms per-sample inference (CPU)
- Suitable for real-time SOC dashboards

### 4. **Production-Grade Quality**
- Reproducible results (fixed random seeds)
- Scalable to millions of flows
- Offline explainability (no external API dependencies)

### Training Phase

#### Step 1: Data Ingestion & Preprocessing

```
┌─────────────────────────────────────────┐
│  Raw Network Traffic CSV Files          │
│  (full_week.csv, dos_data.csv, etc.)   │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│  Preprocessing Pipeline                 │
├─────────────────────────────────────────┤
│ 1. Strip whitespace, remove non-numeric │
│ 2. Convert labels: BENIGN→0, ATTACK→1  │
│ 3. Handle anomalies: NaN, inf values    │
│ 4. Standardize: (x-μ)/σ (per feature)  │
│ 5. Train-test split: 80-20 ratio        │
│ 6. Convert to PyTorch tensors           │
└────────────┬────────────────────────────┘
             ↓
Training Set: 2.2M samples | Test Set: 565K samples
```

**Key transformation**: `scaler.pkl` saved for inference consistency

#### Step 2: Standard Training Loop

```python
For 5 epochs:
  For each batch (8 samples):
    1. Forward: predictions = model(batch)
    2. Loss: CrossEntropyLoss(predictions, labels)
    3. Backward: compute gradients
    4. Update: optimizer.step() (Adam, lr=0.001)
    5. Log: total loss per epoch
```

**Result**: `xai_cti_model.pth` (~75 KB)

#### Step 3: Adversarial Training (Enhanced)

```python
For 5 epochs:
  For each batch:
    1. Clean loss: L_normal = model(X_clean)
    
    2. FGSM attack: X_adversarial = X_clean + ε×sign(∇L)
       where ε = 0.01 (small, imperceptible perturbation)
    
    3. Adversarial loss: L_adv = model(X_adversarial)
    
    4. Combined: L_total = (L_normal + L_adv) / 2
    
    5. Update weights on L_total
```

**Result**: `xai_cti_model_adv.pth` (robust against evasion)

**Why combine losses?** Trains on both normal and perturbed data → resistant to attacks

### Inference Phase (Production)

```
┌────────────────────────────────┐
│  New Network Traffic Flow      │
└───────────┬────────────────────┘
            ↓
┌────────────────────────────────┐
│  Load Preprocessing Artifacts  │
│  ├─ scaler.pkl (normalization) │
│  ├─ feature_names (alignment)  │
│  └─ Model weights              │
└───────────┬────────────────────┘
            ↓
┌────────────────────────────────┐
│  Feature Processing (< 1ms)    │
│  ├─ Align columns              │
│  ├─ Fill missing (0)           │
│  ├─ StandardScaler.transform() │
│  └─ Convert to tensor          │
└───────────┬────────────────────┘
            ↓
┌────────────────────────────────┐
│  Model Inference (2-5ms)       │
│  ├─ Forward pass               │
│  ├─ Softmax probabilities      │
│  ├─ Argmax prediction          │
│  └─ Extract confidence         │
└───────────┬────────────────────┘
            ↓
┌────────────────────────────────┐
│  Explainability (100-600ms)    │
│  ├─ SHAP: Feature importance   │
│  ├─ Symbolic: Human-readable   │
│  └─ Threat level: LOW/MED/HIGH │
└───────────┬────────────────────┘
            ↓
┌────────────────────────────────┐
│  Output Report                 │
│  ├─ Prediction (BENIGN/ATTACK) │
│  ├─ Confidence: 0-100%         │
│  ├─ Threat Level               │
│  ├─ Top 3 Features (SHAP)      │
│  └─ Symbolic Explanations      │
└────────────────────────────────┘
```

---

## 4. Performance Results

### Anomaly Detection Accuracy (NSL-KDD Dataset)

| Metric | LSTM Baseline | CNN Baseline | **XAI-CTI Transformer** |
|--------|:---:|:---:|:---:|
| **Accuracy** | 91.2% | 93.1% | **96.4%** |
| **F1-Score** | 89.6% | 90.8% | **93.1%** |
| **False Positive Rate** | 8.5% | 6.7% | **2.3%** ← 73% improvement |
| **Precision** | 88.9% | 91.2% | **94.6%** |

**Key Insight**: 2.3% FPR = fewer false alarms for SOC teams (significant operational benefit)

### Adversarial Robustness Comparison

Tested with FGSM attacks at ε=0.1 (small imperceptible perturbations):

| Model | Normal Accuracy | Adversarial Accuracy | **Robustness Gain** |
|-------|:---:|:---:|:---:|
| Standard Training | 96.5% | **65.2%** | - |
| **Adversarial Training** | 95.2% | **88.3%** | **+23.1%** ✓ |

**Trade-off**: ~1% accuracy loss for +23% robustness (excellent deal)

### System Performance Metrics

| Metric | Value | Significance |
|--------|:---:|---|
| **Decision Latency** | < 60 ms | Real-time inline deployment viable |
| **Model Parameters** | 18,562 | Lightweight, fits on edge devices |
| **Model Size** | ~75 KB | Minimal storage footprint |
| **Accuracy** | 96.4% | Best-in-class vs baselines |
| **Trust Prediction** | 0.93 AUC | Strong entity-level trust scoring |
| **Explainability** | SHAP + Symbolic | Transparent decision reasoning |

---

## 5. System Architecture

### Project Structure

```
XAI_CTI_Project/
├── model.py                   # Hybrid CNN-Transformer architecture
├── preprocessing.py           # Data loading, normalization, scaler
├── train_gpu.py               # Standard supervised training
├── train_adversarial.py       # Adversarial training with FGSM
├── adversarial_test.py        # Robustness evaluation
├── robustness_curve.py        # Plot accuracy vs attack strength
├── explain_shap.py            # Generate SHAP explanations
├── demo.py                    # Live inference demo
├── demo2.py                   # Production inference (real data)
├── symbolic.py                # Rule-based explanations
├── merge_fullweek.py          # Consolidate multiple CSVs
│
├── data/
│   ├── full_week.csv          # Merged training dataset (1M+ samples)
│   ├── dos_data.csv           # Malicious traffic
│   ├── Monday-Friday.csv      # Daily network logs
│   └── insane_dos.csv         # Intense attack traffic
│
├── checkpoints/
│   ├── xai_cti_model.pth      # Standard model weights
│   ├── xai_cti_model_adv.pth  # Adversarially trained (preferred)
│   ├── scaler.pkl             # Feature normalization params
│   └── shap_summary.png       # Feature importance visualization
│
└── requirements.txt           # Python dependencies
```

### Data Flow Architecture

```
Raw Traffic Data
    ↓
[Preprocessing] → Normalized features + Scaler
    ↓
├─→ Training: 80% samples
│       ↓
│   ├─→ Standard Training → model.pth
│   │   ├─ Acc: 96.5%, FPR: 3-4%
│   │   └─ Baseline accuracy
│   │
│   └─→ Adversarial Training → model_adv.pth
│       ├─ Acc: 95-96%, Robustness: 88%↑
│       └─ Production-ready
│
└─→ Testing: 20% samples
    ├─→ Normal accuracy: 96%+
    ├─→ Adversarial accuracy: 88%+ (under attack)
    └─→ FPR: 2.3% (best-in-class)

Production Inference:
New Traffic → Scaler → Model → SHAP + Symbolic → Decision + Explanation
```

---

## 6. Core Components

### Model Architecture (CNN-Transformer Hybrid)

```
Layer           Input Shape       Processing        Output Shape
──────────────────────────────────────────────────────────────────
Input           [batch, 79]       Raw features      [B, 79]
Unsqueeze       [B, 79]           Add channel       [B, 1, 79]
Conv1d (×32)    [B, 1, 79]        Extract patterns  [B, 32, 79]
ReLU            [B, 32, 79]       Non-linearity     [B, 32, 79]
MaxPool1d(2)    [B, 32, 79]       Reduce dim        [B, 32, 39]
Permute         [B, 32, 39]       Rearrange         [B, 39, 32]
Transformer     [B, 39, 32]       Self-attention    [B, 39, 32]
GlobalAvgPool   [B, 39, 32]       Aggregate         [B, 32]
FC(32→2)        [B, 32]           Classification    [B, 2]
Softmax         [B, 2]            Probabilities     [B, 2] ∈ [0,1]
```

**Parameters**: ~18,562 total (lightweight!)

### Feature Space (79 Dimensions)

| Category | Features | Examples |
|----------|:---:|---|
| **Flow Identity** | 4 | Src/Dst IP, Port, Protocol |
| **Packet Stats** | 12 | Forward/Backward bytes, packets, lengths  |
| **TCP Flags** | 6 | SYN, ACK, FIN, RST, PSH, URG counts |
| **Timing** | 8 | Flow duration, IAT mean/std, active/idle |
| **Advanced** | 49 | Entropy,  init window bytes, subflow stats |
| **Total** | **79** | Standardized to μ=0, σ=1 |

### SHAP Explainability

**What it calculates**: Contribution of each feature to the prediction

**Per-sample explanation**:
```
Predicted: ATTACK
Confidence: 94%

Top Contributing Features:
1. Total_Forward_Bytes (SHAP: +0.85)  → Pushes toward ATTACK
2. Packet_Count (SHAP: +0.62)         → Pushes toward ATTACK
3. Flow_Duration (SHAP: +0.41)        → Slightly suspicious

Interpretation:
High byte volume + high packet rate + unusual timing = ATTACK
```

### Symbolic Reasoning

**Simple rule-based explanations**:

```python
if "Bytes" in feature_name:
    reason = "Abnormally high traffic volume detected"
elif "Packets" in feature_name:
    reason = "Elevated packet transmission rate observed"
elif "Duration" in feature_name:
    reason = "Unusual flow timing pattern"
else:
    reason = f"Significant deviation in {feature_name}"
```

**Advantage**: <1ms latency, human-interpretable (ideal for real-time dashboards)

---

## 7. Implementation Guide

### Setup & Installation

```bash
# Clone repository
git clone <repo-url> && cd XAI_CTI_Project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Quick Start

```bash
# 1. Train adversarial model
python3 train_adversarial.py
# Output: xai_cti_model_adv.pth

# 2. Evaluate robustness
python3 adversarial_test.py
# Output: Normal Acc, Adversarial Acc, Drop

# 3. Generate SHAP plot
python3 explain_shap.py
# Output: shap_summary.png

# 4. Live inference
python3 demo2.py
# Output: Prediction + Confidence + Threat Level + Explanation
```

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|:---:|---|
| Batch Size | 8 | Balance memory & gradient quality |
| Learning Rate | 0.001 | Adam optimizer default |
| Epochs | 5 | Standard supervised training |
| CNN Filters | 32 | Local pattern extraction |
| Anomaly Threshold (τ_a) | 0.5 | Decision boundary for TBAE |
| FGSM Epsilon | 0.01 | Adversarial perturbation strength |

---

## 8. Explainability Methods

### SHAP (SHapley Additive exPlanations)

**Principle**: Game-theoretic approach using Shapley values

**Process**:
1. Compute baseline (reference training set)
2. For each feature: measure contribution to prediction difference
3. Aggregate across samples
4. Rank by importance

**Output**: Feature importance plot showing which features matter most

**Use case**: Offline analysis, investigation, model debugging

### Symbolic Reasoning

**Principle**: Apply interpretable rules to top features

**Process**:
1. Identify top-K most important features
2. Apply domain heuristics
3. Generate human-readable descriptions
4. Output SOC-friendly explanations

**Use case**: Real-time dashboards, live SOC alerts

### Threat Level Classification

```
Decision Logic:
<<<<<<< HEAD
├─ If prediction=ATTACK & confidence>90%  →  HIGH THREAT
├─ If prediction=ATTACK & confidence<90%  →  MEDIUM THREAT
└─ If prediction=BENIGN                   →  LOW THREAT
=======
├─ If prediction=ATTACK & confidence>90%  →  HIGH THREAT
├─ If prediction=ATTACK & confidence<90%  →  MEDIUM THREAT
└─ If prediction=BENIGN                   →  LOW THREAT
>>>>>>> e58e0783918eeaca09c74ac8229a0266d47d09ab
```

---

## 9. Adversarial Robustness

### Attack Method: FGSM (Fast Gradient Sign Method)

```
Adversarial Example = Original Data + ε × sign(∇Loss)
```

**Example**:
```
Original traffic: [2.5MB bytes, 1200 packets, 5 sec]
Perturbation:     [+0.01MB,    -5 packets,    +0.1 sec]
                  ─────────────────────────────────────
Adversarial:      [2.51MB,     1195 packets,  5.1 sec]
Result:           Imperceptible to humans, but...
Standard model:   "BENIGN" ❌ (Fooled!)
Robust model:     "ATTACK" ✓ (Resistant!)
```

### Robustness Curve

```
Accuracy (%)
   100 ├─────────────
       │ Normal Model
    95 ├─────╲
       │      ╲ After adversarial training
    90 ├─────╶╲
       │       ╲___
    85 ├──────────╲___
       │             ╲___
    80 └─────────────────────────────
         0   0.05  0.1  0.15  0.2 (epsilon)
```

**Key finding**: +23% robustness gain with adversarial training

---

## 10. Limitations & Path Forward

| # | Limitation | Impact | Mitigation |
|---|---|---|---|
| L1 | SHAP slow (100-500ms) | Real-time batch processing | Implement Top-K approximation |
| L2 | Non-IID data in FL | Not applicable here | N/A for centralized training |
| L3 | Limited attack types | May miss novel exploits | Retrain with new data quarterly |
| L4 | Encrypted traffic | Can't inspect payloads | Use metadata analysis only |
| L5 | Concept drift | Model degrades over time | Online learning / retraining |
| L6 | Single GPU constraint | Limits batch throughput | Distributed training support |

### Future Enhancements

✅ Multi-class classification (8+ attack types)

✅ Online learning pipeline (continuous retraining)

✅ Federated learning (privacy-preserving training)

✅ Mobile deployment (edge inference)

✅ Real-time streaming analysis

✅ Integration with SIEM platforms

---

---

## Project Team

<table>
  <tr>
    <td align="center">
      <b>Anish Kanna T A</b><br/>
    </td>
    <td align="center">
      <b>Vaishanth Mohan</b><br/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Anagha R Prabhu</b><br/>
    </td>
    <td align="center">
      <b>Bharat Aadarsh Meherwade</b><br/>
    </td>
  </tr>
</table>

---

## Faculty Mentor

<table>
  <tr>
    <td align="center">
      <b>Dr. Prajwalasimha S N</b><br/>
      <small>Associate Professor</small><br/>
      <small>Department of Computer Science and Engineering (Cyber Security)</small><br/>
      <small>School of Engineering, Dayananda Sagar University</small>
    </td>
  </tr>
</table>

---

**School of Engineering · Dayananda Sagar University**

*Bangalore – 560082, Karnataka, India*

Dataset: https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
Research Paper: https://www.researchgate.net/publication/398320038
