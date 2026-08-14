# 🫁 Chest X-Ray Diagnostic Platform

A multi-label pathology detection system for chest X-rays, built on the **NIH ChestX-ray14** dataset. The project trains and compares two CNN backbones — **MobileNetV2** and **EfficientNetB3** — and ships as an interactive **Streamlit** dashboard for exploring model performance and running inference on new scans, complete with Grad-CAM explainability and uncertainty-based review flags.

## 📌 Project Overview

This project builds a production-style deep learning pipeline for detecting 14 thoracic pathologies from chest X-rays, covering everything from raw image preprocessing to an interactive diagnostic dashboard:

- **Multi-label pathology classification** across 14 findings using transfer learning (MobileNetV2 & EfficientNetB3)
- **Per-class optimized decision thresholds**, tuned on the validation set to maximize F1 rather than a flat 0.5 cutoff
- **Weighted binary cross-entropy** to handle heavy class imbalance
- **MC Dropout uncertainty estimation** — stochastic forward passes flag low-confidence predictions for manual review
- **Grad-CAM explainability** — visual heatmaps showing which image regions drove each prediction
- **Model Performance Dashboard** — ROC/PR curves, macro & per-class metrics, side-by-side model comparison

## 🗂️ Project Structure

```
Div_Final/
│
├── notebooks/
│   ├── Div_Final_Xray.ipynb           # Main training & experimentation notebook
│   └── Div_Final_Xray - Copy.ipynb
│
├── Home.py                             # Streamlit entry point
├── utils.py                            # Preprocessing, model loading, inference, Grad-CAM
│
├── pages/
│   ├── 1_Model_Performance.py          # 📊 Metrics, ROC/PR curves, model comparison
│   └── 2_Xray_Analysis.py              # 🔍 Upload & analyze a scan, Grad-CAM overlay
│
├── experiments/
│   ├── chest_mobilenetv2/              # Trained model, thresholds, eval results
│   └── chest_efficientnetb3/
│
├── dataset/
│   ├── stats.json                      # Normalization stats
│   └── split_manifest_chest.csv        # Train/val/test split manifest
│
├── assets/eda/                         # EDA visualizations (class distribution, co-occurrence)
└── requirements.txt
```

## 📓 Notebook Walkthrough

The main notebook (`notebooks/Div_Final_Xray.ipynb`) is organized into the following sections:

**1. 🧹 Data Preparation**
- Loaded and cleaned the NIH ChestX-ray14 metadata
- Built the train/val/test split manifest, stratified across the 14 pathology labels
- Computed normalization statistics (mean/std) for preprocessing

**2. 📊 Exploratory Data Analysis**
- Positive case counts per pathology (severe class imbalance across findings)
- Pathology co-occurrence matrix (many scans show multiple simultaneous findings)

**3. 🤖 Modeling**
- Two-phase transfer learning: frozen-backbone warm-up, then fine-tuning, for both MobileNetV2 and EfficientNetB3
- Weighted binary cross-entropy loss to counteract class imbalance
- `EarlyStopping` (monitored on validation AUC, restoring best weights), `ReduceLROnPlateau`, and `ModelCheckpoint` callbacks during training
- Per-class threshold optimization on the validation set (maximizing F1 instead of using a flat 0.5 cutoff)

**4. 📈 Evaluation**
- Macro & per-class AUC, precision, recall, F1
- Saved `y_true.npy` / `y_probs.npy` per model to power interactive ROC/PR curves in the dashboard

**5. 🔍 Explainability**
- Grad-CAM implementation over each backbone's final conv layer
- MC Dropout for per-prediction uncertainty estimates

## 🖥️ Streamlit App Pages

**🏠 Home** (`Home.py`)
Dataset overview (size, pathology count, trained model status) and EDA visualizations.

**📊 Model Performance** (`pages/1_Model_Performance.py`)
Macro-averaged metrics per model, per-pathology F1 breakdown, interactive ROC and Precision-Recall curves by pathology.

**🔍 X-Ray Analysis** (`pages/2_Xray_Analysis.py`)
Upload a chest X-ray → get per-pathology probabilities against tuned thresholds, an uncertainty-based review flag, and a Grad-CAM heatmap overlay for the top finding. Supports single-model or side-by-side dual-model comparison, with a downloadable text report.

## 📊 Models

| Model | Backbone | Notes |
|---|---|---|
| MobileNetV2 | Lightweight | Faster inference, smaller footprint |
| EfficientNetB3 | Higher capacity | Stronger performance, larger model size |

Both models were trained on 224×224 letterboxed inputs with class-weighted loss and per-class optimized thresholds.

## ⚙️ Setup & Running Locally

Running locally means installing and launching the app on your own machine instead of using the deployed Streamlit Cloud link — useful for development, debugging, or running without an internet dependency.

**Prerequisites**
```bash
git clone https://github.com/EsmiraMammadli/Div_Final.git
cd Div_Final
pip install -r requirements.txt
```

**Run the app**
```bash
streamlit run Home.py
```
The app will open at `http://localhost:8501`. The `pages/` folder is automatically picked up by Streamlit as multi-page app routes.

**Data**
The app expects trained model files under `experiments/<model_name>/best_model.keras` and normalization stats at `dataset/stats.json`. If a model isn't present, its page shows a notice instead of failing.

## 🌐 Live Demo

Streamlit App: (https://divfinal-9zwhunhyjdhggvbm2f87dv.streamlit.app/)

## 🛠️ Tech Stack

- **Language:** Python 3
- **Deep Learning:** TensorFlow / Keras
- **Data:** pandas, numpy
- **Visualization:** Plotly
- **Metrics:** scikit-learn
- **Web App:** Streamlit
- **Image Processing:** Pillow, matplotlib
