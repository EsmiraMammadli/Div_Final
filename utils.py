"""Shared constants and helpers for the Chest X-Ray Diagnostic Platform dashboard.

Imported by Home.py and every page under pages/ so the preprocessing,
model-loading, and inference logic only lives in one place.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
PATHOLOGY_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
STATS_PATH = "dataset/stats.json"
UNCERTAINTY_THRESHOLD = 0.15

MODEL_INFO = {
    "MobileNetV2": {
        "model_path": "experiments/chest_mobilenetv2/best_model.keras",
        "thresholds_path": "experiments/chest_mobilenetv2/thresholds.json",
        "eval_summary_path": "experiments/chest_mobilenetv2/eval_summary.json",
        "eval_per_class_path": "experiments/chest_mobilenetv2/eval_per_class.csv",
    },
    "EfficientNetB3": {
        "model_path": "experiments/chest_efficientnetb3/best_model.keras",
        "thresholds_path": "experiments/chest_efficientnetb3/thresholds.json",
        "eval_summary_path": "experiments/chest_efficientnetb3/eval_summary.json",
        "eval_per_class_path": "experiments/chest_efficientnetb3/eval_per_class.csv",
    },
}


@dataclass
class ChestDetectionResult:
    top_finding: str
    findings: List[str]
    pathology_probs: List[float]
    thresholds_used: List[float]
    uncertainty: List[float]
    review_flag: bool


def step2_resize_letterbox(img: Image.Image, size=IMG_SIZE) -> Image.Image:
    img = img.copy()
    img.thumbnail(size, Image.LANCZOS)
    background = Image.new("RGB", size, (0, 0, 0))
    offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
    background.paste(img, offset)
    return background


@st.cache_resource
def load_stats() -> Tuple[np.ndarray, np.ndarray]:
    with open(STATS_PATH) as f:
        stats = json.load(f)
    return (
        np.array(stats["mean"], dtype=np.float32),
        np.array(stats["std"], dtype=np.float32),
    )


def preprocess_upload(uploaded_file) -> np.ndarray:
    mean, std = load_stats()
    img = Image.open(uploaded_file).convert("RGB")
    img = step2_resize_letterbox(img)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    return arr


@st.cache_resource
def load_models() -> Dict[str, tf.keras.Model]:
    """Loads whichever trained models are available on disk. Missing models
    are skipped (not an error) so the app still works if only one has been
    trained yet."""
    models = {}
    for name, info in MODEL_INFO.items():
        path = Path(info["model_path"])
        if path.exists():
            models[name] = tf.keras.models.load_model(path, compile=False)
    return models


@st.cache_data
def load_thresholds(model_name: str) -> Dict[str, float]:
    path = Path(MODEL_INFO[model_name]["thresholds_path"])
    if not path.exists():
        return {name: 0.5 for name in PATHOLOGY_NAMES}
    with open(path) as f:
        return json.load(f)


def available_models() -> List[str]:
    return [name for name, info in MODEL_INFO.items() if Path(info["model_path"]).exists()]


def predict_with_uncertainty(model, image_batch, n_iter: int = 20):
    """Several stochastic forward passes with dropout kept active
    (training=True) give a mean prediction plus a per-class uncertainty
    (std across passes) instead of a single point estimate."""
    preds = np.stack([model(image_batch, training=True).numpy() for _ in range(n_iter)])
    mean_pred = preds.mean(axis=0)
    uncertainty = preds.std(axis=0)
    return mean_pred, uncertainty, preds


def run_chest_pipeline_inference(image, model, pathology_names, thresholds: Optional[Dict[str, float]] = None):
    if thresholds is None:
        thresholds = {name: 0.5 for name in pathology_names}
    mean_pred, uncertainty, _ = predict_with_uncertainty(model, image[np.newaxis])
    pathology_probs = mean_pred[0]
    thresh_vec = np.array([thresholds[name] for name in pathology_names])

    positive_idx = np.where(pathology_probs >= thresh_vec)[0]
    findings = [pathology_names[i] for i in positive_idx] or ["No Finding"]
    top_finding = pathology_names[int(np.argmax(pathology_probs))]

    review = bool(uncertainty.max() > UNCERTAINTY_THRESHOLD)

    return ChestDetectionResult(
        top_finding=top_finding,
        findings=findings,
        pathology_probs=pathology_probs.tolist(),
        thresholds_used=thresh_vec.tolist(),
        uncertainty=uncertainty[0].tolist(),
        review_flag=review,
    )


def make_gradcam_heatmap(img_array, model, pred_index: Optional[int] = None):
    """Grad-CAM for the base(input) -> gap -> dropout -> output_dense head
    shared by both backbones in this project. Splits the model into a
    conv_model (input -> last conv feature map of the backbone) and a
    classifier_model (that feature map -> prediction), then differentiates
    the target class score with respect to the conv features."""
    base = model.get_layer(index=1)
    last_conv_layer = base.layers[-1]
    conv_model = tf.keras.Model(base.input, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = model.get_layer("gap")(classifier_input)
    x = model.get_layer("output_dense")(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)
        preds = classifier_model(conv_output)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index


def overlay_gradcam(orig_img_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import matplotlib.cm as cm

    orig_rgb = orig_img_pil.convert("RGB")
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(orig_rgb.size, Image.BILINEAR)
    heatmap_resized = np.asarray(heatmap_img, dtype=np.float32) / 255.0

    jet = cm.get_cmap("jet")
    colored = jet(heatmap_resized)[:, :, :3]
    colored_img = Image.fromarray(np.uint8(colored * 255)).convert("RGB")
    return Image.blend(orig_rgb, colored_img, alpha=alpha)


@st.cache_data
def load_eval_results(model_name: str):
    """Returns (summary_dict, per_class_dataframe) for a trained model, or
    (None, None) if that model hasn't been evaluated yet."""
    import pandas as pd

    info = MODEL_INFO[model_name]
    summary_path = Path(info["eval_summary_path"])
    per_class_path = Path(info["eval_per_class_path"])
    if not summary_path.exists() or not per_class_path.exists():
        return None, None
    with open(summary_path) as f:
        summary = json.load(f)
    per_class = pd.read_csv(per_class_path)
    return summary, per_class


@st.cache_data
def load_roc_arrays(model_name: str):
    """Returns (y_true, y_probs) saved during evaluation, or (None, None)."""
    info = MODEL_INFO[model_name]
    base_dir = Path(info["model_path"]).parent
    y_true_path = base_dir / "y_true.npy"
    y_probs_path = base_dir / "y_probs.npy"
    if not y_true_path.exists() or not y_probs_path.exists():
        return None, None
    return np.load(y_true_path), np.load(y_probs_path)
