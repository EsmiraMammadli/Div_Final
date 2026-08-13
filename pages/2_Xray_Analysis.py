import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils import (
    PATHOLOGY_NAMES,
    available_models,
    load_models,
    load_thresholds,
    make_gradcam_heatmap,
    overlay_gradcam,
    preprocess_upload,
    run_chest_pipeline_inference,
    step2_resize_letterbox,
)

st.set_page_config(page_title="X-Ray Analysis", layout="wide", page_icon="🔍")
st.title("🔍 X-Ray Analysis")

trained_models = available_models()
if not trained_models:
    st.warning("No trained models found under `experiments/`. Train a model in the notebook first.")
    st.stop()

mode = st.radio("Mode", ["Single model", "Compare both models"], horizontal=True)

if mode == "Single model":
    model_choice = st.selectbox("Model", trained_models)
    selected_models = [model_choice]
else:
    if len(trained_models) < 2:
        st.info("Only one trained model is available -- falling back to single-model mode.")
        selected_models = trained_models
    else:
        selected_models = trained_models

uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload a chest X-ray to begin.")
    st.stop()

models = load_models()
raw_image = Image.open(uploaded).convert("RGB")
display_image = step2_resize_letterbox(raw_image)
preprocessed = preprocess_upload(uploaded)

st.image(display_image, caption="Uploaded scan (letterboxed to 224x224)", width=280)

results = {}
with st.spinner("Analysing X-ray..."):
    for name in selected_models:
        thresholds = load_thresholds(name)
        result = run_chest_pipeline_inference(preprocessed, models[name], PATHOLOGY_NAMES, thresholds)
        results[name] = result

st.divider()

if len(selected_models) == 1:
    name = selected_models[0]
    result = results[name]
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Top finding", result.top_finding)
        st.write("**Positive findings:**", ", ".join(result.findings))
        if result.review_flag:
            st.warning("High prediction uncertainty -- flagged for manual review")
        else:
            st.success("Prediction uncertainty within normal range")

    with col2:
        prob_df = pd.DataFrame({
            "pathology": PATHOLOGY_NAMES,
            "probability": result.pathology_probs,
            "threshold": result.thresholds_used,
        }).sort_values("probability", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=prob_df["probability"], y=prob_df["pathology"], orientation="h", name="Probability",
        ))
        fig.add_trace(go.Scatter(
            x=prob_df["threshold"], y=prob_df["pathology"], mode="markers",
            marker=dict(symbol="line-ns", size=14, color="red"), name="Decision threshold",
        ))
        fig.update_layout(title=f"Per-pathology probability -- {name}", xaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Grad-CAM explainability")
    gradcam_target = st.selectbox("Highlight region driving the prediction for:", PATHOLOGY_NAMES,
                                   index=PATHOLOGY_NAMES.index(result.top_finding))
    target_idx = PATHOLOGY_NAMES.index(gradcam_target)
    heatmap, _ = make_gradcam_heatmap(preprocessed[np.newaxis], models[name], pred_index=target_idx)
    overlay = overlay_gradcam(display_image, heatmap)
    gc_col1, gc_col2 = st.columns(2)
    gc_col1.image(display_image, caption="Original", use_container_width=True)
    gc_col2.image(overlay, caption=f"Grad-CAM -- {gradcam_target}", use_container_width=True)

    report_lines = [f"Chest X-Ray Analysis Report ({name})", "=" * 40, f"Top finding: {result.top_finding}",
                     f"Positive findings: {', '.join(result.findings)}",
                     f"Review flag: {result.review_flag}", "", "Per-pathology probabilities:"]
    for p, prob, thr in zip(PATHOLOGY_NAMES, result.pathology_probs, result.thresholds_used):
        report_lines.append(f"  {p}: {prob:.3f} (threshold {thr:.2f})")
    st.download_button("Download report (.txt)", "\n".join(report_lines), file_name="xray_report.txt")

else:
    compare_df = pd.DataFrame({"pathology": PATHOLOGY_NAMES})
    for name, result in results.items():
        compare_df[name] = result.pathology_probs

    melted = compare_df.melt(id_vars="pathology", var_name="Model", value_name="Probability")
    fig = go.Figure()
    for name in selected_models:
        sub = melted[melted["Model"] == name]
        fig.add_trace(go.Bar(x=sub["pathology"], y=sub["Probability"], name=name))
    fig.update_layout(barmode="group", title="Model agreement across pathologies", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    summary_cols = st.columns(len(selected_models))
    for col, (name, result) in zip(summary_cols, results.items()):
        with col:
            st.markdown(f"**{name}**")
            st.metric("Top finding", result.top_finding)
            st.write(", ".join(result.findings))
            if result.review_flag:
                st.warning("Flagged for review")

    compare_df["max_disagreement"] = compare_df[selected_models].max(axis=1) - compare_df[selected_models].min(axis=1)
    biggest_disagreements = compare_df.sort_values("max_disagreement", ascending=False).head(5)
    st.subheader("Where the models disagree most")
    st.dataframe(biggest_disagreements[["pathology"] + selected_models + ["max_disagreement"]], use_container_width=True)
