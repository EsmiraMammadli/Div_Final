import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from utils import MODEL_INFO, PATHOLOGY_NAMES, load_eval_results, load_roc_arrays

st.set_page_config(page_title="Model Performance", layout="wide", page_icon="📊")
st.title("📊 Model Performance & Comparison")

results = {name: load_eval_results(name) for name in MODEL_INFO}
trained = {name: (summary, per_class) for name, (summary, per_class) in results.items() if summary is not None}

if not trained:
    st.warning(
        "No evaluation results found yet. Run the evaluation cells in the "
        "notebook (after training) to generate `eval_summary.json` and "
        "`eval_per_class.csv` for each model."
    )
    st.stop()

st.subheader("Macro-averaged metrics")
summary_rows = []
for name, (summary, _) in trained.items():
    row = {"Model": name}
    row.update(summary)
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)

metric_cols = st.columns(len(trained))
for col, (name, (summary, _)) in zip(metric_cols, trained.items()):
    with col:
        st.markdown(f"**{name}**")
        st.metric("Macro AUC", f"{summary['macro_auc']:.3f}")
        st.metric("Macro F1", f"{summary['macro_f1']:.3f}")
        st.metric("Macro Precision", f"{summary['macro_precision']:.3f}")
        st.metric("Macro Recall", f"{summary['macro_recall']:.3f}")

melted = summary_df.melt(
    id_vars="Model",
    value_vars=["macro_auc", "macro_precision", "macro_recall", "macro_f1"],
    var_name="Metric",
    value_name="Score",
)
melted["Metric"] = melted["Metric"].str.replace("macro_", "").str.upper()
fig_bar = px.bar(
    melted, x="Metric", y="Score", color="Model", barmode="group",
    title="Macro metrics by model", range_y=[0, 1],
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()
st.subheader("Per-pathology breakdown")
model_choice = st.selectbox("Model", list(trained.keys()))
_, per_class_df = trained[model_choice]
sort_by = st.selectbox("Sort by", ["auc", "f1", "precision", "recall", "support"], index=1)
st.dataframe(
    per_class_df.sort_values(sort_by, ascending=False).reset_index(drop=True),
    use_container_width=True,
)

fig_perclass = px.bar(
    per_class_df.sort_values("f1", ascending=True),
    x="f1", y="pathology", orientation="h",
    title=f"Per-pathology F1 -- {model_choice}",
    range_x=[0, 1],
)
st.plotly_chart(fig_perclass, use_container_width=True)

st.divider()
st.subheader("ROC & Precision-Recall curves")
curve_model = st.selectbox("Model ", list(trained.keys()), key="curve_model")
pathology_choice = st.selectbox("Pathology", PATHOLOGY_NAMES)
y_true, y_probs = load_roc_arrays(curve_model)

if y_true is None:
    st.info(
        f"No saved prediction arrays for {curve_model}. Save `y_true.npy` / "
        "`y_probs.npy` during evaluation to unlock these interactive curves."
    )
else:
    idx = PATHOLOGY_NAMES.index(pathology_choice)
    yt, yp = y_true[:, idx], y_probs[:, idx]

    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        if yt.sum() > 0:
            fpr, tpr, _ = roc_curve(yt, yp)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=pathology_choice))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Chance"))
            fig_roc.update_layout(title="ROC curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("No positive cases for this pathology in the saved test set.")
    with curve_col2:
        if yt.sum() > 0:
            precision, recall, _ = precision_recall_curve(yt, yp)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=pathology_choice))
            fig_pr.update_layout(title="Precision-Recall curve", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.info("No positive cases for this pathology in the saved test set.")

st.caption(
    "Per-class thresholds are tuned on the validation set to maximize F1 "
    "(see the notebook's Threshold Optimization section) rather than using "
    "a flat 0.5 cutoff, which is what makes precision/F1 usable on this "
    "imbalanced dataset."
)
