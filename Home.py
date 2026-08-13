import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from utils import PATHOLOGY_NAMES, available_models

st.set_page_config(page_title="Chest X-Ray Platform", layout="wide", page_icon="🫁")


def render_home():
    st.title("🫁 Chest X-Ray Diagnostic Platform")
    st.caption("Multi-label pathology detection on the NIH ChestX-ray14 dataset")

    try:
        manifest = pd.read_csv("dataset/split_manifest_chest.csv")
        dataset_size = f"{len(manifest):,}"
        train_size = f"{(manifest['split'] == 'train').sum():,}"
    except FileNotFoundError:
        dataset_size, train_size = "N/A", "N/A"

    trained_models = available_models()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset size", dataset_size)
    col2.metric("Training images", train_size)
    col3.metric("Pathologies tracked", len(PATHOLOGY_NAMES))
    col4.metric("Models trained", f"{len(trained_models)} / 2")

    if not trained_models:
        st.warning(
            "No trained models found yet under `experiments/`. Run the model "
            "training cells in the notebook first, then reload this page."
        )
    else:
        st.success(f"Available models: {', '.join(trained_models)}")

    st.divider()
    st.subheader("Exploratory Data Analysis")
    st.write(
        "The dataset is multi-label (a single scan can show several findings "
        "at once) and heavily imbalanced across pathologies -- both of which "
        "directly shaped the modeling choices on the next page."
    )

    eda_col1, eda_col2 = st.columns(2)
    class_dist_path = Path("assets/eda/class_distribution.png")
    cooc_path = Path("assets/eda/cooccurrence.png")

    with eda_col1:
        if class_dist_path.exists():
            st.image(str(class_dist_path), caption="Positive case counts per pathology", use_container_width=True)
        else:
            st.info("Run the EDA cell in the notebook to generate `assets/eda/class_distribution.png`.")

    with eda_col2:
        if cooc_path.exists():
            st.image(str(cooc_path), caption="Pathology co-occurrence matrix", use_container_width=True)
        else:
            st.info("Run the EDA cell in the notebook to generate `assets/eda/cooccurrence.png`.")

    st.divider()
    st.subheader("Navigate")
    nav1, nav2 = st.columns(2)
    with nav1:
        st.page_link("pages/1_Model_Performance.py", label="📊 Model Performance & Comparison", use_container_width=True)
    with nav2:
        st.page_link("pages/2_Xray_Analysis.py", label="🔍 Analyze an X-Ray", use_container_width=True)


render_home()
