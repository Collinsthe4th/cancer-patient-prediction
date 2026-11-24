"""Streamlit app: lung cancer survey screening score.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lung_cancer.data import SYMPTOMS, load  # noqa: E402
from lung_cancer.models import candidate_models, split_xy  # noqa: E402
from lung_cancer.predict import build_row  # noqa: E402

st.set_page_config(page_title="Lung Cancer Survey Screening", page_icon="🫁", layout="wide")


@st.cache_resource
def train():
    df = load()
    X, y = split_xy(df)
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    model = candidate_models()[metrics["best_model"]].fit(X, y)
    return model, df, metrics


model, df, metrics = train()

st.title("🫁 Lung Cancer Survey: Risk Screening")
st.warning("Learning project on a public survey of 276 people. This is not a medical tool and must not be used to diagnose anyone.")

left, right = st.columns([1, 1.3])
with left:
    st.subheader("Answer the questions")
    age = st.slider("Age", 20, 90, 60)
    gender = st.radio("Gender", ["M", "F"], horizontal=True)
    labels = {s: s.replace("_", " ").capitalize() for s in SYMPTOMS}
    yes = [s for s in SYMPTOMS if st.checkbox(labels[s], value=False, key=s)]

with right:
    score = float(model.predict_proba(build_row(age, gender, yes))[0, 1])
    st.subheader("Result")
    st.metric("Risk score (0 to 1)", f"{score:.2f}")
    st.progress(min(max(score, 0.0), 1.0))
    if score >= 0.5:
        st.error("Above the follow-up threshold of 0.5")
    else:
        st.success("Below the follow-up threshold of 0.5")
    st.caption(
        "The score ranks respondents. It is not a calibrated probability, because the model was trained "
        "with balanced class weights (only 38 of the 276 respondents were negative)."
    )

st.subheader("How the models compared (50 cross-validation folds)")
rows = []
for name, m in metrics["models"].items():
    rows.append({"Model": name, **{k.replace("_", " ").title(): f"{v['mean']:.3f} ± {v['std']:.3f}" for k, v in m.items()}})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Most useful answers")
    st.image(str(ROOT / "results" / "figures" / "feature_importance.png"))
with c2:
    st.subheader("Share answering yes, by label")
    rates = df.groupby("lung_cancer")[SYMPTOMS].mean().T
    rates.columns = ["Negative", "Positive"]
    st.bar_chart(rates, horizontal=True)
    st.caption("Smoking barely separates the groups in this survey (50% against 55%).")

st.markdown("[Source code and README](https://github.com/Collinsthe4th/cancer-patient-prediction)")
