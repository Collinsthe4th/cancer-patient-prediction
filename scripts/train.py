"""Compare the models with repeated cross-validation, then fit and save the best.

Usage:
    python scripts/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lung_cancer.data import load  # noqa: E402
from lung_cancer.models import candidate_models, cross_validate, split_xy  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def main() -> None:
    raw_rows = len(pd.read_csv(ROOT / "data" / "raw" / "survey_lung_cancer.csv"))
    df = load()
    X, y = split_xy(df)

    summary = {}
    for name, model in candidate_models().items():
        summary[name] = {k: {"mean": round(m, 3), "std": round(s, 3)} for k, (m, s) in cross_validate(model, X, y).items()}
        print(f"{name:28s} ROC-AUC {summary[name]['roc_auc']['mean']:.3f}  balanced acc {summary[name]['balanced_accuracy']['mean']:.3f}")

    ranked = [n for n in summary if n != "Majority class (baseline)"]
    best = max(ranked, key=lambda n: summary[n]["roc_auc"]["mean"])

    final = candidate_models()[best].fit(X, y)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(final, ROOT / "models" / "lung_cancer_model.joblib")

    # Permutation importance on the training data of the chosen model.
    imp = permutation_importance(final, X, y, scoring="roc_auc", n_repeats=30, random_state=42)
    importance = pd.Series(imp.importances_mean, index=X.columns).sort_values()

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    metrics = {
        "rows_raw": raw_rows,
        "rows_after_dedup": int(len(df)),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
        "cv": "5-fold stratified, repeated 10 times (50 fits per model)",
        "models": summary,
        "best_model": best,
        "top_features": importance.sort_values(ascending=False).head(5).round(4).to_dict(),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(7, 5))
    importance.plot.barh(ax=ax, color=np.where(importance > 0, "#2b6cb0", "#a0aec0"))
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title(f"Feature importance ({best})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=150)
    plt.close(fig)

    print(json.dumps({"best_model": best, "top_features": metrics["top_features"]}, indent=2))


if __name__ == "__main__":
    main()
