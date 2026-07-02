"""Candidate models and the cross-validation routine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES


def candidate_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Return the models compared in this project.

    Class weights are balanced because only about 1 in 7 respondents is a
    negative case. Without that, a model can score 86% accuracy by always
    saying "yes".
    """
    scale_age = ColumnTransformer([("age", StandardScaler(), ["age"])], remainder="passthrough")
    return {
        "Majority class (baseline)": Pipeline([("clf", DummyClassifier(strategy="prior"))]),
        "Logistic regression": Pipeline(
            [
                ("scale", scale_age),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)),
            ]
        ),
        "Random forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "Gradient boosting": Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        learning_rate=0.05,
                        max_iter=200,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                )
            ]
        ),
    }


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of true negatives that the model correctly calls negative."""
    return float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))


def cross_validate(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Score a model with repeated stratified k-fold cross-validation.

    With only 38 negative cases, a single split is very noisy, so the folds are
    repeated 10 times and the mean and standard deviation are reported.

    Returns:
        Metric name mapped to (mean, standard deviation).
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scores: dict[str, list[float]] = {k: [] for k in ("roc_auc", "pr_auc", "recall", "specificity", "balanced_accuracy")}
    for train_idx, test_idx in cv.split(X, y):
        m = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = m.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        yt = y.iloc[test_idx]
        scores["roc_auc"].append(roc_auc_score(yt, proba))
        scores["pr_auc"].append(average_precision_score(yt, proba))
        scores["recall"].append(recall_score(yt, pred, zero_division=0))
        scores["specificity"].append(specificity(yt.to_numpy(), pred))
        scores["balanced_accuracy"].append(balanced_accuracy_score(yt, pred))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()}


def split_xy(df: pd.DataFrame, target: str = "lung_cancer") -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[target]
