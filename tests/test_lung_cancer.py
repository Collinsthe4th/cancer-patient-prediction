import numpy as np
import pandas as pd
import pytest

from lung_cancer.data import FEATURES, SYMPTOMS, clean
from lung_cancer.models import candidate_models, cross_validate, specificity
from lung_cancer.predict import build_row

RAW_COLUMNS = ["GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE",
               "FATIGUE ", "ALLERGY ", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
               "SWALLOWING DIFFICULTY", "CHEST PAIN", "LUNG_CANCER"]


def synthetic_raw(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.random(n) < 0.7
    data = {"GENDER": rng.choice(["M", "F"], n), "AGE": rng.integers(40, 80, n)}
    for col in RAW_COLUMNS[2:-1]:
        p = np.where(label, 0.7, 0.3)
        data[col] = np.where(rng.random(n) < p, 2, 1)
    data["LUNG_CANCER"] = np.where(label, "YES", "NO")
    return pd.DataFrame(data)[RAW_COLUMNS]


def test_clean_maps_codes_and_names():
    df = clean(synthetic_raw())
    assert list(df.columns) == FEATURES + ["lung_cancer"]
    assert set(np.unique(df[SYMPTOMS].to_numpy())) <= {0, 1}
    assert df["lung_cancer"].isin([0, 1]).all()


def test_clean_rejects_unexpected_codes():
    raw = synthetic_raw()
    raw.loc[0, "SMOKING"] = 3
    with pytest.raises(ValueError):
        clean(raw)


def test_specificity():
    assert specificity(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])) == pytest.approx(0.5)


def test_models_learn_a_clear_signal():
    df = clean(synthetic_raw(200))
    X, y = df[FEATURES], df["lung_cancer"]
    scores = cross_validate(candidate_models()["Logistic regression"], X, y, n_repeats=2)
    assert scores["roc_auc"][0] > 0.8
    baseline = cross_validate(candidate_models()["Majority class (baseline)"], X, y, n_repeats=2)
    assert baseline["roc_auc"][0] == pytest.approx(0.5)


def test_build_row_validates_input():
    row = build_row(60, "f", ["smoking", "coughing"])
    assert row.shape == (1, len(FEATURES))
    assert row.loc[0, "smoking"] == 1 and row.loc[0, "fatigue"] == 0 and row.loc[0, "gender_male"] == 0
    with pytest.raises(ValueError):
        build_row(60, "M", ["sneezing"])
