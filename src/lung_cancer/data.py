"""Loading and cleaning the lung cancer survey.

The raw file codes every symptom as 1 (no) or 2 (yes), uses "M"/"F" for gender
and "YES"/"NO" for the label, and has trailing spaces in two column names. This
module turns it into a tidy table of 0/1 flags with snake_case names.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "survey_lung_cancer.csv"

TARGET = "lung_cancer"

SYMPTOMS = [
    "smoking",
    "yellow_fingers",
    "anxiety",
    "peer_pressure",
    "chronic_disease",
    "fatigue",
    "allergy",
    "wheezing",
    "alcohol_consuming",
    "coughing",
    "shortness_of_breath",
    "swallowing_difficulty",
    "chest_pain",
]

FEATURES = ["gender_male", "age", *SYMPTOMS]


def _snake(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the survey with snake_case columns and 0/1 encodings.

    Raises:
        ValueError: if an expected column is missing or a code is not 1 or 2.
    """
    df = raw.rename(columns=_snake)
    missing = set(SYMPTOMS + ["gender", "age", TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    bad = ~df[SYMPTOMS].isin([1, 2])
    if bad.any().any():
        raise ValueError(f"Symptom codes must be 1 or 2, found {df[SYMPTOMS][bad].stack().unique()}")

    out = pd.DataFrame(
        {
            "gender_male": (df["gender"].str.strip().str.upper() == "M").astype(int),
            "age": df["age"].astype(int),
        }
    )
    for col in SYMPTOMS:
        out[col] = (df[col] == 2).astype(int)
    out[TARGET] = (df[TARGET].str.strip().str.upper() == "YES").astype(int)
    return out


def load(path: Path = RAW_PATH, drop_duplicates: bool = True) -> pd.DataFrame:
    """Load and clean the survey.

    The file holds 33 exact duplicate rows. They are dropped by default so that
    the same person cannot appear in both a training fold and a test fold, which
    would inflate the scores.
    """
    df = clean(pd.read_csv(path))
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df
