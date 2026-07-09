"""Command-line risk score for one respondent.

Example:
    lung-cancer-predict --age 64 --gender M --yes smoking coughing wheezing chest_pain
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import FEATURES, SYMPTOMS

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "lung_cancer_model.joblib"


def build_row(age: int, gender: str, yes: list[str]) -> pd.DataFrame:
    """Turn answers into the one-row frame the model expects."""
    unknown = set(yes) - set(SYMPTOMS)
    if unknown:
        raise ValueError(f"Unknown symptom(s): {sorted(unknown)}. Choose from {SYMPTOMS}")
    if gender.upper() not in {"M", "F"}:
        raise ValueError("gender must be M or F")
    row = {"gender_male": int(gender.upper() == "M"), "age": int(age)}
    row.update({s: int(s in yes) for s in SYMPTOMS})
    return pd.DataFrame([row])[FEATURES]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lung cancer screening score (not a diagnosis).")
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--gender", choices=["M", "F", "m", "f"], required=True)
    parser.add_argument("--yes", nargs="*", default=[], metavar="SYMPTOM", help=f"any of: {', '.join(SYMPTOMS)}")
    args = parser.parse_args(argv)

    if not MODEL_PATH.exists():
        parser.exit(1, "No trained model found. Run scripts/train.py first.\n")
    model = joblib.load(MODEL_PATH)
    score = float(model.predict_proba(build_row(args.age, args.gender, args.yes))[0, 1])
    flag = "FLAG for follow-up" if score >= 0.5 else "below the follow-up threshold"
    print(f"Risk score: {score:.2f} (0 to 1) -> {flag}")
    print(
        "The score ranks respondents. It is not a calibrated probability, because the model\n"
        "was trained with balanced class weights on 276 survey responses. It is not a diagnosis."
    )


if __name__ == "__main__":
    main()
