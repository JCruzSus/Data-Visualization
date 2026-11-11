"""
Baseline machine-learning comparison for the AI Labor Market project.

Run from the repository root:
    python train_models.py

Input:
    data/AI_BLS_AIOE_2020_2023_final.csv

Output:
    results/model_comparison.csv

Models:
    Linear Regression
    Random Forest Regressor
    Gradient Boosting Regressor

Training period: 2020–2022
Testing period: 2023
Predictors: AI_patents, unique_patents, AIOE, year
Outcomes: ln_employment, ln_mean_wage

These models are predictive baselines, not causal estimates.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline

DATA_FILE = Path("data/AI_BLS_AIOE_2020_2023_final.csv")
OUTPUT_FILE = Path("results/model_comparison.csv")
TEST_YEAR = 2023

PREDICTORS = ["AI_patents", "unique_patents", "AIOE", "year"]
OUTCOMES = {
    "ln_employment": "Log employment",
    "ln_mean_wage": "Log annual mean wage",
}


def load_analysis_data():
    """Load data and restrict the sample to occupations with AIOE scores."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_FILE}. Make sure the merged CSV is in data/."
        )

    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["AIOE"]).copy()

    for col in PREDICTORS + list(OUTCOMES):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_models():
    """Create the three baseline regression algorithms."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=3,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


def evaluate_one_outcome(df, outcome):
    """
    Train and evaluate all models for one outcome.

    Rows with a missing value for this outcome are excluded from that
    outcome's fit. Missing predictor values are median-imputed using
    a pipeline fit on training data only.
    """
    train = df[(df["year"] < TEST_YEAR) & df[outcome].notna()].copy()
    test = df[(df["year"] == TEST_YEAR) & df[outcome].notna()].copy()

    X_train = train[PREDICTORS]
    X_test = test[PREDICTORS]
    y_train = train[outcome]
    y_test = test[outcome]

    rows = []

    for model_name, model in build_models().items():
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            model,
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rows.append({
            "outcome": outcome,
            "model": model_name,
            "train_period": "2020-2022",
            "test_period": str(TEST_YEAR),
            "n_train": len(train),
            "n_test": len(test),
            "R2": r2_score(y_test, predictions),
            "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
            "MAE": mean_absolute_error(y_test, predictions),
        })

    return rows


def main():
    df = load_analysis_data()
    print(f"Matched analysis observations: {len(df):,}")
    print(f"Occupations: {df['OCC_CODE'].nunique():,}")
    print(f"States: {df['PRIM_STATE'].nunique():,}")
    print(f"Years: {df['year'].min()}–{df['year'].max()}")

    results = []
    for outcome in OUTCOMES:
        results.extend(evaluate_one_outcome(df, outcome))

    results_df = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
