# Machine-learning methods and preliminary results

## Purpose

We use machine learning as a predictive comparison: how well can state-year AI innovation, occupational AI exposure, and year predict occupational employment and wage outcomes in 2023?

This is not a causal design.

## Models

We estimate three regressors separately for log employment and log annual mean wages:

1. **Linear Regression** — interpretable baseline that fits a linear relationship.
2. **Random Forest Regressor** — ensemble of decision trees that can capture nonlinear patterns and interactions.
3. **Gradient Boosting Regressor** — sequential tree ensemble that attempts to improve prediction by fitting residual errors.

## Predictors

- `AI_patents`: fractional AI patent count for the state-year.
- `unique_patents`: unique AI patent count for the state-year.
- `AIOE`: occupation-level AI exposure score.
- `year`: calendar year.

Outcomes:
- `ln_employment`
- `ln_mean_wage`

The models are baseline models. They do not yet include state or occupation categorical controls, demographics, industry composition, or other labor-market controls.

## Train/test design

- Training period: 2020–2022.
- Held-out test period: 2023.
- The analysis is restricted to occupations with a published AIOE score.
- Missing predictor values are median-imputed using a pipeline fitted on the training data.
- The outcome is evaluated only when observed.

## Preliminary results

See [model_comparison.csv](model_comparison.csv).

Random Forest is the strongest of these three baseline models for both outcomes in the held-out 2023 sample:
- Employment: R² = 0.757.
- Wages: R² = 0.886.

These scores indicate predictive fit in this particular split. They do not show that AI patents cause employment or wage changes. The model may be capturing broad state and occupation patterns through the predictors, and its performance should be re-tested with occupation/state identifiers, simpler benchmarks, and appropriate grouped or time-based validation.

## Metrics

- **R²**: proportion of test-outcome variation explained by predictions (may be negative for poor models).
- **RMSE**: root mean squared prediction error; larger errors receive extra penalty.
- **MAE**: average absolute prediction error.

For log outcomes, RMSE and MAE are measured in log points.

## Interpretation cautions

- Patent counts and unique patent counts are related predictors.
- AIOE measures potential task exposure, not likelihood of automation or job loss.
- The training/test split is temporal, but the same states and occupations occur in both periods.
- The current models are predictive, not causal.
- BLS OEWS estimates have limitations for direct year-to-year comparisons.
