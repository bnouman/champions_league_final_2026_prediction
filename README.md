# Champions League Final Prediction

This repository contains two supervised models for predicting the Champions League final outcome from historical match data:

- `predict_final_form_rf.py`: Random Forest baseline
- `predict_final_hierarchical_bayes.py`: hierarchical Bayesian model

## Data

The project uses `data/dataset_cleaned.csv`, which contains Champions League matches from the 2021-22 to 2025-26 seasons.

## Modeling approach

The current analysis is focused on pre-match information only.

### Baseline model

The Random Forest baseline uses away-form features built from:

- the current season
- the last four completed seasons

### Hierarchical Bayesian model

The Bayesian model uses the same away-form feature set, plus hierarchical season and team effects.

## Validation

The Bayesian script uses a temporal holdout split:

- training: seasons earlier than the validation season
- validation: the latest completed season before the final season

This avoids training on future match outcomes relative to the validation season.

## Final prediction

The final prediction is adjusted for a neutral venue by symmetrizing both team orderings rather than treating Paris as a true home team.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run

```bash
python predict_final_form_rf.py
python predict_final_hierarchical_bayes.py
```

## Notes

- The models are built for research and analysis, not as a general football forecasting system.
- The Bayesian script prints both the temporal holdout comparison and the final neutral-venue probability for Paris vs Arsenal.
