# Toronto Neighbourhood Crime Classification

Supervised machine learning project that classifies Toronto's 158 neighbourhoods
into crime-rate levels (low / medium / high) and explores which crime types
drive those differences.

Final project for SCS 3253 – Machine Learning, University of Toronto (2026).

## Data

- **Source:** [Toronto Police Service – Neighbourhood Crime Rates Open Data](https://data.tps.ca/datasets/TorontoPS::neighbourhood-crime-rates-open-data/about)
- **Size:** 158 neighbourhoods × 222 features (crime counts and rates per 100,000 people, by crime type, 2014–2025)
- **Format:** CSV (`crime_data.csv`, included in this repository)
- **License:** Contains information licensed under the Open Government Licence – Toronto.

## Approach

1. **Exploration:** crime-rate distributions are strongly right-skewed — most
   neighbourhoods have low crime, while a few concentrate very high rates.
2. **Feature selection:** only 2023 crime *rates* were used (not raw counts), so
   neighbourhoods of different population sizes are comparable.
3. **Target engineering:** the dataset has no label, so a `total_crime_rate` was
   computed and split into three balanced classes (low / medium / high) using
   quantile binning.
4. **Models:** Logistic Regression and Random Forest, with an 80/20 train-test
   split and standard scaling.
5. **Tuning:** Grid Search over Random Forest hyperparameters (`n_estimators`, `max_depth`).

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| **Logistic Regression** | **0.94** | **0.94** |
| Random Forest | 0.81 | 0.81 |
| Random Forest (tuned: `max_depth=5`, `n_estimators=100`) | 0.84 | 0.84 |

- Logistic Regression performed best, suggesting a largely linear structure in the data.
- Low-crime neighbourhoods were classified almost perfectly. Most errors came
  from confusing medium- and high-crime neighbourhoods.
- Highest overall crime rates: Yonge-Bay Corridor, Downtown Yonge East and
  West Humber-Clairville.

## Limitations

The target is derived from the same 2023 crime rates used as features, so the
models mainly learn to reproduce that grouping rather than forecast crime.
A natural next step is to predict a neighbourhood's 2025 crime level from
earlier years (2014–2024) to make the task genuinely predictive.

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Google Colab

## How to run

Open the notebook in Google Colab using the badge at the top and run all cells —
the dataset is loaded directly from this repository.
