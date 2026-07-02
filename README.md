# Global AI & Data Jobs — Salary Prediction

Project for training and evaluating regression models that predict salary (`salary_usd`) for AI/Data roles worldwide, based on the public Kaggle dataset:

[`mohankrishnathalla/global-ai-and-data-jobs-salary-dataset`](https://www.kaggle.com/datasets/mohankrishnathalla/global-ai-and-data-jobs-salary-dataset)

## About the dataset

This dataset contains a large global collection of technology job records from 2020 to 2026, covering roles in Artificial Intelligence, Data Science, Machine Learning, Software Engineering, and Analytics.

It includes detailed job-level attributes such as role category, country, company size, experience level, work setup (remote, hybrid, onsite), salary in USD, required skills, and workplace indicators including promotion speed, job security, and work-life balance.

> **Note:** this is a **synthetic (artificially generated) dataset**, not real-world collected data. It is well suited for practicing data cleaning, feature engineering, and regression modeling, but results should not be interpreted as reflecting actual labor market salaries.

Two models are compared:

- **Random Forest Regressor**
- **XGBoost Regressor**

Evaluation is performed via **5-fold cross-validation**, using **MAE** (Mean Absolute Error) and **R²** as metrics.

## Requirements

- Python 3.9+
- Required packages:

  ```bash
  pip install kagglehub pandas numpy matplotlib seaborn scikit-learn xgboost
  ```

- A configured Kaggle account for `kagglehub` (needed to download the dataset). See the [official documentation](https://github.com/Kaggle/kagglehub).

## Project structure

```
salary_prediction.py    # main script
output_plots/            # auto-generated folder
├── salary_distribution.png
├── experience_vs_salary.png
├── feature_importance.png
└── real_vs_predicted.png
README.md
```

## How to run

```bash
python salary_prediction.py
```

The script:

1. **Downloads** the dataset from Kaggle (via `kagglehub`)
2. **Explores the data (EDA)**: prints dataset shape, `df.info()`, descriptive statistics for both numeric and categorical columns, and checks for missing values and duplicate rows
3. **Cleans the data**: drops rows with missing values and duplicate rows
4. **Visualizes the data**:
   - overall salary distribution
   - relationship between seniority/experience and salary (auto-detects whether the experience column is numeric — years of experience — or categorical, e.g. Entry/Mid/Senior/Executive — and picks a regression plot or a boxplot accordingly)
5. **Encodes categorical variables** (one-hot encoding)
6. **Trains and evaluates** Random Forest and XGBoost with 5-fold cross-validation
7. **Checks feature importance**: prints the top 15 features according to XGBoost and flags a warning if a single feature accounts for more than 50% of total importance (possible *data leakage*)
8. **Generates and saves** all plots in `output_plots/`

## Generated plots

| File | Description |
|------|--------------|
| `salary_distribution.png` | Histogram (with KDE) of the overall salary distribution |
| `experience_vs_salary.png` | Salary vs. seniority/years of experience — scatter plot with regression line (numeric experience) or boxplot ordered by median salary (categorical experience level) |
| `feature_importance.png` | Top 15 most important features according to the trained XGBoost model |
| `real_vs_predicted.png` | Actual vs. predicted salary from cross-validated XGBoost predictions |

## Results (example run)

| Model         | MAE                  | R²     |
|---------------|----------------------|--------|
| Random Forest | $8,595.06 ± $66.75   | 0.9340 |
| XGBoost       | $8,278.88 ± $37.95   | 0.9399 |

> **Note:** values may vary slightly depending on the dataset version downloaded and the execution environment.

## Analisi e Risultati

### 1. Data Distribution and Relationships
![Salary Distribution](output_plots/salary_distribution.png)
![Experience vs. Salary](output_plots/experience_vs_salary.png)

### 2. Model Performance
![Feature Importance](output_plots/feature_importance.png)
![Actual vs. Predicted Salaries](output_plots/real_vs_predicted.png)

## Possible future improvements

- Hyperparameter tuning (e.g. `RandomizedSearchCV`, Optuna)
- More sophisticated handling of high-cardinality categorical variables (e.g. target encoding)
- Dedicated holdout split for final model validation
- Model interpretability with SHAP

## License

This project is distributed for educational/demonstration purposes. The original dataset is subject to the usage terms defined on Kaggle by its author, and — as noted above — is synthetic in nature.
