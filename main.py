import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Cartella di output per i grafici
OUTPUT_DIR = "output_plots"

# Possibili nomi di colonna che indicano l'anzianità / esperienza
EXPERIENCE_COLUMN_CANDIDATES = [
    "years_experience",
    "experience_years",
    "yoe",
    "years_of_experience",
    "experience_level",
    "seniority",
    "seniority_level",
]


def find_experience_column(df):
    """Cerca tra le colonne del dataframe una colonna legata all'anzianità/esperienza."""
    for col in EXPERIENCE_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # 1. DOWNLOAD E CARICAMENTO DEL DATASET
    # ==========================================
    print("Download del dataset in corso tramite kagglehub...")
    dataset_dir = kagglehub.dataset_download("mohankrishnathalla/global-ai-and-data-jobs-salary-dataset")

    # Percorso del file CSV scaricato
    csv_path = os.path.join(dataset_dir, "global_ai_jobs.csv")
    print(f"File caricato da: {csv_path}\n")

    df = pd.read_csv(csv_path)

    # ==========================================
    # 2. ESPLORAZIONE DEI DATI (EDA)
    # ==========================================
    print("=" * 50)
    print("ESPLORAZIONE DEI DATI (EDA)")
    print("=" * 50)

    print(f"\nDimensioni del dataset: {df.shape[0]} righe, {df.shape[1]} colonne\n")

    print("Info sul dataset:")
    df.info()

    print("\nStatistiche descrittive (variabili numeriche):")
    print(df.describe())

    print("\nStatistiche descrittive (variabili categoriche):")
    print(df.describe(include=["object", "category"]))

    # Verifica della presenza di valori mancanti e duplicati
    missing_pct = df.isnull().mean() * 100
    print("\nPercentuale di dati mancanti:")
    print(missing_pct)
    print(f"\nRighe duplicate totali: {df.duplicated().sum()}\n")

    # Gestione dei valori mancanti: rimuoviamo le righe con NaN
    # (se una colonna avesse troppi missing, andrebbe valutata un'imputazione
    # o l'eliminazione della colonna stessa invece della riga)
    righe_iniziali = len(df)
    df = df.dropna()
    print(f"Righe rimosse per valori mancanti: {righe_iniziali - len(df)}")

    # Rimozione dei duplicati
    righe_prima_dedup = len(df)
    df = df.drop_duplicates()
    print(f"Righe rimosse per duplicati: {righe_prima_dedup - len(df)}\n")

    sns.set_theme(style="whitegrid")

    # ------------------------------------------
    # 2a. Grafico: distribuzione dello stipendio
    # ------------------------------------------
    plt.figure(figsize=(9, 6))
    sns.histplot(df["salary_usd"], bins=40, kde=True, color="#1f77b4")
    plt.title("Distribuzione dello stipendio (USD)", fontsize=14, fontweight="bold")
    plt.xlabel("Stipendio ($)", fontsize=12)
    plt.ylabel("Frequenza", fontsize=12)
    plt.tight_layout()
    salary_dist_path = os.path.join(OUTPUT_DIR, "salary_distribution.png")
    plt.savefig(salary_dist_path, dpi=300, bbox_inches="tight")
    print(f"Grafico salvato in: {salary_dist_path}")
    plt.show()
    plt.close()

    # ------------------------------------------
    # 2b. Grafico: anzianità (esperienza) vs stipendio
    # ------------------------------------------
    experience_col = find_experience_column(df)

    if experience_col is not None:
        print(f"\nColonna relativa all'anzianità trovata: '{experience_col}'")

        plt.figure(figsize=(9, 6))

        if pd.api.types.is_numeric_dtype(df[experience_col]):
            # Anzianità espressa come valore numerico (anni di esperienza)
            sns.regplot(
                x=df[experience_col],
                y=df["salary_usd"],
                scatter_kws={"alpha": 0.4, "color": "#1f77b4"},
                line_kws={"color": "red"},
            )
            plt.xlabel("Anni di esperienza", fontsize=12)
        else:
            # Anzianità espressa come categoria (es. Entry, Mid, Senior, Executive)
            order = (
                df.groupby(experience_col)["salary_usd"]
                .median()
                .sort_values()
                .index
            )
            sns.boxplot(x=experience_col, y="salary_usd", data=df, order=order, color="#1f77b4")
            plt.xlabel("Livello di esperienza / anzianità", fontsize=12)
            plt.xticks(rotation=30)

        plt.title("Relazione tra anzianità/esperienza e stipendio", fontsize=14, fontweight="bold")
        plt.ylabel("Stipendio ($)", fontsize=12)
        plt.tight_layout()

        experience_plot_path = os.path.join(OUTPUT_DIR, "experience_vs_salary.png")
        plt.savefig(experience_plot_path, dpi=300, bbox_inches="tight")
        print(f"Grafico salvato in: {experience_plot_path}")
        plt.show()
        plt.close()
    else:
        print(
            "\nATTENZIONE: nessuna colonna relativa all'anzianità/esperienza trovata "
            f"tra {EXPERIENCE_COLUMN_CANDIDATES}. Grafico anzianità vs stipendio saltato. "
            "Controllare i nomi delle colonne con df.columns e aggiornare "
            "EXPERIENCE_COLUMN_CANDIDATES se necessario."
        )

    # ==========================================
    # 3. PREPARAZIONE DEI DATI
    # ==========================================
    # Separazione tra variabili esplicative e variabile target
    X = df.drop(columns=['salary_usd'])
    y = df['salary_usd']

    # Conversione delle variabili categoriche in variabili dummy
    X = pd.get_dummies(X, drop_first=True)

    print(f"\nNumero di feature dopo one-hot encoding: {X.shape[1]}\n")

    # ==========================================
    # 4. ADDESTRAMENTO E VALUTAZIONE DEI MODELLI
    # ==========================================
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }

    # Random Forest
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)

    print("Esecuzione della cross-validation per Random Forest...")
    rf_cv_results = cross_validate(
        rf_model, X, y,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1
    )

    rf_mae_mean = -np.mean(rf_cv_results['test_mae'])
    rf_mae_std = np.std(rf_cv_results['test_mae'])
    rf_r2_mean = np.mean(rf_cv_results['test_r2'])

    print("\n" + "=" * 50)
    print("RISULTATI CROSS-VALIDATION (5 FOLD)")
    print("=" * 50)
    print("\nRandom Forest Regressor")
    print(f"MAE: ${rf_mae_mean:,.2f} ± ${rf_mae_std:,.2f}")
    print(f"R² : {rf_r2_mean:.4f}")

    # XGBoost
    xgb_model = XGBRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1
    )

    print("\nEsecuzione della cross-validation per XGBoost...")
    xgb_cv_results = cross_validate(
        xgb_model, X, y,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1
    )

    xgb_mae_mean = -np.mean(xgb_cv_results['test_mae'])
    xgb_mae_std = np.std(xgb_cv_results['test_mae'])
    xgb_r2_mean = np.mean(xgb_cv_results['test_r2'])

    print("\nXGBoost Regressor")
    print(f"MAE: ${xgb_mae_mean:,.2f} ± ${xgb_mae_std:,.2f}")
    print(f"R² : {xgb_r2_mean:.4f}")
    print("=" * 50)

    # ==========================================
    # 5. CONTROLLO FEATURE IMPORTANCE (leakage check)
    # ==========================================
    print("\nAddestramento XGBoost su tutto il dataset per il controllo delle feature importance...")
    xgb_model.fit(X, y)
    importances = pd.Series(
        xgb_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 15 feature per importanza (XGBoost):")
    print(importances.head(15))

    top_feature_share = importances.iloc[0] / importances.sum()
    if top_feature_share > 0.5:
        print(
            f"\nATTENZIONE: la feature '{importances.index[0]}' rappresenta da sola "
            f"il {top_feature_share:.1%} dell'importanza totale. "
            f"Possibile data leakage, verificare la colonna."
        )

    # Grafico feature importance
    plt.figure(figsize=(10, 8))
    top_n = importances.head(15)
    sns.barplot(x=top_n.values, y=top_n.index, color="#1f77b4")
    plt.title("Top 15 Feature Importance (XGBoost)", fontsize=14, fontweight="bold")
    plt.xlabel("Importanza", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    feature_importance_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(feature_importance_path, dpi=300, bbox_inches="tight")
    print(f"\nGrafico salvato in: {feature_importance_path}")
    plt.show()
    plt.close()

    # ==========================================
    # 6. GRAFICO: VALORI REALI VS PREVISIONI
    # ==========================================
    print("\nCalcolo delle previsioni per la visualizzazione...")

    # Predizioni ottenute tramite cross-validation
    xgb_pred_cv = cross_val_predict(
        xgb_model,
        X,
        y,
        cv=kfold,
        n_jobs=-1
    )

    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x=y,
        y=xgb_pred_cv,
        alpha=0.6,
        color="#1f77b4",
        edgecolor=None
    )

    # Linea di riferimento (predizione perfetta)
    min_val = min(y.min(), xgb_pred_cv.min())
    max_val = max(y.max(), xgb_pred_cv.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
        linewidth=2
    )

    plt.title(
        "XGBoost: Stipendi reali vs stipendi previsti",
        fontsize=14,
        fontweight="bold"
    )
    plt.xlabel("Stipendio reale ($)", fontsize=12)
    plt.ylabel("Stipendio previsto ($)", fontsize=12)

    plt.tight_layout()

    scatter_path = os.path.join(OUTPUT_DIR, "real_vs_predicted.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    print(f"Grafico salvato in: {scatter_path}")
    plt.show()
    plt.close()

    print(f"\nTutti i grafici sono stati salvati nella cartella '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()