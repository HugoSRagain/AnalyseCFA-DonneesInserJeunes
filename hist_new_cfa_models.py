from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# =========================================================
# PARAMETRES
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "fr-en-inserjeunes-cfa.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "subsample_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_ORDER = [
    "cumul 2018-2019",
    "cumul 2019-2020",
    "cumul 2020-2021",
    "cumul 2021-2022",
    "cumul 2022-2023",
    "cumul 2023-2024"
]

EARLY_YEARS = ["cumul 2018-2019", "cumul 2019-2020"]


# =========================================================
# FONCTIONS
# =========================================================
def read_inserjeunes_csv(file_path: Path) -> pd.DataFrame:
    separators_to_try = [";", ",", "\t"]
    df = None
    for sep in separators_to_try:
        try:
            tmp = pd.read_csv(file_path, sep=sep, encoding="utf-8-sig")
            if tmp.shape[1] >= 8:
                df = tmp.copy()
                print(f"Séparateur retenu : {repr(sep)}")
                break
        except Exception:
            continue
    if df is None:
        raise ValueError("Impossible de lire correctement le fichier CSV.")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    work = df.rename(columns={
        "Année": "year",
        "UAI": "uai",
        "Libellé": "label",
        "Région": "region",
        "Taux poursuite études": "study",
        "Taux emploi 6 mois": "emp6",
        "VA emploi 6 mois": "va6",
        "Taux emploi 12 mois": "emp12"
    }).copy()

    for col in ["study", "emp6", "va6", "emp12"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in ["year", "uai", "label", "region"]:
        work[col] = work[col].astype(str).str.strip()

    work = work[work["year"].isin(YEAR_ORDER)].copy()
    sample = work.dropna(subset=["year", "uai", "study", "emp6", "va6", "emp12"]).copy()

    historical_uai = sample.loc[sample["year"].isin(EARLY_YEARS), "uai"].unique()
    sample["new_cfa"] = (~sample["uai"].isin(historical_uai)).astype(int)

    return sample


def fit_fe_subsample(data: pd.DataFrame, depvar: str):
    formula = f"{depvar} ~ study + C(uai) + C(year)"
    model = smf.ols(formula=formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["uai"]}
    )
    return model


def star(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt_coef(coef, p):
    if pd.isna(coef):
        return ""
    return f"{coef:.4f}{star(p)}"


def fmt_se(se):
    if pd.isna(se):
        return ""
    return f"({se:.4f})"


# =========================================================
# EXECUTION
# =========================================================
df = read_inserjeunes_csv(DATA_PATH)
sample = prepare_data(df)

hist_sample = sample[sample["new_cfa"] == 0].copy()
new_sample = sample[sample["new_cfa"] == 1].copy()

specs = [
    ("emp6", "Emploi 6 mois"),
    ("va6", "VA emploi 6 mois"),
    ("emp12", "Emploi 12 mois")
]

rows = []
for depvar, dep_label in specs:
    m_hist = fit_fe_subsample(hist_sample, depvar)
    m_new = fit_fe_subsample(new_sample, depvar)

    rows.append({
        "Variable dépendante": dep_label,
        "Échantillon": "CFA historiques",
        "Coef. study": m_hist.params.get("study", np.nan),
        "SE study": m_hist.bse.get("study", np.nan),
        "p-value study": m_hist.pvalues.get("study", np.nan),
        "R2": m_hist.rsquared,
        "N": int(m_hist.nobs)
    })
    rows.append({
        "Variable dépendante": dep_label,
        "Échantillon": "Nouveaux CFA",
        "Coef. study": m_new.params.get("study", np.nan),
        "SE study": m_new.bse.get("study", np.nan),
        "p-value study": m_new.pvalues.get("study", np.nan),
        "R2": m_new.rsquared,
        "N": int(m_new.nobs)
    })

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_DIR / "subsample_hist_new_results.csv", index=False)
print(results)

# =========================================================
# TABLE LATEX
# =========================================================
latex_table = r"""
\begin{table}[H]
\centering
\caption{Estimations séparées sur sous-échantillons historiques et nouveaux CFA}
\label{tab:subsamples_hist_new}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{\textbf{Emploi 6m}} & \multicolumn{2}{c}{\textbf{VA 6m}} & \multicolumn{2}{c}{\textbf{Emploi 12m}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
& \textbf{Hist.} & \textbf{Nvx} & \textbf{Hist.} & \textbf{Nvx} & \textbf{Hist.} & \textbf{Nvx} \\
\midrule
"""

coef_line = ["Taux poursuite études"]
se_line = [""]
r2_line = ["$R^2$"]
n_line = ["Observations"]

for dep in ["Emploi 6 mois", "VA emploi 6 mois", "Emploi 12 mois"]:
    row_hist = results[(results["Variable dépendante"] == dep) & (results["Échantillon"] == "CFA historiques")].iloc[0]
    row_new = results[(results["Variable dépendante"] == dep) & (results["Échantillon"] == "Nouveaux CFA")].iloc[0]

    coef_line.extend([
        fmt_coef(row_hist["Coef. study"], row_hist["p-value study"]),
        fmt_coef(row_new["Coef. study"], row_new["p-value study"])
    ])
    se_line.extend([
        fmt_se(row_hist["SE study"]),
        fmt_se(row_new["SE study"])
    ])
    r2_line.extend([
        f"{row_hist['R2']:.3f}",
        f"{row_new['R2']:.3f}"
    ])
    n_line.extend([
        f"{int(row_hist['N']):,}".replace(",", "\\,"),
        f"{int(row_new['N']):,}".replace(",", "\\,")
    ])

latex_table += " & ".join(coef_line) + r" \\" + "\n"
latex_table += " & ".join(se_line) + r" \\" + "\n"
latex_table += r"\addlinespace" + "\n"
latex_table += " & ".join(r2_line) + r" \\" + "\n"
latex_table += " & ".join(n_line) + r" \\" + "\n"
latex_table += r"""
\bottomrule
\end{tabular}

\vspace{0.2cm}
\begin{minipage}{0.95\textwidth}
\footnotesize
\textit{Notes :} modèles linéaires avec effets fixes CFA et année, estimés séparément sur les sous-échantillons historiques et nouveaux CFA. Erreurs standards clusterisées au niveau du CFA entre parenthèses. *** $p<0.01$, ** $p<0.05$, * $p<0.1$.
\end{minipage}
\end{table}
"""

with open(OUTPUT_DIR / "table_subsamples_hist_new.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print(latex_table)
print(f"Fichiers créés dans : {OUTPUT_DIR}")
