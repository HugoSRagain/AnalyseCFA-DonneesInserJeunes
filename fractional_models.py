from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

# =========================================================
# PARAMETRES
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "fr-en-inserjeunes-cfa.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "fractional_models"
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
        "Taux emploi 12 mois": "emp12"
    }).copy()

    for col in ["study", "emp6", "emp12"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in ["year", "uai", "label", "region"]:
        work[col] = work[col].astype(str).str.strip()

    work = work[work["year"].isin(YEAR_ORDER)].copy()
    sample = work.dropna(subset=["year", "uai", "study", "emp6", "emp12"]).copy()

    historical_uai = sample.loc[sample["year"].isin(EARLY_YEARS), "uai"].unique()
    sample["new_cfa"] = (~sample["uai"].isin(historical_uai)).astype(int)

    sample["emp6_share"] = sample["emp6"] / 100.0
    sample["emp12_share"] = sample["emp12"] / 100.0

    sample["study_between"] = sample.groupby("uai")["study"].transform("mean")
    sample["study_within"] = sample["study"] - sample["study_between"]

    sample["study_within_new"] = sample["study_within"] * sample["new_cfa"]
    sample["study_between_new"] = sample["study_between"] * sample["new_cfa"]

    return sample


def fit_fractional(data: pd.DataFrame, depvar_share: str):
    formula = (
        f"{depvar_share} ~ study_within + study_between + "
        f"study_within_new + study_between_new + C(year)"
    )
    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Binomial()
    ).fit(
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


def linear_combo(model, var1: str, var2: str):
    b1 = model.params.get(var1, np.nan)
    b2 = model.params.get(var2, np.nan)
    cov = model.cov_params()

    est = b1 + b2
    var = cov.loc[var1, var1] + cov.loc[var2, var2] + 2 * cov.loc[var1, var2]
    se = np.sqrt(var) if var >= 0 else np.nan
    z = est / se if pd.notna(se) and se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if pd.notna(z) else np.nan
    return est, se, p


def pseudo_r2(model):
    if hasattr(model, "null_deviance") and model.null_deviance != 0:
        return 1 - model.deviance / model.null_deviance
    return np.nan


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

models = {
    "emp6_share": fit_fractional(sample, "emp6_share"),
    "emp12_share": fit_fractional(sample, "emp12_share")
}

rows = []
for dep, model in models.items():
    total_within_new, total_within_new_se, total_within_new_p = linear_combo(
        model, "study_within", "study_within_new"
    )
    total_between_new, total_between_new_se, total_between_new_p = linear_combo(
        model, "study_between", "study_between_new"
    )

    rows.append({
        "Variable dépendante": dep,
        "Coef. within historiques": model.params.get("study_within", np.nan),
        "SE within historiques": model.bse.get("study_within", np.nan),
        "p within historiques": model.pvalues.get("study_within", np.nan),
        "Coef. within nouveaux diff": model.params.get("study_within_new", np.nan),
        "SE within nouveaux diff": model.bse.get("study_within_new", np.nan),
        "p within nouveaux diff": model.pvalues.get("study_within_new", np.nan),
        "Coef. within nouveaux total": total_within_new,
        "SE within nouveaux total": total_within_new_se,
        "p within nouveaux total": total_within_new_p,

        "Coef. between historiques": model.params.get("study_between", np.nan),
        "SE between historiques": model.bse.get("study_between", np.nan),
        "p between historiques": model.pvalues.get("study_between", np.nan),
        "Coef. between nouveaux diff": model.params.get("study_between_new", np.nan),
        "SE between nouveaux diff": model.bse.get("study_between_new", np.nan),
        "p between nouveaux diff": model.pvalues.get("study_between_new", np.nan),
        "Coef. between nouveaux total": total_between_new,
        "SE between nouveaux total": total_between_new_se,
        "p between nouveaux total": total_between_new_p,

        "Pseudo-R2": pseudo_r2(model),
        "N": int(model.nobs)
    })

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_DIR / "fractional_logit_results.csv", index=False)
print(results)

# =========================================================
# TABLE LATEX
# =========================================================
lines = {
    "Within (CFA historiques)": [],
    "SE Within hist": [],
    "Supplément Within nouveaux CFA": [],
    "SE Suppl. Within": [],
    "Within total nouveaux CFA": [],
    "SE Within total": [],
    "Between (CFA historiques)": [],
    "SE Between hist": [],
    "Supplément Between nouveaux CFA": [],
    "SE Suppl. Between": [],
    "Between total nouveaux CFA": [],
    "SE Between total": [],
    "Pseudo-$R^2$": [],
    "Observations": []
}

for dep in ["emp6_share", "emp12_share"]:
    row = results.loc[results["Variable dépendante"] == dep].iloc[0]

    lines["Within (CFA historiques)"].append(fmt_coef(row["Coef. within historiques"], row["p within historiques"]))
    lines["SE Within hist"].append(fmt_se(row["SE within historiques"]))

    lines["Supplément Within nouveaux CFA"].append(fmt_coef(row["Coef. within nouveaux diff"], row["p within nouveaux diff"]))
    lines["SE Suppl. Within"].append(fmt_se(row["SE within nouveaux diff"]))

    lines["Within total nouveaux CFA"].append(fmt_coef(row["Coef. within nouveaux total"], row["p within nouveaux total"]))
    lines["SE Within total"].append(fmt_se(row["SE within nouveaux total"]))

    lines["Between (CFA historiques)"].append(fmt_coef(row["Coef. between historiques"], row["p between historiques"]))
    lines["SE Between hist"].append(fmt_se(row["SE between historiques"]))

    lines["Supplément Between nouveaux CFA"].append(fmt_coef(row["Coef. between nouveaux diff"], row["p between nouveaux diff"]))
    lines["SE Suppl. Between"].append(fmt_se(row["SE between nouveaux diff"]))

    lines["Between total nouveaux CFA"].append(fmt_coef(row["Coef. between nouveaux total"], row["p between nouveaux total"]))
    lines["SE Between total"].append(fmt_se(row["SE between nouveaux total"]))

    lines["Pseudo-$R^2$"].append(f"{row['Pseudo-R2']:.3f}")
    lines["Observations"].append(f"{int(row['N']):,}".replace(",", "\\,"))

latex_table = rf"""
\begin{{table}}[H]
\centering
\caption{{Robustesses en modèle fractionnaire}}
\label{{tab:fractional_logit}}
\begin{{tabular}}{{lcc}}
\toprule
& \textbf{{(1)}} & \textbf{{(2)}} \\
& \textbf{{Emploi 6m}} & \textbf{{Emploi 12m}} \\
\midrule
Within (CFA historiques) & {' & '.join(lines["Within (CFA historiques)"])} \\
 & {' & '.join(lines["SE Within hist"])} \\
\addlinespace
Supplément Within nouveaux CFA & {' & '.join(lines["Supplément Within nouveaux CFA"])} \\
 & {' & '.join(lines["SE Suppl. Within"])} \\
\addlinespace
Within total nouveaux CFA & {' & '.join(lines["Within total nouveaux CFA"])} \\
 & {' & '.join(lines["SE Within total"])} \\
\addlinespace
Between (CFA historiques) & {' & '.join(lines["Between (CFA historiques)"])} \\
 & {' & '.join(lines["SE Between hist"])} \\
\addlinespace
Supplément Between nouveaux CFA & {' & '.join(lines["Supplément Between nouveaux CFA"])} \\
 & {' & '.join(lines["SE Suppl. Between"])} \\
\addlinespace
Between total nouveaux CFA & {' & '.join(lines["Between total nouveaux CFA"])} \\
 & {' & '.join(lines["SE Between total"])} \\
\addlinespace
Effets fixes année & Oui & Oui \\
\addlinespace
Pseudo-$R^2$ & {' & '.join(lines["Pseudo-$R^2$"])} \\
Observations & {' & '.join(lines["Observations"])} \\
\bottomrule
\end{{tabular}}

\vspace{{0.2cm}}
\begin{{minipage}}{{0.95\textwidth}}
\footnotesize
\textit{{Notes :}} modèles de type \textit{{fractional logit}} estimés sur les taux d'emploi ramenés à l'intervalle $[0,1]$. La décomposition \textit{{within}}/\textit{{between}} suit une approche de type Mundlak. Erreurs standards clusterisées au niveau du CFA entre parenthèses. Les coefficients doivent être interprétés comme des associations conditionnelles. *** $p<0.01$, ** $p<0.05$, * $p<0.1$.
\end{{minipage}}
\end{{table}}
"""

with open(OUTPUT_DIR / "table_fractional_logit.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print(latex_table)
print(f"Fichiers créés dans : {OUTPUT_DIR}")
