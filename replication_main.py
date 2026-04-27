from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import norm

# =========================================================
# PARAMETRES
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "fr-en-inserjeunes-cfa.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "main_models"
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
# FONCTIONS OUTILS
# =========================================================
def read_inserjeunes_csv(file_path: Path) -> pd.DataFrame:
    separators_to_try = [";", ",", "\t"]
    df = None
    chosen_sep = None

    for sep in separators_to_try:
        try:
            tmp = pd.read_csv(file_path, sep=sep, encoding="utf-8-sig")
            if tmp.shape[1] >= 8:
                df = tmp.copy()
                chosen_sep = sep
                break
        except Exception:
            continue

    if df is None:
        raise ValueError("Impossible de lire correctement le fichier CSV.")

    print(f"Séparateur retenu : {repr(chosen_sep)}")
    print(f"Dimensions brutes : {df.shape}")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    needed_cols = [
        "Année",
        "UAI",
        "Libellé",
        "Région",
        "Taux poursuite études",
        "Taux emploi 6 mois",
        "VA emploi 6 mois",
        "Taux emploi 12 mois"
    ]

    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

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
    sample["study_new"] = sample["study"] * sample["new_cfa"]

    print(f"Observations échantillon principal : {len(sample)}")
    print(f"CFA distincts : {sample['uai'].nunique()}")
    print(f"Nouveaux CFA : {sample.loc[sample['new_cfa'] == 1, 'uai'].nunique()}")
    print(f"CFA historiques : {sample.loc[sample['new_cfa'] == 0, 'uai'].nunique()}")

    return sample


def fit_fe_interaction(data: pd.DataFrame, depvar: str):
    formula = f"{depvar} ~ study + study_new + C(uai) + C(year)"
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
    "emp6": fit_fe_interaction(sample, "emp6"),
    "va6": fit_fe_interaction(sample, "va6"),
    "emp12": fit_fe_interaction(sample, "emp12")
}

# =========================================================
# TABLEAU RESULTATS
# =========================================================
rows = []
for dep_label, model in models.items():
    total_new, total_new_se, total_new_p = linear_combo(model, "study", "study_new")

    rows.append({
        "Variable dépendante": dep_label,
        "Coef. historiques": model.params.get("study", np.nan),
        "SE historiques": model.bse.get("study", np.nan),
        "p historiques": model.pvalues.get("study", np.nan),
        "Coef. interaction nouveaux": model.params.get("study_new", np.nan),
        "SE interaction nouveaux": model.bse.get("study_new", np.nan),
        "p interaction nouveaux": model.pvalues.get("study_new", np.nan),
        "Coef. total nouveaux": total_new,
        "SE total nouveaux": total_new_se,
        "p total nouveaux": total_new_p,
        "R2": model.rsquared,
        "N": int(model.nobs)
    })

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_DIR / "fe_interaction_hist_new_results.csv", index=False)
print(results)

# =========================================================
# TABLE LATEX
# =========================================================
coef_hist = ["Taux poursuite études (CFA historiques)"]
se_hist = [""]

coef_inter = ["Interaction \\; $\\times$ \\; nouveaux CFA"]
se_inter = [""]

coef_total = ["Association totale (nouveaux CFA)"]
se_total = [""]

obs_line = ["Observations"]
r2_line = ["$R^2$"]

for dep in ["emp6", "va6", "emp12"]:
    row = results.loc[results["Variable dépendante"] == dep].iloc[0]

    coef_hist.append(fmt_coef(row["Coef. historiques"], row["p historiques"]))
    se_hist.append(fmt_se(row["SE historiques"]))

    coef_inter.append(fmt_coef(row["Coef. interaction nouveaux"], row["p interaction nouveaux"]))
    se_inter.append(fmt_se(row["SE interaction nouveaux"]))

    coef_total.append(fmt_coef(row["Coef. total nouveaux"], row["p total nouveaux"]))
    se_total.append(fmt_se(row["SE total nouveaux"]))

    obs_line.append(f"{int(row['N']):,}".replace(",", "\\,"))
    r2_line.append(f"{row['R2']:.3f}")

latex_table = rf"""
\begin{{table}}[H]
\centering
\caption{{Associations entre poursuite d'études et insertion selon l'ancienneté des CFA}}
\label{{tab:fe_interaction_hist_new}}
\begin{{tabular}}{{lccc}}
\toprule
& \textbf{{(1)}} & \textbf{{(2)}} & \textbf{{(3)}} \\
& \textbf{{Emploi 6m}} & \textbf{{VA 6m}} & \textbf{{Emploi 12m}} \\
\midrule
{' & '.join(coef_hist)} \\
{' & '.join(se_hist)} \\
\addlinespace
{' & '.join(coef_inter)} \\
{' & '.join(se_inter)} \\
\addlinespace
{' & '.join(coef_total)} \\
{' & '.join(se_total)} \\
\addlinespace
Effets fixes CFA & Oui & Oui & Oui \\
Effets fixes année & Oui & Oui & Oui \\
\addlinespace
{' & '.join(obs_line)} \\
{' & '.join(r2_line)} \\
\bottomrule
\end{{tabular}}

\vspace{{0.2cm}}
\begin{{minipage}}{{0.95\textwidth}}
\footnotesize
\textit{{Notes :}} erreurs standards clusterisées au niveau du CFA entre parenthèses. Les coefficients mesurent des associations conditionnelles. La ligne ``Association totale (nouveaux CFA)'' correspond à la somme du coefficient principal et du terme d'interaction. *** $p<0.01$, ** $p<0.05$, * $p<0.1$.
\end{{minipage}}
\end{{table}}
"""

with open(OUTPUT_DIR / "table_fe_interaction_hist_new.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print(latex_table)

# =========================================================
# GRAPHIQUE PRINCIPAL
# =========================================================
dep_names = {
    "emp6": "Emploi 6 mois",
    "va6": "VA emploi 6 mois",
    "emp12": "Emploi 12 mois"
}

plot_rows = []
for dep in ["emp6", "va6", "emp12"]:
    row = results.loc[results["Variable dépendante"] == dep].iloc[0]

    plot_rows.append({
        "label": f"{dep_names[dep]} | Historiques",
        "coef": row["Coef. historiques"],
        "se": row["SE historiques"]
    })
    plot_rows.append({
        "label": f"{dep_names[dep]} | Nouveaux",
        "coef": row["Coef. total nouveaux"],
        "se": row["SE total nouveaux"]
    })

plot_df = pd.DataFrame(plot_rows)
y_pos = np.arange(len(plot_df))

plt.figure(figsize=(10, 6))
plt.errorbar(
    x=plot_df["coef"],
    y=y_pos,
    xerr=1.96 * plot_df["se"],
    fmt="o",
    capsize=4
)
plt.axvline(x=0, linestyle="--")
plt.yticks(y_pos, plot_df["label"])
plt.xlabel("Coefficient estimé du taux de poursuite d'études")
plt.title("Associations estimées entre poursuite d'études et insertion selon le statut des CFA")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "coef_hist_nouveaux_fe.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Fichiers créés dans : {OUTPUT_DIR}")
