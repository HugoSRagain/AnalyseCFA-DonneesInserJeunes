import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

file_path = "/content/fr-en-inserjeunes-cfa.csv"
output_dir = "/content/analyse_inserjeunes_rehaussee"
os.makedirs(output_dir, exist_ok=True)

separators_to_try = [";", ",", "\t"]

df = None
chosen_sep = None

for sep in separators_to_try:
    try:
        tmp = pd.read_csv(file_path, sep=sep, encoding="utf-8-sig")
        print(f"Test séparateur '{sep}' : shape = {tmp.shape}")
        if tmp.shape[1] >= 8:
            df = tmp.copy()
            chosen_sep = sep
            break
    except Exception as e:
        print(f"Echec avec séparateur '{sep}' : {e}")

if df is None:
    raise ValueError("Impossible de lire correctement le fichier avec les séparateurs testés.")

print("\nSéparateur retenu :", repr(chosen_sep))
print("Dimensions brutes :", df.shape)
print("\nColonnes disponibles :")
print(df.columns.tolist())

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

print("\nToutes les colonnes utiles sont présentes.")

work = df.copy()

work = work.rename(columns={
    "Année": "year",
    "UAI": "uai",
    "Libellé": "label",
    "Région": "region",
    "Taux poursuite études": "study",
    "Taux emploi 6 mois": "emp6",
    "VA emploi 6 mois": "va6",
    "Taux emploi 12 mois": "emp12"
})

num_vars = ["study", "emp6", "va6", "emp12"]

for col in num_vars:
    work[col] = pd.to_numeric(work[col], errors="coerce")

for col in ["year", "uai", "label", "region"]:
    work[col] = work[col].astype(str).str.strip()

print("\n=== VALEURS MANQUANTES ===")
print(work[["year", "uai", "region", "study", "emp6", "va6", "emp12"]].isna().sum())

sample = work.dropna(subset=["year", "uai", "region", "study", "emp6", "va6", "emp12"]).copy()

print("\n=== ECHANTILLON PRINCIPAL ===")
print("Observations :", len(sample))
print("CFA distincts :", sample["uai"].nunique())
print("Années distinctes :", sample["year"].nunique())

years_per_cfa = sample.groupby("uai")["year"].nunique().reset_index()
years_per_cfa.columns = ["uai", "n_years"]

print("\n=== NOMBRE D'ANNEES OBSERVEES PAR CFA ===")
print(years_per_cfa["n_years"].value_counts().sort_index())

uai_3plus = years_per_cfa.loc[years_per_cfa["n_years"] >= 3, "uai"]
sample_3plus = sample[sample["uai"].isin(uai_3plus)].copy()

print("\n=== ECHANTILLON ROBUSTESSE (CFA >= 3 années) ===")
print("Observations :", len(sample_3plus))
print("CFA distincts :", sample_3plus["uai"].nunique())

def run_model(data, formula, model_name):
    model = smf.ols(formula=formula, data=data).fit(cov_type="HC1")
    print(f"\n=== {model_name} ===")
    print(model.summary())
    return model

m1_reg = run_model(
    sample,
    "emp6 ~ study + C(region) + C(year)",
    "MODELE 1 - EMPLOI 6 MOIS | FE REGION + ANNEE"
)

m2_reg = run_model(
    sample,
    "va6 ~ study + C(region) + C(year)",
    "MODELE 2 - VA EMPLOI 6 MOIS | FE REGION + ANNEE"
)

m3_reg = run_model(
    sample,
    "emp12 ~ study + C(region) + C(year)",
    "MODELE 3 - EMPLOI 12 MOIS | FE REGION + ANNEE"
)

m1_cfa = run_model(
    sample,
    "emp6 ~ study + C(uai) + C(year)",
    "MODELE 4 - EMPLOI 6 MOIS | FE CFA + ANNEE"
)

m2_cfa = run_model(
    sample,
    "va6 ~ study + C(uai) + C(year)",
    "MODELE 5 - VA EMPLOI 6 MOIS | FE CFA + ANNEE"
)

m3_cfa = run_model(
    sample,
    "emp12 ~ study + C(uai) + C(year)",
    "MODELE 6 - EMPLOI 12 MOIS | FE CFA + ANNEE"
)

m1_cfa_r = run_model(
    sample_3plus,
    "emp6 ~ study + C(uai) + C(year)",
    "ROBUSTESSE 1 - EMPLOI 6 MOIS | FE CFA + ANNEE | CFA >= 3 ans"
)

m2_cfa_r = run_model(
    sample_3plus,
    "va6 ~ study + C(uai) + C(year)",
    "ROBUSTESSE 2 - VA EMPLOI 6 MOIS | FE CFA + ANNEE | CFA >= 3 ans"
)

m3_cfa_r = run_model(
    sample_3plus,
    "emp12 ~ study + C(uai) + C(year)",
    "ROBUSTESSE 3 - EMPLOI 12 MOIS | FE CFA + ANNEE | CFA >= 3 ans"
)

def extract_result(model, depvar_label, spec_label, sample_label):
    coef = model.params.get("study", np.nan)
    se = model.bse.get("study", np.nan)
    pval = model.pvalues.get("study", np.nan)
    r2 = model.rsquared
    nobs = int(model.nobs)

    if pd.isna(pval):
        signif = ""
    elif pval < 0.01:
        signif = "***"
    elif pval < 0.05:
        signif = "**"
    elif pval < 0.10:
        signif = "*"
    else:
        signif = ""

    return {
        "Variable dépendante": depvar_label,
        "Spécification": spec_label,
        "Échantillon": sample_label,
        "Coef. study": coef,
        "Std. Err.": se,
        "p-value": pval,
        "Signif.": signif,
        "R2": r2,
        "Observations": nobs
    }

results = pd.DataFrame([
    extract_result(m1_reg, "Taux emploi 6 mois", "FE région + année", "Principal"),
    extract_result(m1_cfa, "Taux emploi 6 mois", "FE CFA + année", "Principal"),
    extract_result(m1_cfa_r, "Taux emploi 6 mois", "FE CFA + année", "CFA >= 3 ans"),

    extract_result(m2_reg, "VA emploi 6 mois", "FE région + année", "Principal"),
    extract_result(m2_cfa, "VA emploi 6 mois", "FE CFA + année", "Principal"),
    extract_result(m2_cfa_r, "VA emploi 6 mois", "FE CFA + année", "CFA >= 3 ans"),

    extract_result(m3_reg, "Taux emploi 12 mois", "FE région + année", "Principal"),
    extract_result(m3_cfa, "Taux emploi 12 mois", "FE CFA + année", "Principal"),
    extract_result(m3_cfa_r, "Taux emploi 12 mois", "FE CFA + année", "CFA >= 3 ans"),
])

print("\n=== TABLEAU SYNTHETIQUE DES RESULTATS ===")
print(results)

results.to_csv(os.path.join(output_dir, "tableau_resultats_rehausses.csv"), index=False)

def fmt_coef_with_stars(coef, pval):
    if pd.isna(coef):
        return ""
    if pval < 0.01:
        stars = "***"
    elif pval < 0.05:
        stars = "**"
    elif pval < 0.10:
        stars = "*"
    else:
        stars = ""
    return f"{coef:.4f}{stars}"

main_models = [
    ("(1)", m1_reg, "Oui", "Non", "Oui"),
    ("(2)", m1_cfa, "Non", "Oui", "Oui"),
    ("(3)", m2_reg, "Oui", "Non", "Oui"),
    ("(4)", m2_cfa, "Non", "Oui", "Oui"),
    ("(5)", m3_reg, "Oui", "Non", "Oui"),
    ("(6)", m3_cfa, "Non", "Oui", "Oui"),
]

coef_line = ["Taux poursuite études"]
se_line = [""]
region_line = ["Effets fixes région"]
cfa_line = ["Effets fixes CFA"]
year_line = ["Effets fixes année"]
obs_line = ["Observations"]
r2_line = ["$R^2$"]

for _, model, reg_fe, cfa_fe, year_fe in main_models:
    coef_line.append(fmt_coef_with_stars(model.params["study"], model.pvalues["study"]))
    se_line.append(f"({model.bse['study']:.4f})")
    region_line.append(reg_fe)
    cfa_line.append(cfa_fe)
    year_line.append(year_fe)
    obs_line.append(f"{int(model.nobs):,}".replace(",", "\\,"))
    r2_line.append(f"{model.rsquared:.3f}")

latex_table = rf"""
\begin{{table}}[H]
\centering
\caption{{Association entre le taux de poursuite d'études et les indicateurs d'insertion des CFA}}
\label{{tab:regressions_rehaussees}}
\begin{{tabular}}{{lcccccc}}
\toprule
& \textbf{{(1)}} & \textbf{{(2)}} & \textbf{{(3)}} & \textbf{{(4)}} & \textbf{{(5)}} & \textbf{{(6)}} \\
& \textbf{{Emp. 6m}} & \textbf{{Emp. 6m}} & \textbf{{VA 6m}} & \textbf{{VA 6m}} & \textbf{{Emp. 12m}} & \textbf{{Emp. 12m}} \\
\midrule
{' & '.join(coef_line)} \\
{' & '.join(se_line)} \\
\addlinespace
{' & '.join(region_line)} \\
{' & '.join(cfa_line)} \\
{' & '.join(year_line)} \\
\addlinespace
{' & '.join(obs_line)} \\
{' & '.join(r2_line)} \\
\bottomrule
\end{{tabular}}

\vspace{{0.2cm}}
\begin{{minipage}}{{0.95\textwidth}}
\footnotesize
\textit{{Notes :}} erreurs standards robustes à l'hétéroscédasticité entre parenthèses.
Les colonnes (1), (3) et (5) incluent des effets fixes de région et d'année.
Les colonnes (2), (4) et (6) incluent des effets fixes CFA et d'année.
*** $p<0.01$, ** $p<0.05$, * $p<0.1$.
\end{{minipage}}
\end{{table}}
"""

with open(os.path.join(output_dir, "tableau_regressions_rehaussees.tex"), "w", encoding="utf-8") as f:
    f.write(latex_table)

print("\n=== TABLE LATEX ===")
print(latex_table)

plot_df = results.copy()
plot_df["label_plot"] = (
    plot_df["Variable dépendante"] + "\n" +
    plot_df["Spécification"] + "\n" +
    plot_df["Échantillon"]
)

plot_df = plot_df.reset_index(drop=True)

plt.figure(figsize=(10, 8))
y_pos = np.arange(len(plot_df))

plt.errorbar(
    x=plot_df["Coef. study"],
    y=y_pos,
    xerr=1.96 * plot_df["Std. Err."],
    fmt="o"
)

plt.axvline(x=0, linestyle="--")
plt.yticks(y_pos, plot_df["label_plot"])
plt.xlabel("Coefficient estimé du taux de poursuite d'études")
plt.ylabel("")
plt.title("Comparaison des coefficients estimés selon les spécifications")
plt.tight_layout()

coef_plot_path = os.path.join(output_dir, "graphe_coefficients_rehausses.png")
plt.savefig(coef_plot_path, dpi=300, bbox_inches="tight")
plt.show()

with open(os.path.join(output_dir, "resume_coefficients.txt"), "w", encoding="utf-8") as f:
    for _, row in results.iterrows():
        f.write(
            f"{row['Variable dépendante']} | {row['Spécification']} | {row['Échantillon']} : "
            f"coef={row['Coef. study']:.4f}, se={row['Std. Err.']:.4f}, "
            f"p={row['p-value']:.4f}, R2={row['R2']:.4f}, N={row['Observations']}\n"
        )

print("\nFichiers exportés dans :", output_dir)
print(os.listdir(output_dir))
