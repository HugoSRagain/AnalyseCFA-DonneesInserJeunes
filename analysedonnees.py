import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path

file_path = "/content/fr-en-inserjeunes-cfa.csv"
output_dir = "/content/analyse_inserjeunes_rehaussee"
graph_dir = Path(output_dir) / "graphiques"

os.makedirs(output_dir, exist_ok=True)
graph_dir.mkdir(exist_ok=True)

separators_to_try = [";", ",", "\t"]

df = None
chosen_sep = None

for sep in separators_to_try:
    try:
        tmp = pd.read_csv(file_path, sep=sep, encoding="utf-8-sig")
        print(f"Test separateur '{sep}' : shape = {tmp.shape}")
        if tmp.shape[1] >= 8:
            df = tmp.copy()
            chosen_sep = sep
            break
    except Exception as e:
        print(f"Echec avec separateur '{sep}' : {e}")

if df is None:
    raise ValueError("Impossible de lire correctement le fichier avec les separateurs testes.")

print("\nSeparateur retenu :", repr(chosen_sep))
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

print("\nToutes les colonnes utiles sont presentes.")

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
print("Annees distinctes :", sample["year"].nunique())

years_per_cfa = sample.groupby("uai")["year"].nunique().reset_index()
years_per_cfa.columns = ["uai", "n_years"]

print("\n=== NOMBRE D'ANNEES OBSERVEES PAR CFA ===")
print(years_per_cfa["n_years"].value_counts().sort_index())

uai_3plus = years_per_cfa.loc[years_per_cfa["n_years"] >= 3, "uai"]
sample_3plus = sample[sample["uai"].isin(uai_3plus)].copy()

print("\n=== ECHANTILLON ROBUSTESSE (CFA >= 3 annees) ===")
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
        "Variable dependante": depvar_label,
        "Specification": spec_label,
        "Echantillon": sample_label,
        "Coef_study": coef,
        "Std_Err": se,
        "p_value": pval,
        "Signif": signif,
        "R2": r2,
        "Observations": nobs
    }

results = pd.DataFrame([
    extract_result(m1_reg, "Taux emploi 6 mois", "FE region + annee", "Principal"),
    extract_result(m1_cfa, "Taux emploi 6 mois", "FE CFA + annee", "Principal"),
    extract_result(m1_cfa_r, "Taux emploi 6 mois", "FE CFA + annee", "CFA >= 3 ans"),

    extract_result(m2_reg, "VA emploi 6 mois", "FE region + annee", "Principal"),
    extract_result(m2_cfa, "VA emploi 6 mois", "FE CFA + annee", "Principal"),
    extract_result(m2_cfa_r, "VA emploi 6 mois", "FE CFA + annee", "CFA >= 3 ans"),

    extract_result(m3_reg, "Taux emploi 12 mois", "FE region + annee", "Principal"),
    extract_result(m3_cfa, "Taux emploi 12 mois", "FE CFA + annee", "Principal"),
    extract_result(m3_cfa_r, "Taux emploi 12 mois", "FE CFA + annee", "CFA >= 3 ans"),
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

coef_line = ["Taux poursuite etudes"]
se_line = [""]
region_line = ["Effets fixes region"]
cfa_line = ["Effets fixes CFA"]
year_line = ["Effets fixes annee"]
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
\caption{{Association entre le taux de poursuite d'etudes et les indicateurs d'insertion des CFA}}
\label{{tab:regressions_rehaussees}}
\resizebox{{\textwidth}}{{!}}{{%
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
\end{{tabular}}%
}}

\vspace{{0.2cm}}
\begin{{minipage}}{{0.95\textwidth}}
\footnotesize
\textit{{Notes :}} erreurs standards robustes a l'heteroscedasticite entre parentheses.
Les colonnes (1), (3) et (5) incluent des effets fixes de region et d'annee.
Les colonnes (2), (4) et (6) incluent des effets fixes CFA et d'annee.
*** $p<0.01$, ** $p<0.05$, * $p<0.1$.
\end{{minipage}}
\end{{table}}
"""

with open(os.path.join(output_dir, "tableau_regressions_rehaussees.tex"), "w", encoding="utf-8") as f:
    f.write(latex_table)

year_order = [
    "cumul 2018-2019",
    "cumul 2019-2020",
    "cumul 2020-2021",
    "cumul 2021-2022",
    "cumul 2022-2023",
    "cumul 2023-2024"
]

plt.figure(figsize=(8, 5))
plt.hist(sample["emp6"].dropna(), bins=30, edgecolor="black")
plt.xlabel("Taux d'emploi a 6 mois")
plt.ylabel("Frequence")
plt.title("Distribution du taux d'emploi a 6 mois des CFA")
plt.tight_layout()
plt.savefig(graph_dir / "graphique_1_distribution_emploi6.png", dpi=300)
plt.show()

x = sample["study"].dropna()
y = sample.loc[x.index, "emp6"]

plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.25)

coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, poly1d_fn(x_line), linewidth=2)

plt.xlabel("Taux de poursuite d'etudes")
plt.ylabel("Taux d'emploi a 6 mois")
plt.title("Poursuite d'etudes et insertion a 6 mois")
plt.tight_layout()
plt.savefig(graph_dir / "graphique_2_scatter_poursuite_emploi6.png", dpi=300)
plt.show()

year_stats = (
    sample
    .groupby("year")[["emp6", "emp12", "study"]]
    .mean()
    .reset_index()
)

plt.figure(figsize=(9, 5))
plt.plot(year_stats["year"], year_stats["emp6"], marker="o", label="Taux emploi 6 mois")
plt.plot(year_stats["year"], year_stats["emp12"], marker="o", label="Taux emploi 12 mois")
plt.plot(year_stats["year"], year_stats["study"], marker="o", label="Taux poursuite etudes")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Pourcentage moyen")
plt.title("Evolution moyenne des indicateurs par annee")
plt.legend()
plt.tight_layout()
plt.savefig(graph_dir / "graphique_3_evolution_annee.png", dpi=300)
plt.show()

region_stats = (
    sample
    .groupby("region")["emp6"]
    .mean()
    .sort_values()
    .reset_index()
)

plt.figure(figsize=(10, 8))
plt.barh(region_stats["region"], region_stats["emp6"])
plt.xlabel("Taux d'emploi a 6 mois moyen")
plt.ylabel("Region")
plt.title("Ecarts regionaux du taux d'emploi a 6 mois")
plt.tight_layout()
plt.savefig(graph_dir / "graphique_4_regions_emploi6.png", dpi=300)
plt.show()

def make_boxplot_by_year(dataframe, variable, ylabel, title, filename):
    box_data = [
        dataframe.loc[dataframe["year"] == y, variable].dropna()
        for y in year_order
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(box_data, tick_labels=year_order)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(graph_dir / filename, dpi=300)
    plt.show()

make_boxplot_by_year(
    sample,
    "emp6",
    "Taux d'emploi a 6 mois",
    "Distribution du taux d'emploi a 6 mois par annee",
    "boxplot_emploi6_par_annee.png"
)

make_boxplot_by_year(
    sample,
    "emp12",
    "Taux d'emploi a 12 mois",
    "Distribution du taux d'emploi a 12 mois par annee",
    "boxplot_emploi12_par_annee.png"
)

make_boxplot_by_year(
    sample,
    "study",
    "Taux de poursuite d'etudes",
    "Distribution du taux de poursuite d'etudes par annee",
    "boxplot_poursuite_par_annee.png"
)

make_boxplot_by_year(
    sample,
    "va6",
    "Valeur ajoutee d'emploi a 6 mois",
    "Distribution de la valeur ajoutee d'emploi a 6 mois par annee",
    "boxplot_va6_par_annee.png"
)

plot_df = results.copy()
plot_df["label_plot"] = (
    plot_df["Variable dependante"] + "\n" +
    plot_df["Specification"] + "\n" +
    plot_df["Echantillon"]
)

plot_df = plot_df.reset_index(drop=True)

plt.figure(figsize=(10, 8))
y_pos = np.arange(len(plot_df))

plt.errorbar(
    x=plot_df["Coef_study"],
    y=y_pos,
    xerr=1.96 * plot_df["Std_Err"],
    fmt="o"
)

plt.axvline(x=0, linestyle="--")
plt.yticks(y_pos, plot_df["label_plot"])
plt.xlabel("Coefficient estime du taux de poursuite d'etudes")
plt.ylabel("")
plt.title("Comparaison des coefficients estimes selon les specifications")
plt.tight_layout()
plt.savefig(graph_dir / "graphe_coefficients_rehausses.png", dpi=300, bbox_inches="tight")
plt.show()

with open(os.path.join(output_dir, "resume_coefficients.txt"), "w", encoding="utf-8") as f:
    for _, row in results.iterrows():
        f.write(
            f"{row['Variable dependante']} | {row['Specification']} | {row['Echantillon']} : "
            f"coef={row['Coef_study']:.4f}, se={row['Std_Err']:.4f}, "
            f"p={row['p_value']:.4f}, R2={row['R2']:.4f}, N={row['Observations']}\n"
        )

print("\nFichiers exportes dans :", output_dir)
print(os.listdir(output_dir))
\end{lstlisting}
