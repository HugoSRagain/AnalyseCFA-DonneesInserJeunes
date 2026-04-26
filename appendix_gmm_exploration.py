!pip -q install pydynpd

import pandas as pd
import numpy as np
import re
import io
from contextlib import redirect_stdout
from pathlib import Path
import matplotlib.pyplot as plt

from pydynpd import regression

file_path = "/content/fr-en-inserjeunes-cfa.csv"
output_dir = Path("/content/analyse_inserjeunes_endogeneite_gmm")
output_dir.mkdir(parents=True, exist_ok=True)

separators_to_try = [";", ",", "\t"]

df = None
chosen_sep = None

for sep in separators_to_try:
    try:
        tmp = pd.read_csv(file_path, sep=sep, encoding="utf-8-sig")
        print(f"Test séparateur {repr(sep)} : shape = {tmp.shape}")
        if tmp.shape[1] >= 8:
            df = tmp.copy()
            chosen_sep = sep
            break
    except Exception as e:
        print(f"Echec avec séparateur {repr(sep)} : {e}")

if df is None:
    raise ValueError("Impossible de lire correctement le fichier avec les séparateurs testés.")

print("\nSéparateur retenu :", repr(chosen_sep))
print("Dimensions brutes :", df.shape)

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

work = df.copy().rename(columns={
    "Année": "year",
    "UAI": "uai",
    "Libellé": "label",
    "Région": "region",
    "Taux poursuite études": "study",
    "Taux emploi 6 mois": "emp6",
    "VA emploi 6 mois": "va6",
    "Taux emploi 12 mois": "emp12"
})

for col in ["study", "emp6", "va6", "emp12"]:
    work[col] = pd.to_numeric(work[col], errors="coerce")

for col in ["year", "uai", "label", "region"]:
    work[col] = work[col].astype(str).str.strip()

year_order = [
    "cumul 2018-2019",
    "cumul 2019-2020",
    "cumul 2020-2021",
    "cumul 2021-2022",
    "cumul 2022-2023",
    "cumul 2023-2024"
]
year_rank_map = {y: i + 1 for i, y in enumerate(year_order)}

work = work[work["year"].isin(year_order)].copy()
work["year_rank"] = work["year"].map(year_rank_map)

sample = work.dropna(subset=["year", "uai", "study", "emp6", "va6", "emp12"]).copy()
sample["study10"] = sample["study"] / 10.0  # +10 points de poursuite d'études

years_per_cfa = sample.groupby("uai")["year_rank"].nunique().reset_index()
years_per_cfa.columns = ["uai", "n_years"]

uai_gmm = years_per_cfa.loc[years_per_cfa["n_years"] >= 4, "uai"]
gmm_sample = sample[sample["uai"].isin(uai_gmm)].copy()
gmm_sample = gmm_sample.sort_values(["uai", "year_rank"]).reset_index(drop=True)

print("\n=== ECHANTILLON GMM ===")
print("Observations :", len(gmm_sample))
print("CFA distincts :", gmm_sample["uai"].nunique())
print("Années distinctes :", gmm_sample["year_rank"].nunique())
print("\nRépartition du nombre d'années par CFA :")
print(years_per_cfa["n_years"].value_counts().sort_index())

def star(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""

def parse_console_output(txt):
    def extract(pattern):
        m = re.search(pattern, txt, flags=re.IGNORECASE)
        return float(m.group(1)) if m else np.nan

    out = {
        "n_obs": extract(r"Number of obs\s*=\s*([0-9]+(?:\.[0-9]+)?)"),
        "n_groups": extract(r"Number of groups\s*=\s*([0-9]+(?:\.[0-9]+)?)"),
        "n_instruments": extract(r"Number of instruments\s*=\s*([0-9]+(?:\.[0-9]+)?)"),
        "hansen_p": extract(r"Hansen test of overid\..*?Prob\s*>\s*Chi2\s*=\s*([0-9.]+)"),
        "ar1_p": extract(r"AR\(1\).*?Pr\s*>\s*z\s*=\s*([0-9.]+)"),
        "ar2_p": extract(r"AR\(2\).*?Pr\s*>\s*z\s*=\s*([0-9.]+)")
    }
    return out

def extract_regression_row(reg_table, variable_name):
    row = reg_table.loc[reg_table["variable"] == variable_name]
    if row.empty:
        return {"coef": np.nan, "se": np.nan, "p": np.nan}
    row = row.iloc[0]
    return {
        "coef": float(row["coefficient"]),
        "se": float(row["std_err"]),
        "p": float(row["p_value"])
    }

def run_gmm_model(data, depvar, estimator_type="system"):
    if estimator_type not in ["system", "difference"]:
        raise ValueError("estimator_type doit être 'system' ou 'difference'.")

    options = "collapse timedumm"
    if estimator_type == "difference":
        options += " nolevel"

    command = f"{depvar} L1.{depvar} study10 | gmm({depvar}, 2:3) gmm(study10, 2:3) | {options}"

    print("\n" + "=" * 100)
    print(f"{estimator_type.upper()} GMM | Variable dépendante = {depvar}")
    print("Commande pydynpd :", command)
    print("=" * 100)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = regression.abond(command, data, ["uai", "year_rank"])
    console_txt = buffer.getvalue()

    print(console_txt)

    model0 = result.models[0]
    reg_table = model0.regression_table.copy()

    diag = parse_console_output(console_txt)

    lag_row = extract_regression_row(reg_table, f"L1.{depvar}")
    study_row = extract_regression_row(reg_table, "study10")

    summary_row = {
        "Variable dépendante": depvar,
        "Estimateur": "System GMM" if estimator_type == "system" else "Difference GMM",
        "Coef. L1.Y": lag_row["coef"],
        "SE L1.Y": lag_row["se"],
        "p-value L1.Y": lag_row["p"],
        "Coef. study10": study_row["coef"],
        "SE study10": study_row["se"],
        "p-value study10": study_row["p"],
        "Signif. study10": star(study_row["p"]),
        "Observations": int(diag["n_obs"]) if not np.isnan(diag["n_obs"]) else np.nan,
        "Groupes CFA": int(diag["n_groups"]) if not np.isnan(diag["n_groups"]) else np.nan,
        "Instruments": int(diag["n_instruments"]) if not np.isnan(diag["n_instruments"]) else np.nan,
        "Hansen p-value": diag["hansen_p"],
        "AR(1) p-value": diag["ar1_p"],
        "AR(2) p-value": diag["ar2_p"],
        "Commande": command,
        "Sortie console": console_txt
    }

    return result, reg_table, summary_row

models_to_run = [
    ("emp6", "system"),
    ("emp6", "difference"),
    ("va6", "system"),
    ("va6", "difference"),
    ("emp12", "system"),
    ("emp12", "difference"),
]

all_results = []
reg_tables = {}

for depvar, est_type in models_to_run:
    try:
        result_obj, reg_table, summary_row = run_gmm_model(
            data=gmm_sample,
            depvar=depvar,
            estimator_type=est_type
        )
        all_results.append(summary_row)
        reg_tables[f"{depvar}_{est_type}"] = reg_table

    except Exception as e:
        print(f"\nECHEC sur {depvar} - {est_type} : {e}")
        all_results.append({
            "Variable dépendante": depvar,
            "Estimateur": "System GMM" if est_type == "system" else "Difference GMM",
            "Coef. L1.Y": np.nan,
            "SE L1.Y": np.nan,
            "p-value L1.Y": np.nan,
            "Coef. study10": np.nan,
            "SE study10": np.nan,
            "p-value study10": np.nan,
            "Signif. study10": "",
            "Observations": np.nan,
            "Groupes CFA": np.nan,
            "Instruments": np.nan,
            "Hansen p-value": np.nan,
            "AR(1) p-value": np.nan,
            "AR(2) p-value": np.nan,
            "Commande": np.nan,
            "Sortie console": str(e)
        })

results_df = pd.DataFrame(all_results)

print("\n=== TABLEAU SYNTHETIQUE GMM ===")
print(results_df[[
    "Variable dépendante", "Estimateur",
    "Coef. L1.Y", "p-value L1.Y",
    "Coef. study10", "SE study10", "p-value study10", "Signif. study10",
    "Observations", "Groupes CFA", "Instruments",
    "Hansen p-value", "AR(1) p-value", "AR(2) p-value"
]])

results_df.to_csv(output_dir / "resultats_gmm_dynamiques.csv", index=False)

for name, tab in reg_tables.items():
    tab.to_csv(output_dir / f"regtable_{name}.csv", index=False)

def fmt_num(x, nd=4):
    if pd.isna(x):
        return ""
    return f"{x:.{nd}f}"

def fmt_p(x):
    if pd.isna(x):
        return ""
    return f"{x:.3f}"

def get_row(dep, est):
    sub = results_df[(results_df["Variable dépendante"] == dep) & (results_df["Estimateur"] == est)]
    if sub.empty:
        return None
    return sub.iloc[0]

row_emp6_sys = get_row("emp6", "System GMM")
row_emp6_diff = get_row("emp6", "Difference GMM")
row_va6_sys = get_row("va6", "System GMM")
row_va6_diff = get_row("va6", "Difference GMM")
row_emp12_sys = get_row("emp12", "System GMM")
row_emp12_diff = get_row("emp12", "Difference GMM")

latex_table = rf"""
\begin{{table}}[H]
\centering
\caption{{Panel dynamique GMM : traitement de l'endogénéité potentielle de la poursuite d'études}}
\label{{tab:gmm_endogeneite}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccc}}
\toprule
& \multicolumn{{2}}{{c}}{{\textbf{{Emp. 6m}}}}
& \multicolumn{{2}}{{c}}{{\textbf{{VA 6m}}}}
& \multicolumn{{2}}{{c}}{{\textbf{{Emp. 12m}}}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}}
& \textbf{{System}} & \textbf{{Diff.}} & \textbf{{System}} & \textbf{{Diff.}} & \textbf{{System}} & \textbf{{Diff.}} \\
\midrule
$L1.Y$
& {fmt_num(row_emp6_sys["Coef. L1.Y"])}{star(row_emp6_sys["p-value L1.Y"])}
& {fmt_num(row_emp6_diff["Coef. L1.Y"])}{star(row_emp6_diff["p-value L1.Y"])}
& {fmt_num(row_va6_sys["Coef. L1.Y"])}{star(row_va6_sys["p-value L1.Y"])}
& {fmt_num(row_va6_diff["Coef. L1.Y"])}{star(row_va6_diff["p-value L1.Y"])}
& {fmt_num(row_emp12_sys["Coef. L1.Y"])}{star(row_emp12_sys["p-value L1.Y"])}
& {fmt_num(row_emp12_diff["Coef. L1.Y"])}{star(row_emp12_diff["p-value L1.Y"])} \\
& ({fmt_num(row_emp6_sys["SE L1.Y"])})
& ({fmt_num(row_emp6_diff["SE L1.Y"])})
& ({fmt_num(row_va6_sys["SE L1.Y"])})
& ({fmt_num(row_va6_diff["SE L1.Y"])})
& ({fmt_num(row_emp12_sys["SE L1.Y"])})
& ({fmt_num(row_emp12_diff["SE L1.Y"])}) \\

Taux poursuite études / 10
& {fmt_num(row_emp6_sys["Coef. study10"])}{row_emp6_sys["Signif. study10"]}
& {fmt_num(row_emp6_diff["Coef. study10"])}{row_emp6_diff["Signif. study10"]}
& {fmt_num(row_va6_sys["Coef. study10"])}{row_va6_sys["Signif. study10"]}
& {fmt_num(row_va6_diff["Coef. study10"])}{row_va6_diff["Signif. study10"]}
& {fmt_num(row_emp12_sys["Coef. study10"])}{row_emp12_sys["Signif. study10"]}
& {fmt_num(row_emp12_diff["Coef. study10"])}{row_emp12_diff["Signif. study10"]} \\
& ({fmt_num(row_emp6_sys["SE study10"])})
& ({fmt_num(row_emp6_diff["SE study10"])})
& ({fmt_num(row_va6_sys["SE study10"])})
& ({fmt_num(row_va6_diff["SE study10"])})
& ({fmt_num(row_emp12_sys["SE study10"])})
& ({fmt_num(row_emp12_diff["SE study10"])}) \\

\addlinespace
Observations
& {int(row_emp6_sys["Observations"]) if pd.notna(row_emp6_sys["Observations"]) else ""}
& {int(row_emp6_diff["Observations"]) if pd.notna(row_emp6_diff["Observations"]) else ""}
& {int(row_va6_sys["Observations"]) if pd.notna(row_va6_sys["Observations"]) else ""}
& {int(row_va6_diff["Observations"]) if pd.notna(row_va6_diff["Observations"]) else ""}
& {int(row_emp12_sys["Observations"]) if pd.notna(row_emp12_sys["Observations"]) else ""}
& {int(row_emp12_diff["Observations"]) if pd.notna(row_emp12_diff["Observations"]) else ""} \\

CFA
& {int(row_emp6_sys["Groupes CFA"]) if pd.notna(row_emp6_sys["Groupes CFA"]) else ""}
& {int(row_emp6_diff["Groupes CFA"]) if pd.notna(row_emp6_diff["Groupes CFA"]) else ""}
& {int(row_va6_sys["Groupes CFA"]) if pd.notna(row_va6_sys["Groupes CFA"]) else ""}
& {int(row_va6_diff["Groupes CFA"]) if pd.notna(row_va6_diff["Groupes CFA"]) else ""}
& {int(row_emp12_sys["Groupes CFA"]) if pd.notna(row_emp12_sys["Groupes CFA"]) else ""}
& {int(row_emp12_diff["Groupes CFA"]) if pd.notna(row_emp12_diff["Groupes CFA"]) else ""} \\

Instruments
& {int(row_emp6_sys["Instruments"]) if pd.notna(row_emp6_sys["Instruments"]) else ""}
& {int(row_emp6_diff["Instruments"]) if pd.notna(row_emp6_diff["Instruments"]) else ""}
& {int(row_va6_sys["Instruments"]) if pd.notna(row_va6_sys["Instruments"]) else ""}
& {int(row_va6_diff["Instruments"]) if pd.notna(row_va6_diff["Instruments"]) else ""}
& {int(row_emp12_sys["Instruments"]) if pd.notna(row_emp12_sys["Instruments"]) else ""}
& {int(row_emp12_diff["Instruments"]) if pd.notna(row_emp12_diff["Instruments"]) else ""} \\

Hansen $p$-value
& {fmt_p(row_emp6_sys["Hansen p-value"])}
& {fmt_p(row_emp6_diff["Hansen p-value"])}
& {fmt_p(row_va6_sys["Hansen p-value"])}
& {fmt_p(row_va6_diff["Hansen p-value"])}
& {fmt_p(row_emp12_sys["Hansen p-value"])}
& {fmt_p(row_emp12_diff["Hansen p-value"])} \\

AR(1) $p$-value
& {fmt_p(row_emp6_sys["AR(1) p-value"])}
& {fmt_p(row_emp6_diff["AR(1) p-value"])}
& {fmt_p(row_va6_sys["AR(1) p-value"])}
& {fmt_p(row_va6_diff["AR(1) p-value"])}
& {fmt_p(row_emp12_sys["AR(1) p-value"])}
& {fmt_p(row_emp12_diff["AR(1) p-value"])} \\

AR(2) $p$-value
& {fmt_p(row_emp6_sys["AR(2) p-value"])}
& {fmt_p(row_emp6_diff["AR(2) p-value"])}
& {fmt_p(row_va6_sys["AR(2) p-value"])}
& {fmt_p(row_va6_diff["AR(2) p-value"])}
& {fmt_p(row_emp12_sys["AR(2) p-value"])}
& {fmt_p(row_emp12_diff["AR(2) p-value"])} \\
\bottomrule
\end{{tabular}}%
}}

\vspace{{0.2cm}}
\begin{{minipage}}{{0.95\textwidth}}
\footnotesize
\textit{{Notes :}} estimations en panel dynamique GMM. Les colonnes \textit{{System}} correspondent aux estimations system GMM ; les colonnes \textit{{Diff.}} aux estimations difference GMM. La variable \textit{{Taux poursuite études / 10}} correspond au taux de poursuite d'études divisé par 10 ; le coefficient se lit donc pour une hausse de 10 points de pourcentage. Les instruments internes sont limités aux lags 2 à 3 et l'option \textit{{collapse}} est utilisée pour limiter la prolifération instrumentale. Les effets annuels sont inclus via des variables muettes temporelles. *** $p<0.01$, ** $p<0.05$, * $p<0.1$.
\end{{minipage}}
\end{{table}}
"""

with open(output_dir / "table_gmm_endogeneite.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print("\n=== TABLE LATEX GMM ===\n")
print(latex_table)

plot_df = results_df.copy()
plot_df["label"] = plot_df["Variable dépendante"] + " — " + plot_df["Estimateur"]

plt.figure(figsize=(10, 6))
y_pos = np.arange(len(plot_df))

plt.errorbar(
    plot_df["Coef. study10"],
    y_pos,
    xerr=1.96 * plot_df["SE study10"],
    fmt="o",
    capsize=4
)

plt.axvline(x=0, linestyle="--")
plt.yticks(y_pos, plot_df["label"])
plt.xlabel("Coefficient estimé du taux de poursuite d'études / 10")
plt.ylabel("")
plt.title("Panel dynamique GMM : coefficient de la poursuite d'études")
plt.tight_layout()
plt.savefig(output_dir / "coef_gmm_study10.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n=== FICHIERS CREES ===")
for p in sorted(output_dir.iterdir()):
    print(p.name)
