import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

file_path = "/content/fr-en-inserjeunes-cfa.csv"

DEP_VAR   = "Taux emploi 6 mois"
ENDOG_VAR = "Taux poursuite études"
UAI_COL   = "UAI"
YEAR_COL  = "Année"

df = pd.read_csv(file_path, sep=";", low_memory=False, encoding="utf-8-sig")

cols_needed = [UAI_COL, YEAR_COL, DEP_VAR, ENDOG_VAR, "Région", "VA emploi 6 mois", "Taux emploi 12 mois"]
df_clean = df[cols_needed].dropna()

for c in [DEP_VAR, ENDOG_VAR, "VA emploi 6 mois", "Taux emploi 12 mois"]:
    df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
df_clean = df_clean.dropna().reset_index(drop=True)

print(f"Échantillon : {len(df_clean)} obs, {df_clean[UAI_COL].nunique()} CFA")

year_dummies = pd.get_dummies(df_clean[YEAR_COL], prefix="annee", drop_first=True).astype(float)
df_clean = pd.concat([df_clean, year_dummies], axis=1)
year_cols = list(year_dummies.columns)

def within_demean(df, group_col, cols):
    df = df.copy()
    means = df.groupby(group_col)[cols].transform("mean")
    for c in cols:
        df[f"{c}_dm"] = df[c] - means[c]
    return df

demean_cols = [DEP_VAR, ENDOG_VAR] + year_cols
df_clean = within_demean(df_clean, UAI_COL, demean_cols)

dm_dep   = f"{DEP_VAR}_dm"
dm_endog = f"{ENDOG_VAR}_dm"
dm_years = [f"{c}_dm" for c in year_cols]

X_fs_prelim  = sm.add_constant(df_clean[dm_years])
ols_prelim   = sm.OLS(df_clean[dm_endog], X_fs_prelim).fit()
residuals_fs = ols_prelim.resid

bp_stat, bp_pval, _, _ = het_breuschpagan(residuals_fs, X_fs_prelim)
print(f"\n── Test de Breusch-Pagan ────────────────────────────────")
print(f"Statistique : {bp_stat:.3f}  |  p-valeur : {bp_pval:.4f}")
if bp_pval < 0.05:
    print("→ Hétéroscédasticité détectée : condition Lewbel satisfaite ✓")
else:
    print("→ Hétéroscédasticité faible : Lewbel moins crédible ✗")

exog_centered = df_clean[dm_years].copy()
for c in dm_years:
    exog_centered[c] = exog_centered[c] - exog_centered[c].mean()

lewbel_all = pd.DataFrame(index=df_clean.index)
for c in dm_years:
    lewbel_all[f"lew_{c}"] = residuals_fs.values * exog_centered[c].values

residuals_centered = residuals_fs - residuals_fs.mean()
lewbel_all["lew_resid_sq"] = residuals_centered.values ** 2

all_lew_cols = list(lewbel_all.columns)
df_clean = pd.concat([df_clean, lewbel_all], axis=1)

fs_scores = {}
for col in all_lew_cols:
    X_single = sm.add_constant(df_clean[dm_years + [col]])
    m = sm.OLS(df_clean[dm_endog], X_single).fit()
    fs_scores[col] = abs(float(m.tvalues[col]))

top2 = sorted(fs_scores, key=fs_scores.get, reverse=True)[:2]
lew_cols = top2
print(f"\n── Instruments retenus (top 2) : {lew_cols}")
print(f"   t-stats : {[round(fs_scores[c], 3) for c in lew_cols]}")

X_fs        = sm.add_constant(df_clean[dm_years + lew_cols])
first_stage = sm.OLS(df_clean[dm_endog], X_fs).fit()

r_matrix    = np.zeros((len(lew_cols), X_fs.shape[1]))
lew_indices = [list(X_fs.columns).index(c) for c in lew_cols]
for i, idx in enumerate(lew_indices):
    r_matrix[i, idx] = 1.0

f_test = first_stage.f_test(r_matrix)
f_stat = float(f_test.fvalue)
f_pval = float(f_test.pvalue)

print(f"\n── Premier stade (force des instruments) ───────────────")
print(f"F-stat instruments Lewbel : {f_stat:.3f}  |  p-valeur : {f_pval:.4f}")
if f_stat > 10:
    print("→ Instruments suffisamment forts ✓")
elif f_stat > 5:
    print("→ Instruments modérément forts — interpréter avec prudence")
else:
    print("→ Instruments faibles (F < 5) — Lewbel non recommandé ✗")

df_clean["endog_hat"] = first_stage.predict(X_fs)

X_ss         = sm.add_constant(df_clean[["endog_hat"] + dm_years])
second_stage = sm.OLS(df_clean[dm_dep], X_ss).fit(cov_type="HC3")

coef_2sls = second_stage.params["endog_hat"]
se_2sls   = second_stage.bse["endog_hat"]

print(f"\n── Second stade 2SLS Lewbel ────────────────────────────")
print(f"Coefficient : {coef_2sls:.4f}  |  SE : {se_2sls:.4f}")
print(f"IC 95% : [{coef_2sls - 1.96*se_2sls:.4f} ; {coef_2sls + 1.96*se_2sls:.4f}]")

X_ols      = sm.add_constant(df_clean[[dm_endog] + dm_years])
ols_within = sm.OLS(df_clean[dm_dep], X_ols).fit(cov_type="HC3")
coef_ols   = ols_within.params[dm_endog]
se_ols     = ols_within.bse[dm_endog]

print(f"\n── Comparaison OLS within vs 2SLS Lewbel ───────────────")
print(f"{'Estimateur':<20} {'Coeff':>10} {'SE':>10} {'Significatif':>14}")
print(f"{'OLS within':<20} {coef_ols:>10.4f} {se_ols:>10.4f} {'oui *' if abs(coef_ols/se_ols)>1.96 else 'non':>14}")
print(f"{'2SLS Lewbel':<20} {coef_2sls:>10.4f} {se_2sls:>10.4f} {'oui *' if abs(coef_2sls/se_2sls)>1.96 else 'non':>14}")

diff = coef_2sls - coef_ols
print(f"\nDifférence 2SLS - OLS : {diff:.4f}")
if abs(diff) > 2 * max(se_ols, se_2sls):
    print("→ Divergence notable : biais d'endogénéité probable")
else:
    print("→ OLS et 2SLS proches : biais d'endogénéité probablement limité")

# Avec 2 instruments et 1 variable endogène : 1 degré de liberté
resid_2sls = df_clean[dm_dep].values - second_stage.predict(X_ss)
X_j        = sm.add_constant(df_clean[dm_years + lew_cols])
j_reg      = sm.OLS(resid_2sls, X_j).fit()
j_stat     = len(df_clean) * j_reg.rsquared
j_dof      = len(lew_cols) - 1  # = 1
j_pval     = 1 - stats.chi2.cdf(j_stat, j_dof)

print(f"\n── Test de suridentification (Hansen J) ────────────────")
print(f"J-stat : {j_stat:.3f}  |  ddl : {j_dof}  |  p-valeur : {j_pval:.4f}")
if j_pval > 0.05:
    print("→ Instruments valides ✓")
elif j_pval > 0.01:
    print("→ Validité limite — résultats à interpréter avec prudence")
else:
    print("→ Certains instruments potentiellement invalides ✗")

print("\n✓ Analyse Lewbel terminée.")
