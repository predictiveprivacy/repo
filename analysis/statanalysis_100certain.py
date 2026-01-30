# Note: this is the approach with GCP

from google.colab import auth
from google.oauth2 import service_account
from googleapiclient.discovery import build

auth.authenticate_user()

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

scope = ['https://www.googleapis.com/auth/spreadsheets']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)

client = gspread.authorize(creds)

spreadsheet_url = '' # insert sheet URL
sheet = client.open_by_url(spreadsheet_url)

worksheet = sheet.get_worksheet(0)
data = worksheet.get_all_records()
df_synthetic = pd.DataFrame(data)

df_synthetic.head()

# mount drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# Load your data file (adjust file name and method if it's an Excel file)
df = pd.read_csv('data/survey1_results.csv')

# Define the mapping from Likert-scale text to numbers
likert_mapping = {
    "Not at all harmful": 1,
    "Slightly harmful": 2,
    "Moderately harmful": 3,
    "Very harmful": 4,
    "Extremely harmful": 5
}

# List the columns that contain Likert-scale responses.
# Update this list to include all columns that need conversion.
likert_columns = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]

# Replace the text values with numeric values in those columns
for col in likert_columns:
    if col in df.columns:  # only replace if column exists in the dataframe
        df[col] = df[col].replace(likert_mapping)

import pandas as pd

# Assuming your DataFrame is called df and has columns:
# ["ProlificID", "SyntheticPersonID", "AttnCheck", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Condition"]

# 1) Group by SyntheticPersonID and compute the mean for Q1..Q8
df_avg = (
    df.groupby("SyntheticPersonID")[["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]]
      .mean()  # average across repeated rows for each SyntheticPersonID
      .reset_index()
)

# 2) (Optional) If you also want a single "avg_harm" column across Q1..Q8
df_avg["avg_harm"] = df_avg[["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]].mean(axis=1)

# Now df_avg contains one row per unique SyntheticPersonID
# with columns for Q1..Q8 averages, and optionally avg_harm.
print(df_avg.head())

import pandas as pd
import itertools
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

# 1) Group by SyntheticPersonID and Condition, averaging Q1..Q8
#    for repeated rows of the same SyntheticPersonID.
grouped_df = (
    df.groupby(["SyntheticPersonID", "Condition"])[["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]]
      .mean()
      .reset_index()
)

# 2) Identify the questions you want to test separately
questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]

# 3) Get the unique conditions
conditions = grouped_df["Condition"].unique()

# 4) Prepare a list to store results from each test
results = []

# 5) For each question, do pairwise t-tests across all condition pairs
for question in questions:
    for cond1, cond2 in itertools.combinations(conditions, 2):
        # Extract data for this question, for each condition
        group1 = grouped_df.loc[grouped_df["Condition"] == cond1, question].dropna()
        group2 = grouped_df.loc[grouped_df["Condition"] == cond2, question].dropna()

        # Perform Welch's t-test (equal_var=False)
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)

        # Store the results
        results.append((question, cond1, cond2, t_stat, p_val))

# 6) Convert results to a DataFrame
results_df = pd.DataFrame(
    results,
    columns=["Question", "Condition1", "Condition2", "t_stat", "p_value"]
)

# 7) Apply Bonferroni correction across all tests (all questions, all pairs)
adjusted = multipletests(results_df["p_value"], method="bonferroni")
results_df["p_adjusted"] = adjusted[1]

# 8) Print the final results
print("Separate t-test results for each question (with Bonferroni correction):")
print(results_df)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

# Dictionary mapping question codes to descriptive labels
custom_labels = {
    'Q1': 'Hackers/Cybercriminals',
    'Q2': 'Government',
    'Q3': 'Corporations',
    'Q4': 'Employer/Colleagues',
    'Q5': 'Family',
    'Q6': 'Close Friends',
    'Q7': 'Acquaintances',
    'Q8': 'Publicly Available'
}

# We'll assume `grouped_df` has one row per (SyntheticPersonID, Condition),
# with columns: ["SyntheticPersonID", "Condition", "Q1", "Q2", ..., "Q8"].

questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]
conditions = grouped_df["Condition"].unique()

for cond in conditions:
    # Filter data for this condition
    cond_data = grouped_df[grouped_df["Condition"] == cond]

    # Compute means and standard deviations for each question
    mean_vals = cond_data[questions].mean(axis=0)
    std_vals  = cond_data[questions].std(axis=0)

    # Number of SyntheticPersonIDs in this condition
    n = cond_data.shape[0]

    # Standard error for each question
    se_vals = std_vals / np.sqrt(n)  # If n <= 1, be mindful of dividing by zero

    # 95% CI using the t-distribution (two-tailed)
    if n > 1:
        t_multiplier = st.t.ppf(1 - 0.025, df=n-1)
        ci_vals = t_multiplier * se_vals
    else:
        # If there's only one data point in this condition, CI is not meaningful
        ci_vals = np.zeros_like(se_vals)

    # X positions for plotting
    x_positions = np.arange(len(questions))

    # Create the plot
    plt.figure(figsize=(10,6))
    plt.errorbar(
        x_positions,
        mean_vals,
        yerr=ci_vals,
        fmt='o',         # 'o' = circular markers
        capsize=5,       # error bar cap size
        color='blue',
        ecolor='black'   # color for error bar lines
    )

    # Use custom labels for the questions
    question_labels = [custom_labels[q] for q in questions]

    # Labeling
    plt.xticks(x_positions, question_labels, rotation=45, ha='right')
    plt.xlabel("Question")
    plt.ylabel("Mean Response")
    plt.title(f"Mean Response (95% CI) for Condition: {cond}")

    plt.tight_layout()
    plt.show()

"""In summary, based on your adjusted p‑values, significant differences exist between:

*   Financial vs. Health
*   Financial vs. Sensitive
*   Health vs. Sensitive
*   Control vs. Sensitive.


Financial vs. Control and Health vs. Control do not show significant differences after adjustment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

# We assume grouped_df has one row per (SyntheticPersonID, Condition),
# including averaged columns ["Q1", ..., "Q8"].

questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]
conditions = grouped_df["Condition"].unique()

for question in questions:
    # Prepare lists to store the mean and CI for each condition
    means = []
    ci_vals = []

    for cond in conditions:
        # Extract data for this question in the current condition
        cond_data = grouped_df.loc[grouped_df["Condition"] == cond, question].dropna()
        mean_val = cond_data.mean()
        std_val = cond_data.std()
        n = cond_data.shape[0]

        # Compute 95% CI using the t-distribution
        if n > 1:
            se_val = std_val / np.sqrt(n)
            t_multiplier = st.t.ppf(1 - 0.025, df=n-1)  # 95% CI
            ci = t_multiplier * se_val
        else:
            # If there's only one data point, the CI can't be computed meaningfully
            ci = 0.0

        means.append(mean_val)
        ci_vals.append(ci)

    # Create a figure for this question
    plt.figure(figsize=(8,6))
    x_positions = np.arange(len(conditions))

    # Plot means with error bars only (no bars)
    plt.errorbar(
        x_positions,
        means,
        yerr=ci_vals,
        fmt='o',        # 'o' = circular marker
        capsize=5,      # length of the error bar caps
        color='blue',
        ecolor='black'  # color for the error bars
    )

    # Labeling
    plt.xticks(x_positions, conditions, rotation=45, ha='right')
    plt.xlabel("Condition")
    plt.ylabel("Mean Response")
    plt.title(f"Mean {question} by Condition (95% CI)")

    plt.tight_layout()
    plt.show()

score_types = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]

import pandas as pd

# long-form summary: mean across participants × certainty
harm_summary = (
    grouped_df
    .groupby("Condition")[score_types]
    .mean()
    .T
    .sort_values(by="Health", ascending=False)
)
print(harm_summary)

import matplotlib.pyplot as plt

ax = harm_summary.plot(kind="bar", figsize=(10,6))
plt.ylabel("Mean harm rating (1–5)")
plt.title("Mean Harm Ratings by Condition and Question")
plt.legend(title="Condition")
plt.tight_layout()
plt.show()

import seaborn as sns

plt.figure(figsize=(8,6))
sns.heatmap(harm_summary, annot=True, cmap="Blues", fmt=".2f")
plt.title("Mean Harm Ratings (1–5)")
plt.ylabel("Question")
plt.xlabel("Condition")
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
from scipy import stats

results = []

for q in [f"Q{i}" for i in range(1,9)]:
    scores_c = grouped_df.loc[grouped_df["Condition"] == "Sensitive", q]
    scores_h = grouped_df.loc[grouped_df["Condition"] == "Health", q]

    # means and SDs
    mean_c, sd_c = scores_c.mean(), scores_c.std()
    mean_h, sd_h = scores_h.mean(), scores_h.std()

    # t-test
    t, p = stats.ttest_ind(scores_c, scores_h, equal_var=False)

    # mean diff (Control - Health)
    mean_diff = mean_c - mean_h

    # Cohen's d
    pooled_sd = np.sqrt(((sd_c**2) + (sd_h**2)) / 2)
    cohens_d = mean_diff / pooled_sd

    results.append([
        q,
        f"{mean_c:.2f} ({sd_c:.2f})",
        f"{mean_h:.2f} ({sd_h:.2f})",
        mean_diff,
        t,
        p,
        cohens_d
    ])

control_vs_health = pd.DataFrame(
    results,
    columns=["Question", "Sensitive Mean (SD)", "Health Mean (SD)", "MeanDiff", "t", "p", "Cohen's d"]
)

# Round for readability
control_vs_health = control_vs_health.round({"MeanDiff": 2, "t": 2, "p": 3, "Cohen's d": 2})

print(control_vs_health)

df_responses = pd.read_csv('data/survey1_results.csv')       # Adjust filename/path as needed
df_prolific  = pd.read_csv('data/prolific_export.csv')   # Note: this has to come from your own surveys
df_prolific.head()

df_responses = pd.read_csv('data/survey1_results.csv')       # Adjust filename/path as needed
df_prolific  = pd.read_csv('data/prolific_export.csv')   # Note: this has to come from your own surveys

# Define the mapping from Likert-scale text to numbers
likert_mapping = {
    "Not at all harmful": 1,
    "Slightly harmful": 2,
    "Moderately harmful": 3,
    "Very harmful": 4,
    "Extremely harmful": 5
}

# List the columns that contain Likert-scale responses.
# Update this list to include all columns that need conversion.
likert_columns = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]

# Replace the text values with numeric values in those columns
for col in likert_columns:
    if col in df.columns:  # only replace if column exists in the dataframe
        df_responses[col] = df_responses[col].replace(likert_mapping)

# --- Step 2: Merge on the participant id columns ---
# Change these column names as appropriate:
merged_df = pd.merge(df_responses, df_prolific,
                     left_on='ProlificID',
                     right_on='Participant id',
                     how='inner')

# Now merged_df contains columns from both datasets.
print("Merged DataFrame:")
print(merged_df.head())

# --- Step 3: Analyze responses across conditions (or by other columns) ---

# Example: How do the mean responses for Q1–Q8 vary by gender?
# (Assuming "gender" is a column in the prolific export)

questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]

import pandas as pd
import numpy as np
import scipy.stats as stats
import itertools

# Define the questions and the demographic columns you want to analyze
questions = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]

# List of demographic columns (categorical) from the Prolific export.
# Update this list based on the columns available in your prolific_export.
demographic_vars = ["Sex", "Ethnicity simplified", "Country of residence", "Language", "Employment status"]

# Prepare a list to store the t-test results
results = []

# Loop over each demographic variable
for demo in demographic_vars:
    # Get the unique groups in this demographic (drop missing values)
    groups = merged_df[demo].dropna().unique()

    # If there are less than 2 groups, skip this demographic.
    if len(groups) < 2:
        continue

    # For each question, perform pairwise comparisons between groups in this demographic.
    for question in questions:
        # Use itertools.combinations to get all unique pairwise comparisons.
        for group1, group2 in itertools.combinations(groups, 2):
            # Filter data for each group for the given question
            data1 = merged_df.loc[merged_df[demo] == group1, question].dropna()
            data2 = merged_df.loc[merged_df[demo] == group2, question].dropna()

            # Only perform the t-test if both groups have at least 2 observations
            if len(data1) >= 2 and len(data2) >= 2:
                t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
            else:
                t_stat, p_val = np.nan, np.nan  # Not enough data

            results.append((demo, question, group1, group2, t_stat, p_val))

# Convert the results list to a DataFrame for easier viewing
results_df = pd.DataFrame(results, columns=["Demographic", "Question", "Group1", "Group2", "t_stat", "p_value"])

# Optionally, adjust p-values for multiple comparisons using Bonferroni correction.
from statsmodels.stats.multitest import multipletests
adjusted = multipletests(results_df["p_value"].dropna(), method="bonferroni")
# Place adjusted p-values back into our results DataFrame.
# (We need to match the indices since multipletests was applied to non-NaN values.)
results_df.loc[results_df["p_value"].notna(), "p_adjusted"] = adjusted[1]

# Display the results
print("Pairwise t-test results for each question by demographic variable:")
print(results_df)
# get df to csv
results_df.to_csv('t-tests.csv', index=False)

"""## Supervised ML"""

from sklearn.preprocessing import LabelEncoder

feature_list = ["GENDER", "RACETHN", "EDUCCAT5", "DIVISION", "MARITAL_ACS",
                "CHILDRENCAT", "CITIZEN_REC", "BORN_ACS", "AGE_INT",
                "HIV_STAT", "PREG_STAT", "NumChronicIllness",
                "FAMINC5", "CC_NUM", "FDSTMP_CPS",
                "SEXUALITY", "OWNGUN_GSS", "RELIGCAT"
]

df_synthetic.head()

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd

likert_columns = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]

df_merged = pd.merge(
    df_synthetic,
    df_avg,
    how="left",
    left_on="id",
    right_on="SyntheticPersonID",
    suffixes=("", "_drop")
)

cols_to_drop = [col for col in df_merged.columns if col.endswith("_drop")]
df_final = df_merged.drop(columns=cols_to_drop)

# Fill only Q columns with 0
df_final[likert_columns] = df_final[likert_columns].fillna(0)

print(df_final.head())

# --- Add Condition back into df_final ---
cond_map = (
    grouped_df[["SyntheticPersonID", "Condition"]]
    .dropna()
    .drop_duplicates()
)

dupes = cond_map["SyntheticPersonID"][cond_map["SyntheticPersonID"].duplicated()].unique()
if len(dupes) > 0:
    print(f"WARNING: {len(dupes)} SyntheticPersonIDs appear in multiple conditions. Using majority vote.")
    cond_map = (
        grouped_df[["SyntheticPersonID", "Condition"]]
        .dropna()
        .groupby("SyntheticPersonID")["Condition"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )

# Your df_final uses id (not SyntheticPersonID) as the synthetic key
df_final = df_final.merge(cond_map, left_on="id", right_on="SyntheticPersonID", how="left")

# drop the extra join column we just added (keep the Condition)
df_final = df_final.drop(columns=["SyntheticPersonID"], errors="ignore")

print(df_final["Condition"].value_counts(dropna=False))

# =========================
# Survey 1 — PRE analysis (observed participant data only)
# =========================

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.stats import norm

# -------------------------
# Config
# -------------------------
questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]

recipient_labels = {
    "Q1": "Hackers / Cybercriminals",
    "Q2": "Government",
    "Q3": "Corporations",
    "Q4": "Employer / Colleagues",
    "Q5": "Family",
    "Q6": "Close Friends",
    "Q7": "Acquaintances",
    "Q8": "Publicly Available",
}

# -------------------------
# Safety checks
# -------------------------
required_cols = ["SyntheticPersonID", "Condition"] + questions
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in df: {missing}")

# Drop rows with no condition (these cannot be used in condition analysis)
df_pre = df.dropna(subset=["Condition"]).copy()

print("Condition counts (PRE):")
print(df_pre["Condition"].value_counts())

# -------------------------
# Build long-form dataset
#   (average repeated responses per SyntheticPersonID × Condition)
# -------------------------
grouped = (
    df_pre
    .groupby(["SyntheticPersonID", "Condition"], as_index=False)[questions]
    .mean()
)

long_pre = grouped.melt(
    id_vars=["SyntheticPersonID", "Condition"],
    value_vars=questions,
    var_name="Question",
    value_name="Response"
).dropna(subset=["Response"])

long_pre["Condition"] = long_pre["Condition"].astype("category")
long_pre["Question"] = pd.Categorical(long_pre["Question"], categories=questions, ordered=True)

# -------------------------
# Mixed-effects model
#   Random intercept: SyntheticPersonID
#   Fixed effects: Condition × Recipient
# -------------------------
model_pre = smf.mixedlm(
    "Response ~ C(Condition) * C(Question)",
    data=long_pre,
    groups=long_pre["SyntheticPersonID"]
)

mres_pre = model_pre.fit(reml=True, method="lbfgs")
print("converged:", getattr(mres_pre, "converged", None))
print("any NaN fe_params:", mres_pre.fe_params.isna().any())
print("any NaN cov_params:", np.isnan(mres_pre.cov_params().values).any())
print("min/max fe_params:", np.nanmin(mres_pre.fe_params.values), np.nanmax(mres_pre.fe_params.values))
print("first 10 fe_params:\n", mres_pre.fe_params.head(10))
print("random effect var:\n", mres_pre.cov_re)

print(mres_pre.summary())

# -------------------------
# Predicted means + 95% CI (fixed effects only)
# -------------------------
conditions = list(long_pre["Condition"].cat.categories)

grid = pd.DataFrame(
    [(c, q) for c in conditions for q in questions],
    columns=["Condition", "Question"]
)

grid["Condition"] = pd.Categorical(grid["Condition"], categories=conditions)
grid["Question"] = pd.Categorical(grid["Question"], categories=questions, ordered=True)

X = dmatrix(mres_pre.model.data.design_info, grid, return_type="dataframe")
# --- KEY FIX: use fixed-effect names only (exclude Group Var / RE params) ---
fe_names = mres_pre.fe_params.index
X = X.reindex(columns=fe_names, fill_value=0.0)


V = mres_pre.cov_params().loc[fe_names, fe_names]
beta = mres_pre.fe_params.loc[fe_names]


pred = np.asarray(X @ beta)
se = np.sqrt(np.sum((np.asarray(X @ V) * np.asarray(X)), axis=1))


z = norm.ppf(0.975)


pre_means = grid.copy()
pre_means["PredMean"] = pred
pre_means["CI_Low"] = pred - z * se
pre_means["CI_High"] = pred + z * se
pre_means["Recipient"] = pre_means["Question"].map(recipient_labels)


pre_means["Qnum"] = pre_means["Question"].str.replace("Q","", regex=False).astype(int)
pre_means = pre_means.sort_values(["Condition", "Qnum"]).drop(columns="Qnum")

display(pre_means)

# =========================
# Survey 1 — PRE: Omnibus tests + Effect sizes vs Control
#   (run this cell AFTER you already created: df_pre, grouped, long_pre, mres_pre, pre_means)
# =========================

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.stats import norm, chi2
from statsmodels.stats.multitest import multipletests

# -------------------------
# 1) OMNIBUS TESTS (Wald tests on MixedLM fixed effects)
#    Tests:
#      (a) Condition main effect
#      (b) Question main effect
#      (c) Condition × Question interaction
# -------------------------

def _wald_test_terms_mixedlm(mres, term_prefixes, term_name):
    """
    Wald chi-square test for a set of fixed-effect coefficients in a fitted MixedLM result.

    term_prefixes: list[str] of prefixes to select coefficients from mres.fe_params.index
      e.g. ["C(Condition)[T."] for condition main effects
           ["C(Question)[T."]  for question main effects
           ["C(Condition)[T.", ":C(Question)[T."] for interactions (handled below)
    """
    fe = mres.fe_params
    fe_names = fe.index.tolist()

    # Select coefficient names
    if term_name == "Condition × Question":
        # interactions look like: C(Condition)[T.X]:C(Question)[T.Qy]
        sel = [n for n in fe_names if (n.startswith("C(Condition)[T.") and ":C(Question)[T." in n)]
    else:
        # any of the prefixes
        sel = [n for n in fe_names if any(n.startswith(pref) for pref in term_prefixes)]

    if len(sel) == 0:
        raise ValueError(f"No coefficients found for term '{term_name}'. Check coding / names.")

    beta = fe.loc[sel].values
    V = mres.cov_params().loc[sel, sel].values

    # Wald statistic: beta' V^{-1} beta
    stat = float(beta.T @ np.linalg.inv(V) @ beta)
    df_ = int(len(sel))
    p = float(chi2.sf(stat, df_))
    return {"Term": term_name, "Chi2": stat, "df": df_, "p_value": p, "n_coefs": len(sel)}

omni_rows = []
omni_rows.append(_wald_test_terms_mixedlm(
    mres_pre,
    term_prefixes=["C(Condition)[T."],
    term_name="Condition"
))
omni_rows.append(_wald_test_terms_mixedlm(
    mres_pre,
    term_prefixes=["C(Question)[T."],
    term_name="Recipient (Question)"
))
omni_rows.append(_wald_test_terms_mixedlm(
    mres_pre,
    term_prefixes=[],
    term_name="Condition × Question"
))

omnibus_table = pd.DataFrame(omni_rows)
omnibus_table["p_value_fmt"] = omnibus_table["p_value"].map(lambda x: f"{x:.3g}")
display(omnibus_table[["Term","Chi2","df","p_value_fmt","n_coefs"]])

# -------------------------
# 2) EFFECT SIZES: differences vs Control (per Question × Condition)
#    Outputs a 24-row table (3 conditions × 8 questions):
#      - Δ mean vs Control
#      - 95% CI for Δ (delta method using fixed-effect covariance)
#      - Cohen’s d (using pooled SD of observed responses; after averaging within SyntheticPersonID×Condition)
# -------------------------

# Helper: compute contrasts from fitted MixedLM fixed effects
def _design_row_for(cond, q, conditions, questions, design_info, fe_names):
    row = pd.DataFrame({"Condition":[cond], "Question":[q]})
    row["Condition"] = pd.Categorical(row["Condition"], categories=conditions)
    row["Question"]   = pd.Categorical(row["Question"], categories=questions, ordered=True)
    Xrow = dmatrix(design_info, row, return_type="dataframe")
    Xrow = Xrow.reindex(columns=fe_names, fill_value=0.0)
    return Xrow

def contrast_vs_control_table(
    mres,
    grouped_profile_df,          # grouped = df_pre.groupby([SyntheticPersonID,Condition])[Q1..Q8].mean().reset_index()
    conditions_order,            # list of condition levels (must include "Control")
    questions,
    recipient_labels,
    control_label="Control",
    alpha=0.05
):
    fe_names = mres.fe_params.index
    design_info = mres.model.data.design_info
    V = mres.cov_params().loc[fe_names, fe_names]
    beta = mres.fe_params.loc[fe_names]

    z = norm.ppf(1 - alpha/2)

    # for Cohen's d, compute pooled SD from observed data per (cond, question)
    # using the profile-averaged rows (one row per SyntheticPersonID×Condition)
    out_rows = []
    compare_conds = [c for c in conditions_order if c != control_label]

    for q in questions:
        # Control distribution for d
        ctrl_vals = grouped_profile_df.loc[grouped_profile_df["Condition"] == control_label, q].dropna().values
        n0 = len(ctrl_vals)
        s0 = np.std(ctrl_vals, ddof=1) if n0 > 1 else np.nan

        for c in compare_conds:
            cond_vals = grouped_profile_df.loc[grouped_profile_df["Condition"] == c, q].dropna().values
            n1 = len(cond_vals)
            s1 = np.std(cond_vals, ddof=1) if n1 > 1 else np.nan

            # Contrast: (c,q) - (Control,q)
            X_c   = _design_row_for(c, q, conditions_order, questions, design_info, fe_names)
            X_ctl = _design_row_for(control_label, q, conditions_order, questions, design_info, fe_names)
            L = (X_c.values - X_ctl.values).reshape(-1)  # 1 x p -> vector

            diff = float(L @ beta.values)
            se = float(np.sqrt(L @ V.values @ L.T))
            ci_low = diff - z*se
            ci_high = diff + z*se

            # Cohen's d (pooled SD)
            if (n0 > 1) and (n1 > 1) and np.isfinite(s0) and np.isfinite(s1) and (s0 > 0 or s1 > 0):
                sp = np.sqrt(((n0 - 1)*(s0**2) + (n1 - 1)*(s1**2)) / (n0 + n1 - 2))
                d = diff / sp if sp > 0 else np.nan
            else:
                d = np.nan

            out_rows.append({
                "Condition": c,
                "Question": q,
                "Recipient": recipient_labels.get(q, q),
                "Delta_vs_Control": diff,
                "CI_Low": ci_low,
                "CI_High": ci_high,
                "SE": se,
                "Cohens_d": d,
                "n_cond": n1,
                "n_control": n0
            })

    res = pd.DataFrame(out_rows)

    # Round for display
    res = res.sort_values(["Condition", "Question"]).reset_index(drop=True)
    res["Delta_vs_Control"] = res["Delta_vs_Control"].round(3)
    res["CI_Low"] = res["CI_Low"].round(3)
    res["CI_High"] = res["CI_High"].round(3)
    res["Cohens_d"] = res["Cohens_d"].round(3)

    return res

# Make sure Condition labels are consistent and include Control
conditions_order = list(long_pre["Condition"].cat.categories)
if "Control" not in conditions_order:
    raise ValueError(f"'Control' not found in Condition categories: {conditions_order}")

# Use the profile-averaged grouped df you already created earlier as `grouped`
# (one row per SyntheticPersonID×Condition with Q1..Q8 means)
grouped_profile = grouped.copy()

effect_table = contrast_vs_control_table(
    mres=mres_pre,
    grouped_profile_df=grouped_profile,
    conditions_order=conditions_order,
    questions=questions,
    recipient_labels=recipient_labels,
    control_label="Control"
)

display(effect_table)

# Optional: add Holm correction across the 24 contrasts (Wald z-tests from diff/SE)
effect_table["z"] = effect_table["Delta_vs_Control"].astype(float) / effect_table["SE"].astype(float)
effect_table["p_value"] = 2 * norm.sf(np.abs(effect_table["z"].values))
effect_table["p_holm"] = multipletests(effect_table["p_value"], method="holm")[1]
effect_table["p_holm_fmt"] = effect_table["p_holm"].map(lambda x: f"{x:.3g}")

display(effect_table[["Condition","Question","Recipient","Delta_vs_Control","CI_Low","CI_High","Cohens_d","p_holm_fmt","n_cond","n_control"]])

# Optional: top-k largest absolute deltas (for main text)
topk = (
    effect_table
    .assign(abs_delta=lambda d: d["Delta_vs_Control"].abs())
    .sort_values("abs_delta", ascending=False)
    .head(8)
    .drop(columns="abs_delta")
)
print("Top deltas vs Control (by |Δ|):")
display(topk[["Condition","Question","Recipient","Delta_vs_Control","CI_Low","CI_High","Cohens_d","p_holm_fmt"]])

train_data = df_final[df_final['Q1'] > 0]
predict_data = df_final[df_final['Q1'] == 0]

X_train = train_data[feature_list]
y_train = train_data['Q1']

onehot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore',
    drop='first'
)

X_train_encoded = onehot_encoder.fit_transform(X_train.astype(str))

scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()

models = [
    ('RandomForest', RandomForestRegressor(n_estimators=100)),
    ('GradientBoosting', GradientBoostingRegressor()),
    ('LinearRegression', LinearRegression()),
    ('Ridge', Ridge()),
    ('Lasso', Lasso()),
    ('DecisionTree', DecisionTreeRegressor()),
    ('SVR', SVR())
]

def evaluate_model(model_name, model, X_train, y_train):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)
    print(f'Model: {model_name}')
    print(f'MSE on training data: {mse}')
    print(f'R^2 on training data: {r2}\n')

for name, model in models:
    evaluate_model(name, model, X_train_encoded, y_train_scaled)

questions = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
for q in questions:

  train_data = df_final[df_final[q] > 0]
  predict_data = df_final[df_final[q] == 0]

  X_train = train_data[feature_list]
  y_train = train_data[q]

  onehot_encoder = OneHotEncoder(
      sparse_output=False,
      handle_unknown='ignore',
      drop='first'
  )

  X_train_encoded = onehot_encoder.fit_transform(X_train.astype(str))

  scaler = StandardScaler()
  y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()

  models = [
      ('DecisionTree', DecisionTreeRegressor())
  ]

  def evaluate_model(model_name, model, X_train, y_train):
      model.fit(X_train, y_train)
      y_train_pred = model.predict(X_train)
      mse = mean_squared_error(y_train, y_train_pred)
      r2 = r2_score(y_train, y_train_pred)
      print(f'Model: {model_name}')
      print(f'MSE on training data: {mse}')
      print(f'R^2 on training data: {r2}\n')

  for name, model in models:
      evaluate_model(name, model, X_train_encoded, y_train_scaled)


  best_model = DecisionTreeRegressor(
      max_depth=30,
      max_features=None,
      min_samples_leaf=4,
      min_samples_split=5,
      random_state=42
  )
  best_model.fit(X_train_encoded, y_train_scaled)

  X_predict = predict_data[feature_list]
  X_predict_encoded = onehot_encoder.transform(X_predict.astype(str))
  predicted_scores_scaled = best_model.predict(X_predict_encoded)

  predicted_scores = scaler.inverse_transform(predicted_scores_scaled.reshape(-1, 1)).flatten()

  df_final.loc[df_final[q] == 0, q] = predicted_scores

  print(df_final.head())

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd

def cv_mae_for_imputation(
    df,
    feature_list,
    questions,
    n_splits=10,
    random_state=42
):
    results = []

    for q in questions:
        # Use only observed values (same rule as your imputation)
        data = df[df[q] > 0].copy()

        X = data[feature_list].astype(str)
        y = data[q].values.reshape(-1, 1)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_maes = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # One-hot encoding (fit on TRAIN only)
            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                drop='first'
            )
            X_train_enc = encoder.fit_transform(X_train)
            X_test_enc  = encoder.transform(X_test)

            # Scale y (exactly like your imputation)
            scaler = StandardScaler()
            y_train_scaled = scaler.fit_transform(y_train).ravel()

            # Model (same hyperparameters)
            model = DecisionTreeRegressor(
                max_depth=30,
                max_features=None,
                min_samples_leaf=4,
                min_samples_split=5,
                random_state=random_state
            )

            model.fit(X_train_enc, y_train_scaled)

            # Predict + invert scaling
            y_pred_scaled = model.predict(X_test_enc)
            y_pred = scaler.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()

            fold_maes.append(mean_absolute_error(y_test.ravel(), y_pred))

        results.append({
            "Question": q,
            "MAE_Mean": np.mean(fold_maes),
            "MAE_SD": np.std(fold_maes),
            "n_observed": len(data)
        })

    return pd.DataFrame(results)

mae_table = cv_mae_for_imputation(
    df=df_final,
    feature_list=feature_list,
    questions=questions,
    n_splits=10
)

display(mae_table)

df_final.head()

# drop syntheticpersonid column
df_final = df_final.drop(columns=['SyntheticPersonID'])
df_final = df_final.drop(columns=['avg_harm'])

"""# Participant Graphs"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_responses(df, question, demographic, kind="bar"):
    """
    Visualize responses for one question by a demographic group.

    Parameters
    ----------
    df : pandas.DataFrame
        Your merged survey dataframe.
    question : str
        Column name of the question (e.g., 'Q5').
    demographic : str
        Column name of the demographic variable (e.g., 'Sex').
    kind : {'bar', 'violin', 'box'}, optional
        Plot type (default 'bar').
    """
    plt.figure(figsize=(10, 5))

    if kind == "bar":
        sns.barplot(data=df, x=demographic, y=question, ci="sd", palette="muted")
    elif kind == "violin":
        sns.violinplot(data=df, x=demographic, y=question, inner="quartile", palette="Set3")
    elif kind == "box":
        sns.boxplot(data=df, x=demographic, y=question, palette="Set2")

    plt.title(f"{question} responses by {demographic}")
    plt.ylabel("Likert Score")
    plt.xlabel(demographic)
    plt.ylim(1, 5)  # assuming a 1–5 Likert scale
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

plot_responses(merged_df, question="Q5", demographic="Ethnicity simplified", kind="violin")
#plot_responses(merged_df, question="Q8", demographic="Employment status", kind="bar")

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(8, 5))
question = "Q6"

sns.barplot(
    data=merged_df[merged_df["Ethnicity simplified"] != "DATA_EXPIRED"],
    x="Ethnicity simplified",
    y=question,
    estimator="mean",          # plot means
    errorbar=("ci", 95),        # 95% confidence interval
    color="white",
    edgecolor="black",
    ax=ax
)

ax.set_title(f"{question} responses by Ethnicity (Mean ± 95% CI)")
ax.set_xlabel("Ethnicity")
ax.set_ylabel("Average harm rating")
ax.set_ylim(1, 5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(f"{question.lower()}_ethnicity_barplot_means.pdf", format="pdf", bbox_inches="tight")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

sns.set(style="whitegrid")

def plot_responses(df, question, demographic, kind="bar"):
    plt.figure(figsize=(10, 5))
    if kind == "bar":
        sns.barplot(data=df, x=demographic, y=question, ci="sd", palette="muted")
    elif kind == "violin":
        sns.violinplot(data=df, x=demographic, y=question, inner="quartile", palette="Set3")
    elif kind == "box":
        sns.boxplot(data=df, x=demographic, y=question, palette="Set2")
    plt.title(f"{question} responses by {demographic}")
    plt.ylabel("Likert Score")
    plt.xlabel(demographic)
    plt.ylim(1, 5)  # Likert scale
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

# >>> Replace this with your real dataframe
df = merged_df

questions = [f"Q{i}" for i in range(1, 9)]
demographics = [
    "Sex",
    "Ethnicity simplified",
    "Country of residence",
    "Language",
    "Employment status",
]

output_path = "survey_plots.pdf"
with PdfPages(output_path) as pdf:
    for q in questions:
        for demo in demographics:
            plot_responses(df, q, demo, kind="violin")
            pdf.savefig()  # save the current figure
            plt.close()

print(f"PDF saved to {output_path}")

"""## Graphs"""

import seaborn as sns

score_types = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
palette = sns.color_palette("Set2")

q = 1

for score_type in score_types:
    average_scores_by_race_sexuality = df_final.groupby(['RACETHN', 'SEXUALITY'])[score_type].median().reset_index()
    plt.figure(figsize=(12, 8))

    sns.barplot(x='RACETHN', y=score_type, hue='SEXUALITY', data=average_scores_by_race_sexuality, palette=palette)

    plt.title(f'Average Scores by Race/Ethnicity and Sexuality for Question {q}')
    q += 1
    plt.xlabel('Race/Ethnicity')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45)
    plt.legend(title='Sexuality')

    plt.tight_layout()
    plt.show()

"""Does sexuality impact harm?"""

import itertools
import scipy.stats as stats

score_types = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]

# Loop over each question
for score in score_types:
    print(f"\nResults for {score}:")

    # Pairwise t-tests for SEXUALITY
    print("Comparisons by SEXUALITY:")
    sexuality_groups = df_final['SEXUALITY'].dropna().unique()
    for group1, group2 in itertools.combinations(sexuality_groups, 2):
        data1 = df_final[df_final['SEXUALITY'] == group1][score].dropna()
        data2 = df_final[df_final['SEXUALITY'] == group2][score].dropna()
        if len(data1) >= 2 and len(data2) >= 2:
            t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
            print(f"  {group1} vs {group2}: t = {t_stat:.3f}, p = {p_val:.3e}")

    # # Pairwise t-tests for RACETHN
    # print("Comparisons by RACETHN:")
    # race_groups = df_final['RACETHN'].dropna().unique()
    # for group1, group2 in itertools.combinations(race_groups, 2):
    #     data1 = df_final[df_final['RACETHN'] == group1][score].dropna()
    #     data2 = df_final[df_final['RACETHN'] == group2][score].dropna()
    #     if len(data1) >= 2 and len(data2) >= 2:
    #         t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
    #         print(f"  {group1} vs {group2}: t = {t_stat:.3f}, p = {p_val:.3e}")

import itertools
import scipy.stats as stats

score_types = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]

# Loop over each harm question
for score in score_types:
    print(f"\n=== T-test Results for {score} ===")

    # 1) Pairwise t-tests by HIV_STAT
    print("Comparisons by HIV_STAT:")
    hiv_groups = df_final['HIV_STAT'].dropna().unique()
    for grp1, grp2 in itertools.combinations(hiv_groups, 2):
        data1 = df_final.loc[df_final['HIV_STAT'] == grp1, score].dropna()
        data2 = df_final.loc[df_final['HIV_STAT'] == grp2, score].dropna()

        # Ensure both groups have at least 2 values to run a t-test
        if len(data1) >= 2 and len(data2) >= 2:
            t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
            print(f"  {grp1} vs {grp2}: t = {t_stat:.3f}, p = {p_val:.3e}")
        else:
            print(f"  {grp1} vs {grp2}: Not enough data.")

import itertools
import scipy.stats as stats

score_types = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]

# Loop over each harm question
for score in score_types:
    print(f"\n=== T-test Results for {score} ===")

    # 1) Pairwise t-tests by HIV_STAT
    print("Comparisons by FAMINC5:")
    hiv_groups = df_final['FAMINC5'].dropna().unique()
    for grp1, grp2 in itertools.combinations(hiv_groups, 2):
        data1 = df_final.loc[df_final['FAMINC5'] == grp1, score].dropna()
        data2 = df_final.loc[df_final['FAMINC5'] == grp2, score].dropna()

        # Ensure both groups have at least 2 values to run a t-test
        if len(data1) >= 2 and len(data2) >= 2:
            t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
            print(f"  {grp1} vs {grp2}: t = {t_stat:.3f}, p = {p_val:.3e}")
        else:
            print(f"  {grp1} vs {grp2}: Not enough data.")

import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap_per_question(df, questions, mask_zeros=True, save=False, outdir="."):
    """
    Generate heatmaps of mean values for each question (Q1..Q8)
    by Division × Family Income group.
    """
    canonical_levels = [
        "Less than $20K",
        "$20K to less than $40K",
        "$40K to less than $75K",
        "$75K to less than $150K",
        "$150K or more",
    ]

    for score in questions:
        df_copy = df.copy()

        # If 0 was just a placeholder, treat as missing
        if mask_zeros and score in df_copy:
            df_copy.loc[df_copy[score] == 0, score] = np.nan

        df_copy["FAMINC5"] = pd.Categorical(
            df_copy["FAMINC5"], categories=canonical_levels, ordered=True
        )

        # Compute group means
        means = (
            df_copy.groupby(["DIVISION","FAMINC5"])[score]
                   .mean()
                   .unstack()
                   .reindex(columns=canonical_levels)
        )

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(means, annot=True, fmt=".2f", cmap="Greys", cbar=False, ax=ax)
        ax.set_title(f"{score}: Mean by Division × Family Income")
        ax.set_xlabel("Family Income")
        ax.set_ylabel("Division")
        plt.tight_layout()

        if save:
            pdf_path = f"{outdir}/Heatmap_{score}.pdf"
            png_path = f"{outdir}/Heatmap_{score}.png"
            plt.savefig(pdf_path, bbox_inches="tight")
            plt.savefig(png_path, bbox_inches="tight", dpi=300)
            print(f"Saved {pdf_path} and {png_path}")

        plt.show()

# --- Example usage ---
questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]
plot_heatmap_per_question(df_final, questions, mask_zeros=True, save=True, outdir=".")

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_heatmap_per_question(
    df, questions, mask_zeros=True, save=False, outdir=".",
    fontsize=18, figsize=(8, 6), dpi=100, save_dpi=300
):
    """
    Heatmaps of mean values (Q1..Q8) by Division × Family Income.
    Shows at a normal size in Jupyter and saves with large fonts.
    """
    canonical_levels = [
        "Less than $20K",
        "$20K to less than $40K",
        "$40K to less than $75K",
        "$75K to less than $150K",
        "$150K or more",
    ]

    sns.set_theme(style="white")

    for score in questions:
        df_copy = df.copy()

        if mask_zeros and score in df_copy:
            df_copy.loc[df_copy[score] == 0, score] = np.nan

        df_copy["FAMINC5"] = pd.Categorical(
            df_copy["FAMINC5"], categories=canonical_levels, ordered=True
        )

        means = (
            df_copy.groupby(["DIVISION", "FAMINC5"])[score]
            .mean()
            .unstack()
            .reindex(columns=canonical_levels)
        )

        # normal-size preview
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        sns.heatmap(
            means,
            annot=True,
            fmt=".2f",
            cmap="Greys",
            cbar=False,
            ax=ax,
            annot_kws={"size": fontsize - 2},     # numbers inside cells
            linewidths=0.5,
            linecolor="lightgray"
        )

        # Explicitly set label and tick font sizes
        ax.set_title(f"{score}: Mean by Division × Family Income",
                     fontsize=fontsize + 2, pad=10)
        ax.set_xlabel("Family Income", fontsize=fontsize)
        ax.set_ylabel("Division", fontsize=fontsize)

        ax.tick_params(axis="x", labelsize=fontsize - 2, rotation=30)
        ax.tick_params(axis="y", labelsize=fontsize - 2)

        plt.tight_layout()

        if save:
            base = f"{outdir}/Heatmap_{score}"
            plt.savefig(f"{base}.pdf", bbox_inches="tight", dpi=save_dpi)
            plt.savefig(f"{base}.png", bbox_inches="tight", dpi=save_dpi)
            print(f"Saved {base}.pdf / .png")

        plt.show()

questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]

plot_heatmap_per_question(
    df_final,
    questions,
    fontsize=18,     # big labels
    figsize=(8,6),   # normal display
    dpi=100,         # notebook view
    save=True,
    save_dpi=300     # publication quality
)

import seaborn as sns
import matplotlib.pyplot as plt

score_types = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]

for score_type in score_types:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='DIVISION', y=score_type, hue='FAMINC5', data=df_final)
    plt.title(f'Distribution of {score_type} by Family Income and Race/Ethnicity')
    plt.xlabel('Race/Ethnicity')
    plt.ylabel(score_type)
    plt.legend(title='Family Income')
    plt.xticks(rotation=45)
    plt.show()

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# List of your harm questions
questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]

# We'll loop over each question and fit a model like:
# QX ~ C(HIV_STAT)*C(GENDER) + C(HIV_STAT)*C(RACETHN) + C(HIV_STAT)*C(SEXUALITY)
for q in questions:
    formula = (
        f"{q} ~ C(HIV_STAT)*C(GENDER) "
        f"+ C(HIV_STAT)*C(RACETHN) "
        f"+ C(HIV_STAT)*C(SEXUALITY)"
    )
    model = ols(formula, data=df_final).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(f"ANOVA for {q} (HIV status interactions):")
    print(anova_table)
    print("\n")

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# List of your harm questions
questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]

# We'll loop over each question and fit a model like:
# QX ~ C(HIV_STAT)*C(GENDER) + C(HIV_STAT)*C(RACETHN) + C(HIV_STAT)*C(SEXUALITY)
for q in questions:
    formula = (
        f"{q} ~ C(PREG_STAT)*C(OWNGUN_GSS) "
        f"+ C(PREG_STAT)*C(EDUCCAT5) "
        f"+ C(PREG_STAT)*C(AGE_INT)"
    )
    model = ols(formula, data=df_final).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(f"ANOVA for {q} (Pregnancy):")
    print(anova_table)
    print("\n")

score_types = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
score_type_to_question = {
    'Q1': 'Question 1',
    'Q2': 'Question 2',
    'Q3': 'Question 3',
    'Q4': 'Question 4',
    'Q5': 'Question 5',
    'Q6': 'Question 6',
    'Q7': 'Question 7',
    'Q8': 'Question 8'
}

for score_type in score_types:
    grouped_scores = df_final.groupby(['AGE', 'PREG_STAT'])[score_type].median().reset_index()

    g = sns.lmplot(x='AGE', y=score_type, data=grouped_scores, hue='PREG_STAT', height=6, aspect=1.5, legend_out=True)

    plt.title(f'Mean {score_type_to_question[score_type]} vs. Age colored by Pregnancy Status')
    plt.xlabel('Age')
    plt.ylabel(score_type)
    plt.xlim(16, 40)

    plt.subplots_adjust(top=0.9)
    plt.show()

import itertools
import scipy.stats as stats

score_types = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]

# Loop over each harm question
for score in score_types:
    print(f"\n=== T-test Results for {score} ===")

    # 1) Pairwise t-tests by HIV_STAT
    print("Comparisons by PREG_STAT:")
    hiv_groups = df_final['PREG_STAT'].dropna().unique()
    for grp1, grp2 in itertools.combinations(hiv_groups, 2):
        data1 = df_final.loc[df_final['PREG_STAT'] == grp1, score].dropna()
        data2 = df_final.loc[df_final['PREG_STAT'] == grp2, score].dropna()

        # Ensure both groups have at least 2 values to run a t-test
        if len(data1) >= 2 and len(data2) >= 2:
            t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
            print(f"  {grp1} vs {grp2}: t = {t_stat:.3f}, p = {p_val:.3e}")
        else:
            print(f"  {grp1} vs {grp2}: Not enough data.")
