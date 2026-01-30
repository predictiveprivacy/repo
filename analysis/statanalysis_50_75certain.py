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
df = pd.read_csv('data/survey2_results.csv')

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

df.head()

import pandas as pd

# Assuming your DataFrame is called df and has columns:
# ["ProlificID", "SyntheticPersonID", "AttnCheck", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Condition"]

# 1) Group by SyntheticPersonID and compute the mean for Q1..Q8
df_avg = (
    df.groupby("SyntheticPersonID")[["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8", "Certainty"]]
      .mean()  # average across repeated rows for each SyntheticPersonID
      .reset_index()
)

# Now df_avg contains one row per unique SyntheticPersonID
# with columns for Q1..Q8 averages, and optionally avg_harm.

# include columnn Certainty without taking the mean of that column to df_avg
print(df_avg.head())

import pandas as pd
import itertools
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

# 1) Group by SyntheticPersonID and Condition, averaging Q1..Q8
#    for repeated rows of the same SyntheticPersonID.
grouped_df = (
    df.groupby(["SyntheticPersonID", "Condition"])[["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8", "Certainty"]]
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

# We'll assume `grouped_df` has columns:
# ["SyntheticPersonID", "Condition", "Certainty", "Q1", "Q2", ..., "Q8"]

questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]
conditions = grouped_df["Condition"].unique()
certainties = grouped_df["Certainty"].unique()

# Sort certainties if desired (e.g., 50, then 75)
certainties = sorted(certainties)

for cond in conditions:
    plt.figure(figsize=(10, 6))

    for i, cert in enumerate(certainties):
        subset = grouped_df[
            (grouped_df["Condition"] == cond) &
            (grouped_df["Certainty"] == cert)
        ]

        mean_vals = subset[questions].mean(axis=0)
        std_vals  = subset[questions].std(axis=0)
        n = subset.shape[0]
        se_vals = std_vals / np.sqrt(n) if n > 1 else np.zeros_like(std_vals)

        if n > 1:
            t_multiplier = st.t.ppf(1 - 0.025, df=n-1)
            ci_vals = t_multiplier * se_vals
        else:
            ci_vals = np.zeros_like(se_vals)

        x_positions = np.arange(len(questions))
        offset = i * 0.1
        shifted_positions = x_positions + offset

        plt.errorbar(
            shifted_positions,
            mean_vals,
            yerr=ci_vals,
            fmt='o',
            capsize=5,
            label=f"Certainty = {cert}%"
        )

    question_labels = [custom_labels[q] for q in questions]
    plt.xticks(np.arange(len(questions)), question_labels, rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=18)

    plt.xlabel("Scenario", fontsize=18)
    plt.ylabel("Mean Response", fontsize=18)
    plt.title(f"Mean Response (95% CI) by Certainty for Condition: {cond}", fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()

    # Save the figure as a PDF
    plt.savefig(f'mean_response_{cond}.pdf', bbox_inches='tight')

    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

# We'll assume grouped_df has one row per (SyntheticPersonID, Condition, Certainty)
# and columns for Q1..Q8, like:
# ["SyntheticPersonID", "Condition", "Certainty", "Q1", "Q2", ..., "Q8"]

questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]
conditions = grouped_df["Condition"].unique()
certainties = sorted(grouped_df["Certainty"].unique())  # e.g., [50, 75]

for question in questions:
    # Prepare lists to store the mean, CI, and x-axis labels
    means = []
    ci_vals = []
    x_labels = []

    # Loop over each Condition and Certainty combination
    for cond in conditions:
        for cert in certainties:
            # Filter rows that match the current Condition and Certainty
            subset = grouped_df.loc[
                (grouped_df["Condition"] == cond) & (grouped_df["Certainty"] == cert),
                question
            ].dropna()

            mean_val = subset.mean()
            std_val = subset.std()
            n = subset.shape[0]

            # Compute 95% CI using the t-distribution
            if n > 1:
                se_val = std_val / np.sqrt(n)
                t_multiplier = st.t.ppf(1 - 0.025, df=n-1)  # for a 95% CI
                ci = t_multiplier * se_val
            else:
                # If there's only one data point, the CI isn't meaningful
                ci = 0.0

            means.append(mean_val)
            ci_vals.append(ci)
            x_labels.append(f"{cond}-{int(cert)}%")  # e.g. "Health-50%" or "Control-75%"

    # Now we can plot the results for this question
    x_positions = np.arange(len(means))  # one position per (Condition, Certainty) pair

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x_positions,
        means,
        yerr=ci_vals,
        fmt='o',
        capsize=5,
        color='blue',
        ecolor='black'
    )

    plt.xticks(x_positions, x_labels, rotation=45, ha='right')
    plt.xlabel("Condition - Certainty")
    plt.ylabel("Mean Response")
    plt.title(f"Mean {question} by Condition and Certainty (95% CI)")
    plt.tight_layout()
    plt.show()

import scipy.stats as stats

score_types = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
conditions_of_interest = ["Health", "Control"]  # or use grouped_df["Condition"].unique()

for cond in conditions_of_interest:
    print(f"\n=== Condition: {cond} ===")

    # Subset the data for just this condition
    subset_cond = grouped_df[grouped_df["Condition"] == cond]

    for score in score_types:
        # Separate data for 50% vs 75% certainty
        data_50 = subset_cond[subset_cond["Certainty"] == 50][score].dropna()
        data_75 = subset_cond[subset_cond["Certainty"] == 75][score].dropna()

        # Only run a t-test if both groups have sufficient data
        if len(data_50) >= 2 and len(data_75) >= 2:
            t_stat, p_val = stats.ttest_ind(data_50, data_75, equal_var=False)
            print(f"{score} | t = {t_stat:.3f}, p = {p_val:.3e}")
        else:
            print(f"{score} | Not enough data for t-test.")

df_responses = pd.read_csv('data/survey2_results.csv')       # Adjust filename/path as needed
df_prolific  = pd.read_csv('data/prolific_export_survey2.csv') # Note: this data has to come from your own surveys

df_responses.head()

df_responses = pd.read_csv('data/survey2_results.csv')       # Adjust filename/path as needed
df_prolific  = pd.read_csv('data/prolific_export_survey2.csv')   # Note: this data has to come from your own surveys


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
results_df.to_csv('t-tests_survey2.csv', index=False)

grouped_df = (
    df.groupby(["SyntheticPersonID", "Condition", "Certainty"])[questions]
      .mean()
      .reset_index()
)

long_df = grouped_df.melt(
    id_vars=["SyntheticPersonID", "Condition", "Certainty"],
    value_vars=questions,
    var_name="Question",
    value_name="Response"
).dropna(subset=["Response"])

long_df["Certainty_cat"] = pd.Categorical(long_df["Certainty"], categories=[50, 75])

long_df.head()

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.stats import norm

questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]

# --- Ensure categorical baselines match what was fit ---
long_df["Question"] = pd.Categorical(long_df["Question"], categories=questions)
long_df["Certainty_cat"] = pd.Categorical(long_df["Certainty_cat"], categories=[50,75], ordered=True)
if "Control" in long_df["Condition"].unique():
    long_df["Condition"] = pd.Categorical(long_df["Condition"], categories=["Control","Health"])

# --- Fit model ---
md = smf.mixedlm(
    "Response ~ Condition * Certainty_cat * Question",
    data=long_df,
    groups=long_df["SyntheticPersonID"]
)
mres = md.fit(reml=True, method="lbfgs")
print(mres.summary())

# ============================================================
# Fix: use ONLY fixed-effect covariance aligned to X columns
# ============================================================
fe_names = list(mres.fe_params.index)                 # fixed effect parameter names
V_full = mres.cov_params()                            # may include extra params
V_fe = V_full.loc[fe_names, fe_names]                 # align to fixed effects ONLY
z = norm.ppf(0.975)

# --- Prediction grid ---
conds = list(long_df["Condition"].cat.categories) if isinstance(long_df["Condition"].dtype, pd.CategoricalDtype) \
        else sorted(long_df["Condition"].unique())

pred_grid = pd.DataFrame(
    [{"Condition": c, "Certainty_cat": cert, "Question": q}
     for c in conds for cert in [50,75] for q in questions]
)

pred_grid["Condition"] = pd.Categorical(pred_grid["Condition"], categories=conds)
pred_grid["Certainty_cat"] = pd.Categorical(pred_grid["Certainty_cat"], categories=[50,75], ordered=True)
pred_grid["Question"] = pd.Categorical(pred_grid["Question"], categories=questions)

# --- Predicted means (fixed effects) ---
pred_grid["pred_mean"] = mres.predict(pred_grid)

# --- Design matrix for grid ---
X = dmatrix(mres.model.data.design_info, pred_grid, return_type="dataframe")

# Make sure X columns align to fixed effects exactly
X = X[fe_names]

# --- Delta-method SE and CI for predicted means ---
XV = X.to_numpy() @ V_fe.to_numpy()
se = np.sqrt(np.sum(XV * X.to_numpy(), axis=1))

pred_grid["ci_low"]  = pred_grid["pred_mean"] - z * se
pred_grid["ci_high"] = pred_grid["pred_mean"] + z * se

predicted_means = pred_grid.copy()

# --- Recipient-specific AME: (75 - 50) within each condition & question ---
rows = []
for c in conds:
    for q in questions:
        d50 = pd.DataFrame([{"Condition": c, "Certainty_cat": 50, "Question": q}])
        d75 = pd.DataFrame([{"Condition": c, "Certainty_cat": 75, "Question": q}])

        d50["Condition"] = pd.Categorical(d50["Condition"], categories=conds)
        d75["Condition"] = pd.Categorical(d75["Condition"], categories=conds)
        d50["Certainty_cat"] = pd.Categorical(d50["Certainty_cat"], categories=[50,75], ordered=True)
        d75["Certainty_cat"] = pd.Categorical(d75["Certainty_cat"], categories=[50,75], ordered=True)
        d50["Question"] = pd.Categorical(d50["Question"], categories=questions)
        d75["Question"] = pd.Categorical(d75["Question"], categories=questions)

        mu50 = float(mres.predict(d50))
        mu75 = float(mres.predict(d75))
        ame  = mu75 - mu50

        X50 = dmatrix(mres.model.data.design_info, d50, return_type="dataframe")[fe_names].to_numpy()
        X75 = dmatrix(mres.model.data.design_info, d75, return_type="dataframe")[fe_names].to_numpy()
        dX = X75 - X50

        se_diff = float(np.sqrt(dX @ V_fe.to_numpy() @ dX.T))

        rows.append({
            "Condition": c,
            "Question": q,
            "Mean_50": mu50,
            "Mean_75": mu75,
            "AME_75_minus_50": ame,
            "CI_low": ame - z * se_diff,
            "CI_high": ame + z * se_diff,
        })

recipient_ame = pd.DataFrame(rows)

# Optional readable labels
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
recipient_ame["Recipient"] = recipient_ame["Question"].map(custom_labels)

print("\nRecipient-specific marginal effects (75% − 50%):")
display(recipient_ame)

import numpy as np
import pandas as pd


# Create two counterfactual datasets:
# one with Certainty=50, one with Certainty=75

# Copy data
df_50 = long_df.copy()
df_75 = long_df.copy()

# Set certainty values directly
df_50["Certainty"] = 50
df_75["Certainty"] = 75

# Recreate categorical version with same categories
df_50["Certainty_cat"] = pd.Categorical(
    df_50["Certainty"],
    categories=long_df["Certainty_cat"].cat.categories
)

df_75["Certainty_cat"] = pd.Categorical(
    df_75["Certainty"],
    categories=long_df["Certainty_cat"].cat.categories
)

# Predict under both
pred_50 = mres.predict(df_50)
pred_75 = mres.predict(df_75)


# Average marginal effect
ame_certainty = np.mean(pred_75 - pred_50)


# Bootstrap CI
rng = np.random.default_rng(0)
boot = []
for _ in range(1000):
  idx = rng.choice(len(pred_50), len(pred_50), replace=True)
  boot.append(np.mean(pred_75[idx] - pred_50[idx]))


ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

ame_certainty, ci_lo, ci_hi

"""## Supervised ML"""

from sklearn.preprocessing import LabelEncoder

feature_list = ["GENDER", "RACETHN", "EDUCCAT5", "DIVISION", "MARITAL_ACS",
                "CHILDRENCAT", "CITIZEN_REC", "BORN_ACS", "AGE_INT",
                "HIV_STAT", "PREG_STAT", "NumChronicIllness",
                "FAMINC5", "CC_NUM", "FDSTMP_CPS",
                "SEXUALITY", "OWNGUN_GSS", "RELIGCAT", "Certainty"
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

# 1) Merge df_synthetic and df_avg on their ID columns, using suffixes to keep them distinct initially.
df_merged = pd.merge(
    df_synthetic,
    df_avg,
    how="left",  # or "inner" if you only want rows with a matching ID in df_avg
    left_on="id",
    right_on="SyntheticPersonID",
    suffixes=("", "_drop")  # Keep df_synthetic columns as-is, add "_drop" to df_avg duplicates
)

# 2) If you prefer to keep the question columns from df_avg and drop the old ones from df_synthetic:
#    (assuming df_synthetic might already have Q1..Q8 but we want the new ones from df_avg)
cols_to_drop = [col for col in df_merged.columns if col.endswith("_drop")]
df_final = df_merged.drop(columns=cols_to_drop)
df_final = df_final.fillna(0)

# 3) Inspect the final result
print(df_final.head())

# for the rows where Certainty is 0, randomly assign them a value of 50 or 75
df_final.loc[df_final['Certainty'] == 0, 'Certainty'] = np.random.choice([50, 75], size=df_final.loc[df_final['Certainty'] == 0].shape[0])

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

# drop syntheticpersonid column
df_final = df_final.drop(columns=['SyntheticPersonID'])

df_final.head()

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, make_scorer

# Questions to evaluate
questions = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]

# MAE scorer (lower is better → negate for sklearn)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

results = []

for q in questions:
    # Use only observed (non-imputed) values
    data = df_final[df_final[q] > 0].copy()

    X = data[feature_list]
    y = data[q]

    # Preprocessing: one-hot encode categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), X.columns)
        ]
    )

    # Model (same family you used)
    model = DecisionTreeRegressor(
        max_depth=30,
        min_samples_leaf=4,
        min_samples_split=5,
        random_state=42
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    # 10-fold CV
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    cv_mae = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=mae_scorer
    )

    results.append({
        "Question": q,
        "MAE_mean": -cv_mae.mean(),
        "MAE_sd": cv_mae.std()
    })

# Final table
cv_mae_df = pd.DataFrame(results)
cv_mae_df

"""## Graphs"""

import seaborn as sns
import matplotlib.pyplot as plt

score_types = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
palette = sns.color_palette("Set2")

for score_type in score_types:
    # Filter the dataframe for each certainty level
    df_50 = df_final[df_final['Certainty'] == 50]
    df_75 = df_final[df_final['Certainty'] == 75]

    # Group and calculate the median (or mean if you prefer) for each subgroup
    average_scores_50 = (
        df_50.groupby(['RACETHN', 'SEXUALITY'])[score_type]
        .median()
        .reset_index()
    )
    average_scores_75 = (
        df_75.groupby(['RACETHN', 'SEXUALITY'])[score_type]
        .median()
        .reset_index()
    )

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Left subplot: Certainty = 50
    sns.barplot(
        x='RACETHN',
        y=score_type,
        hue='SEXUALITY',
        data=average_scores_50,
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title(f'{score_type} (Certainty = 50%)')
    axes[0].set_xlabel('Race/Ethnicity')
    axes[0].set_ylabel('Median Score')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title='Sexuality')

    # Right subplot: Certainty = 75
    sns.barplot(
        x='RACETHN',
        y=score_type,
        hue='SEXUALITY',
        data=average_scores_75,
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title(f'{score_type} (Certainty = 75%)')
    axes[1].set_xlabel('Race/Ethnicity')
    axes[1].set_ylabel('Median Score')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Sexuality')

    # Overall figure title
    plt.suptitle(f'Comparison of Certainty Levels for {score_type}', fontsize=14)

    plt.tight_layout()
    plt.show()

"""Does certainty of 50 vs 75 produce significantly diff harm scores? In almost all cases, it does."""

import itertools
from scipy import stats

score_types = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]

for score in score_types:
    # Separate data for 50% vs 75% certainty
    data_50 = df_final[df_final['Certainty'] == 50][score].dropna()
    data_75 = df_final[df_final['Certainty'] == 75][score].dropna()

    # Only run a t-test if both groups have sufficient data
    if len(data_50) >= 2 and len(data_75) >= 2:
        t_stat, p_val = stats.ttest_ind(data_50, data_75, equal_var=False)
        print(f"{score} | t = {t_stat:.3f}, p = {p_val:.3e}")
    else:
        print(f"{score} | Not enough data for t-test.")
