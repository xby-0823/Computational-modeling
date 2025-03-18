import pandas as pd
import numpy as np
import itertools, random
from math import comb
from scipy import stats

# -----------------------------
# Step 1. Read the input Excel file
# -----------------------------
data_file = r""
df = pd.read_excel(data_file)

# Extract patient IDs as strings
ID_list = [str(x) for x in df['ID']]

# Define metrics and corresponding column names for baseline and follow-up measurements
metrics = [
    ("OCT_GCC", "baseline oct gcc thickness", "first follow oct gcc thickness"),
    ("MD", "baseline md", "first follow md"),
    ("VFI", "baseline vfi", "first follow vfi")
]

# Dictionary to store detailed simulation results for each metric
results_dict = {}

# -----------------------------
# Step 2. Run simulations for each metric
# -----------------------------
for metric_name, base_col, follow_col in metrics:
    baseline_values = df[base_col].values
    follow_values   = df[follow_col].values

    # List to store each simulation result (detailed result)
    results_rows = []

    # Loop through sample sizes from 5 to 25
    for N in range(5, 26):
        total_combos = comb(len(df), N)  # Total number of combinations from 25 patients choose N

        # If total combinations are <= 1000, use all combinations; otherwise, select 1000 unique random combinations
        if total_combos <= 1000:
            combo_iterable = itertools.combinations(range(len(df)), N)
            combos_list = list(combo_iterable)
            random.shuffle(combos_list)  # Optional: shuffle the order
        else:
            combos_set = set()
            combos_list = []
            while len(combos_list) < 1000:
                combo = tuple(sorted(random.sample(range(len(df)), N)))
                if combo not in combos_set:
                    combos_set.add(combo)
                    combos_list.append(combo)

        # For each combination, perform the statistical tests
        for idx, combo in enumerate(combos_list, start=1):
            indices = list(combo)
            # Extract baseline and follow-up values for the selected patients
            vals_base   = baseline_values[indices]
            vals_follow = follow_values[indices]
            differences = vals_follow - vals_base

            # Perform Shapiro–Wilk test for normality on the differences
            if np.allclose(differences, 0):
                normal = True
                normal_pvalue = 1.0
            else:
                W_stat_sw, p_shapiro = stats.shapiro(differences)
                normal_pvalue = p_shapiro
                normal = p_shapiro > 0.05

            # Depending on normality, choose the appropriate test
            if normal:
                test_used = "Paired t-test"
                t_stat, p_val = stats.ttest_rel(vals_base, vals_follow)
                effect_val = t_stat  # Use t-value as the effect size
            else:
                test_used = "Wilcoxon"
                wilcoxon_res = stats.wilcoxon(vals_base, vals_follow)
                W_stat = wilcoxon_res.statistic  # Raw Wilcoxon test statistic
                p_val = wilcoxon_res.pvalue

                # Calculate standardized z-value from the Wilcoxon statistic
                n_eff = np.sum(differences != 0)
                if n_eff > 0:
                    mean_w = n_eff * (n_eff + 1) / 4
                    std_w = np.sqrt(n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24)
                    z_val = (W_stat - mean_w) / std_w
                else:
                    z_val = 0
                effect_val = z_val  # Use z-value as the effect size

            # Add a new column "Significantly Different": yes if p-value < 0.05, else no
            sig_diff = "yes" if p_val < 0.05 else "no"

            # Format the selected patient IDs as a comma-separated string and create a sample label ("N.idx")
            selected_ids_str = ", ".join(ID_list[i] for i in indices)
            sample_label = f"{N}.{idx}"

            # Append the simulation result row
            results_rows.append([
                sample_label, selected_ids_str,
                "Yes" if normal else "No", normal_pvalue,
                test_used, effect_val, p_val, sig_diff
            ])

    # Create a DataFrame for the detailed simulation results of the metric
    df_results = pd.DataFrame(results_rows, columns=[
        "Sample", "IDs", "Normality", "Normality Pvalue",
        "Test", "Statistic", "P-value", "Significantly Different"
    ])
    df_results.set_index("Sample", inplace=True)
    results_dict[metric_name] = df_results

# -----------------------------
# Step 3. Compute summary statistics for each metric
# For each sample size n, calculate the total number of trials and the proportion with p-value < 0.05.
# -----------------------------
summary_dict = {}
for metric_name, df_results in results_dict.items():
    df_temp = df_results.copy()
    # Extract the sample size (n) from the Sample label (e.g., "5.1" -> 5)
    df_temp['n'] = df_temp.index.str.split('.').str[0].astype(int)
    summary = df_temp.groupby('n').apply(
        lambda group: pd.Series({
            "Total Trials": group.shape[0],
            "Significant Count": (group["Significantly Different"] == "yes").sum(),
            "Proportion": (group["Significantly Different"] == "yes").sum() / group.shape[0]
        })
    ).reset_index()
    summary_dict[metric_name] = summary

# -----------------------------
# Step 4. Write two separate Excel files:
# (a) One Excel file with detailed simulation results (3 sheets, one per metric)
# (b) One Excel file with summary statistics (3 sheets, one per metric)
# -----------------------------
# Write detailed simulation results to an Excel file
detailed_output = r""
with pd.ExcelWriter(detailed_output, engine='openpyxl') as writer:
    for metric_name, df_results in results_dict.items():
        df_results.to_excel(writer, sheet_name=metric_name)
print(f"Detailed simulation results have been saved to {detailed_output}")

# Write summary statistics to another Excel file
summary_output = r""
with pd.ExcelWriter(summary_output, engine='openpyxl') as writer:
    for metric_name, summary_df in summary_dict.items():
        summary_df.to_excel(writer, sheet_name=metric_name, index=False)
print(f"Summary statistics have been saved to {summary_output}")

