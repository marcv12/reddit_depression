import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
import argparse


# Argument parser
parser = argparse.ArgumentParser(description="Perform statistical tests on a dataset")
parser.add_argument("file_path", type=str, help="Path to the dataset file")
args = parser.parse_args()



df = pd.read_csv(args.file_path)

# Determine group from file path
group = "experimental" if "predicted_dataset.csv" in args.file_path else "control"

print(f"Significance tests pre/post on {group} group")

emotions = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']
emotions_label = ['anger', 'forgetfulness', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']

# Results placeholder
mann_whitney_results = []
chi_square_results = []

for emotion in emotions:
    # Mann-Whitney U test
    pre_scores = df[df['pandemic_period'] == 'pre-pandemic'][emotion]
    post_scores = df[df['pandemic_period'] == 'post-pandemic'][emotion]
    mw_stat, mw_p = mannwhitneyu(pre_scores, post_scores)
    mann_whitney_results.append((emotion, mw_stat, mw_p))
    
    # Chi-square or Fisher's Exact Test
    contingency_table = pd.crosstab(df['pandemic_period'], df[emotion])
    if contingency_table.min().min() < 5:  # If any cell has an expected count less than 5
        # Fisher's Exact Test
        _, fisher_p = fisher_exact(contingency_table)
        chi_square_results.append((emotion, 'Fisher', fisher_p))
    else:
        # Chi-square Test
        _, chi_p, _, _ = chi2_contingency(contingency_table)
        chi_square_results.append((emotion, 'Chi-square', chi_p))

# Display the results
print("Mann-Whitney U test results:")
for emotion, stat, p in mann_whitney_results:
    print(f"{emotion}: U-statistic={stat:.3f}, p-value={p:.4f}")

print("\nChi-square/Fisher's Exact Test results:")
for emotion, test, p in chi_square_results:
    print(f"{emotion}: Test={test}, p-value={p:.4f}")

# Mean proportions
pre_means = df[df['pandemic_period'] == 'pre-pandemic'][emotions].mean()
post_means = df[df['pandemic_period'] == 'post-pandemic'][emotions].mean()

# Visualization
fig, ax = plt.subplots(figsize=(14, 10))
x = np.arange(len(emotions))
width = 0.35
ax.bar(x - width/2, pre_means, width, label='Pre-pandemic')
ax.bar(x + width/2, post_means, width, label='Post-pandemic')
ax.set_ylabel('Proportion')
#ax.set_title(f'Proportion of Posts Expressing Each Emotion by Pandemic Period for {group} Group')
ax.set_xticks(x)
ax.set_xticklabels(emotions_label, rotation=45, ha="right", fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()




