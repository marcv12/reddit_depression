import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu
import numpy as np


print("Significance tests pre/post on control group")

df = pd.read_csv("data/predicted_dataset_probs.csv")



# Define emotions columns
emotions = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']
periods = ["pre-pandemic", "post-pandemic"]

# # Adjust the figure size as necessary
plt.figure(figsize=(18, 10))  # Make the figure slightly smaller

# Loop through the emotions and create a subplot for each
for i, emotion in enumerate(emotions, start=1):  # Start at 1 for subplot indexing
    ax = plt.subplot(2, 4, i)  # Adjust the grid size as necessary (2 rows, 4 columns here)
    sns.boxplot(x='pandemic_period', y=emotion, data=df)
    ax.set_title(emotion.capitalize())
    ax.set_ylabel('')  # Remove the y-axis label
    ax.set_xlabel('')  # Remove the x-axis label "pandemic period"
    # Custom x-tick labels
    ax.set_xticklabels(['Pre-pandemic', 'Post-pandemic'])

    plt.tight_layout(pad=2.0)  # Adjust spacing between subplots to prevent overlap

plt.show()

#Let's first check for normality of our data to see which test to apply (parametric vs non-parametric)
#Dictionary to hold p-values for each emotion and period
p_values = {"pre-pandemic": {}, "post-pandemic": {}}

print("Normality test, Shapiro-Wilk: \n")
for emotion in emotions:
    for period in periods:
        #Select the scores for the given emotion and period
        scores = df[df["pandemic_period"] == period][emotion]

        #Perform the Schapiro-Wilk test (normality test)
        stat, p = stats.shapiro(scores)

        # Save the p-value in the dictionary
        p_values[period][emotion] = p

        # Output the results
        print(f"{emotion.capitalize()} ({period}): Statistics={stat:.3f}, p-value={p:.3f}")


#None of the data is normaly distributed, so we proceed with a non-parametric test, namely Mann-Whitney U test
        
results = []

for emotion in emotions:
    # Extract scores for pre-pandemic and post-pandemic
    pre_scores = df[(df['pandemic_period'] == 'pre-pandemic')][emotion]
    post_scores = df[(df['pandemic_period'] == 'post-pandemic')][emotion]
    
    # Conduct the Mann-Whitney U test
    stat, p = mannwhitneyu(pre_scores, post_scores)
    
    # Store the results
    results.append((emotion, stat, p))

print("\n\nMann-Whitney-U test pre/post experimental: \n")
# Print the results
for emotion, stat, p in results:
    print(f"{emotion}: U-statistic={stat:.3f}, p-value={p:.4f}")

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))  # Adjust as needed
axs = axs.ravel()  # Flatten the array for easy indexing




print("\n\nMean \n")


# Calculate the means
pre_means = [round(df[df['pandemic_period'] == 'pre-pandemic'][emotion].mean(), 2) for emotion in emotions]
post_means = [round(df[df['pandemic_period'] == 'post-pandemic'][emotion].mean(), 2) for emotion in emotions]

# Set up the bar chart
x = np.arange(len(emotions))  # the label locations
width = 0.35  # the width of the bars

colors = ['#377eb8', '#ff7f00']  # Blue, Orange

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, pre_means, width, label='Pre-pandemic', color=colors[0])
rects2 = ax.bar(x + width/2, post_means, width, label='Post-pandemic', color=colors[1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean')
ax.set_title('Mean emotion scores pre-pandemic vs post-pandemic')
ax.set_xticks(x)
ax.set_xticklabels(emotions, rotation=45)
ax.legend()

# Attach a text label above each bar in *rects*, displaying its height.
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

fig.tight_layout()
plt.show()