import numpy as np
import matplotlib.pyplot as plt
from CovidDataset import load_data

def plot_graph(X, y):
    x_ticks = ["anger","brain dysfunction (forget)","emptiness","hopelessness","loneliness","sadness","suicide intent","worthlessness"]
    print(y)
    for era, label in [(0, "Post-Pandemic"),(1, "Pre-Pandemic")]:
        plt.subplot(1, 2, 1)
        era_idx = np.argwhere(y == era).flatten()
        X_era = X[era_idx]
        prob_mean = np.mean(X_era, axis = 0)
        plt.bar(x_ticks, prob_mean)

        for j in range(len(x_ticks)):
            plt.text(j - 0.1,prob_mean[j] + 0.001,round(prob_mean[j], 3))
            
        plt.ylabel("Probability")
        plt.xlabel("Emotions")
        plt.xticks(rotation=45, ha='right')

        plt.subplot(1, 2, 2)
        
        plt.boxplot( X_era,labels=x_ticks)
        plt.ylabel("Probability")
        plt.xlabel("Emotions")
        plt.xticks(rotation=45, ha='right')

        plt.suptitle(f"Depression {label} Posts Analysis")
        plt.show()

X, y = load_data("experimental") 
plot_graph(X, y)