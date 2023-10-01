import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os



df_averages = pd.read_csv("Results_Tuning/averages/average_performances.csv")
df_allruns = pd.read_csv("Results_Tuning/individuals/all_performances.csv")



max_fitness_index = df_averages['Average Performance'].idxmax()
max_alpha = df_averages['alpha'][max_fitness_index]
max_fitness = df_averages['Average Performance'][max_fitness_index]

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_averages, x='alpha', y='Average Performance', color='blue')

for i in range(1,11):
    plt.scatter(df_allruns['alpha'], df_allruns[f"Run {i}"], color='red', alpha=0.7)

plt.xlim(0, 1)
plt.xticks(np.arange(0, 1.1, 0.1))

plt.ylim(0, 100)

plt.annotate(f'\u03B1: {max_alpha:.2f}', (max_alpha, 0), textcoords="offset points", xytext=(0, 10),
             ha='right', fontsize=10, color='red')

plt.plot([max_alpha, max_alpha], [max_fitness, 0], color='red', linestyle='--')

plt.xlabel('Alpha')
plt.ylabel('Fitness')
plt.title('Fitness vs. Alpha')
plt.savefig("alphavsfitness.png", dpi=300)

plt.show()