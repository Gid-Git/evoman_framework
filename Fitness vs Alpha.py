import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = {
    'alpha': [0.443369574, 0.501800865, 0.77161859, 0.360233636, 0.622359959, 0.28382617,
              0.098785591, 0.143007111, 0.843879534, 0.94463396, 0.99, 0.99, 0.976685263,
              0.961823656, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    'Fitness': [63.7090774, 58.49942274, 80.4594627, 60.92361677, 69.58009797, 55.93347776,
                35.84872157, 49.00222219, 87.01861416, 91.50406559, 94.09429896, 94.22599583,
                93.39282769, 92.30792229, 94.01159629, 93.89793627, 94.05446996, 93.62179903,
                94.05426611, 93.97760208]
}

df = pd.DataFrame(data)

max_fitness_index = df['Fitness'].idxmax()
max_alpha = df['alpha'][max_fitness_index]
max_fitness = df['Fitness'][max_fitness_index]

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='alpha', y='Fitness', color='blue')

plt.xlim(0, 1)
plt.xticks(np.arange(0, 1.1, 0.1))

plt.ylim(df['Fitness'].min(),100)

plt.annotate(f'\u03B1: {max_alpha:.2f}', (max_alpha, df['Fitness'].min()), textcoords="offset points", xytext=(0, 10),
             ha='right', fontsize=10, color='red')

plt.plot([max_alpha, max_alpha], [max_fitness, df['Fitness'].min()], color='red', linestyle='--')

plt.xlabel('Alpha')
plt.ylabel('Fitness')
plt.title('Fitness vs. Alpha')
plt.savefig("alphavsfitness.png", dpi=300)

plt.show()
