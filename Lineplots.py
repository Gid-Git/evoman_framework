import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

avg_health_gains = []
best_health_gains = []

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
enemies = [4, 6, 7]
EA = ["with_fitness_", "withouth_fitness_"]

ea_row_mapping = {"with_fitness_": 0, "withouth_fitness_": 1}

y_min = -20
y_max = 130

for t in EA:  
    for j, enemy in enumerate(enemies):
        concatenated_df = pd.DataFrame()
        for i in range(1, 11):
            dir_name = f"resulsEAStest/{t}{enemy}_{i}"
            file_path = os.path.join(dir_name, "results.csv")

            df = pd.read_csv(file_path)

            concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)

        grouped_by_generation = concatenated_df.groupby("Generation")

        best_fitness_means = grouped_by_generation["Best Fitness"].mean()
        avg_fitness_means = grouped_by_generation["Average Fitness"].mean()

        upper_best_bounds_1sigma = best_fitness_means + grouped_by_generation["Std Fitness"].mean()
        lower_best_bounds_1sigma = best_fitness_means - grouped_by_generation["Std Fitness"].mean()

        upper_best_bounds_2sigma = best_fitness_means + 2 * grouped_by_generation["Std Fitness"].mean()
        lower_best_bounds_2sigma = best_fitness_means - 2 * grouped_by_generation["Std Fitness"].mean()

        row = ea_row_mapping[t]
        col = j  
        ax = axes[row, col]

        ax.plot(
            best_fitness_means,
            label="Best Fitness",
            color="orange",  
            linestyle='-'
        )
        ax.plot(
            avg_fitness_means,
            label="Average Fitness",
            color="blue",  
            linestyle='-'
        )
        ax.fill_between(
            range(len(best_fitness_means)),
            lower_best_bounds_1sigma,
            upper_best_bounds_1sigma,
            color="lightblue",  
            alpha=0.5  
        )
        ax.fill_between(
            range(len(best_fitness_means)),
            lower_best_bounds_2sigma,
            upper_best_bounds_2sigma,
            color="lightblue",  
            alpha=0.3  
        )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        
        if t == "with_fitness_":
            ax.set_title(f"EA with Optimized Fitness Function - Enemy {enemy}")
        else:
            ax.set_title(f"EA without Optimized Fitness - Enemy {enemy}")
        
        ax.legend()
        
        ax.set_ylim(y_min, y_max)

plt.tight_layout()

plt.savefig("lineplots.png", dpi=300)

plt.show()