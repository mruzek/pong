import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def make_chart(name):
    with open(f"{name}.csv", mode='r') as file:
        csv_reader = csv.DictReader(file)
        data = [row for row in csv_reader]
    #print(data)

    # Clean and convert data to correct types
    df = pd.DataFrame(data)
    df.columns = [col.strip() for col in df.columns]  # Remove leading/trailing spaces in column names
    df['alpha'] = df['alpha'].astype(float)
    df['gamma'] = df['gamma'].astype(float)
    df['perf'] = df['perf'].astype(float)

    # Option 1: Heatmap visualization
    plt.figure(figsize=(10, 8))
    # Create a pivot table for the heatmap
    pivot_table = df.pivot_table(index='alpha', columns='gamma', values='perf', aggfunc='mean')
    ax = sns.heatmap(pivot_table, annot=True, cmap='viridis', cbar_kws={'label': 'Performance'})
    plt.title('Q Learning ')
    plt.xlabel('Gamma')
    plt.ylabel('Alpha')
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.show()
