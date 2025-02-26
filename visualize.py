import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


with open('log.csv', mode='r') as file:
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
plt.title('Performance Heatmap (Alpha vs. Gamma)')
plt.xlabel('Gamma')
plt.ylabel('Alpha')
plt.tight_layout()
plt.savefig('heatmap_performance.png')
plt.show()

# Option 2: 3D Surface plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for the surface plot
alphas = sorted(df['alpha'].unique())
gammas = sorted(df['gamma'].unique())
X, Y = np.meshgrid(gammas, alphas)

# Create Z values (performance)
Z = np.zeros((len(alphas), len(gammas)))
for i, alpha in enumerate(alphas):
    for j, gamma in enumerate(gammas):
        matching = df[(df['alpha'] == alpha) & (df['gamma'] == gamma)]
        if not matching.empty:
            Z[i, j] = matching['perf'].values[0]

# Create the surface plot
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Performance')

# Set labels and title
ax.set_xlabel('Gamma')
ax.set_ylabel('Alpha')
ax.set_zlabel('Performance')
ax.set_title('3D Surface Plot of Performance by Alpha and Gamma')

# View adjustment for better visualization
ax.view_init(elev=30, azim=45)
plt.savefig('3d_performance.png')
plt.show()

# Option 3: Scatter plot with color representing performance
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['gamma'], df['alpha'], c=df['perf'], s=100, cmap='viridis', alpha=0.8)

# Add color bar
cbar = plt.colorbar(scatter)
cbar.set_label('Performance')

# Set labels and title
plt.title('Performance by Alpha and Gamma')
plt.xlabel('Gamma')
plt.ylabel('Alpha')

# Add contour lines (optional)
pivot_table = df.pivot_table(index='alpha', columns='gamma', values='perf', aggfunc='mean')
contour_x = np.array(pivot_table.columns)
contour_y = np.array(pivot_table.index)
contour_z = pivot_table.values
plt.contour(contour_x, contour_y, contour_z, colors='black', alpha=0.4)

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('scatter_performance.png')
plt.show()


# That clarifies things! With only 500 episodes for a 20x20 grid, your agent doesn't have enough time to fully propagate values across the entire state space with higher gamma values.
# Here's what's likely happening:

# With low gamma (0.001-0.3), the agent focuses on immediate or near-immediate rewards. In a pathfinding context, this creates a "greedy" behavior that works well for simple paths.
# Higher gamma values (0.5+) require more training episodes to be effective. With just 500 episodes, the agent hasn't had enough time to properly propagate the distant reward signals backward through the state space.
# The 20x20 grid has 400 states, and finding an optimal path requires value information to propagate across many steps. With high gamma, this propagation is slower to stabilize without enough training episodes.
# The performance cliff around gamma 0.4 likely represents the point where the agent starts trying to account for long-term rewards but hasn't had enough episodes to make those estimates reliable.

# This isn't necessarily a flaw - it's showing you exactly how these parameters behave with limited training. If your goal is to intentionally throttle performance to see some failures, then your current setup is achieving that.
# If you were to increase the episode count significantly (perhaps to 5,000 or 10,000), you'd likely see better performance with higher gamma values as the agent would have time to properly propagate the reward signals throughout the state space.