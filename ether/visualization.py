import pandas as pd
import matplotlib.pyplot as plt

# Load the data
topsis_data = pd.read_csv('results/topsis.csv')
random_data = pd.read_csv('results/random.csv')

# Calculate means
mean_topsis_latency = topsis_data['Latency'].mean()
mean_random_latency = random_data['Latency'].mean()
mean_topsis_cell_cost = topsis_data['Cell Cost'].mean()
mean_random_cell_cost = random_data['Cell Cost'].mean()

# Sort data
sorted_topsis_latency = topsis_data['Latency'].sort_values().reset_index()
sorted_random_latency = random_data['Latency'].sort_values().reset_index()
sorted_topsis_cell_cost = topsis_data['Cell Cost'].sort_values().reset_index()
sorted_random_cell_cost = random_data['Cell Cost'].sort_values().reset_index()

# Set global font size
plt.rcParams['font.size'] = 16

# Set up the figure and subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

# Latency Line Plot with Flipped Axes
axes[0].plot(sorted_topsis_latency['Latency'], sorted_topsis_latency.index, label='PolyNet', color='#1f77b4')
axes[0].plot(sorted_random_latency['Latency'], sorted_random_latency.index, label='Baseline', color='#ff7f0e')
axes[0].axvline(x=mean_topsis_latency, color='#1f77b4', linestyle='dashed', linewidth=1, label=f'PolyNet Mean Latency: {mean_topsis_latency:.1f}')
axes[0].axvline(x=mean_random_latency, color='#ff7f0e', linestyle='dashed', linewidth=1, label=f'Baseline Mean Latency: {mean_random_latency:.1f}')
axes[0].set_ylabel('Data Point Index')
axes[0].set_xlabel('Latency (ms)')
axes[0].set_title('Latency')
axes[0].legend(loc='lower right')

# Cell Cost Line Plot with Flipped Axes
axes[1].plot(sorted_topsis_cell_cost['Cell Cost'], sorted_topsis_cell_cost.index, label='PolyNet', color='#2ca02c')
axes[1].plot(sorted_random_cell_cost['Cell Cost'], sorted_random_cell_cost.index, label='Baseline', color='#d62728')
axes[1].axvline(x=mean_topsis_cell_cost, color='#2ca02c', linestyle='dashed', linewidth=1, label=f'PolyNet Mean Cell Cost Index: {mean_topsis_cell_cost:.1f}')
axes[1].axvline(x=mean_random_cell_cost, color='#d62728', linestyle='dashed', linewidth=1, label=f'Baseline Mean Cell Cost Index: {mean_random_cell_cost:.1f}')
axes[1].set_ylabel('Data Point Index')
axes[1].set_xlabel('Cell Cost Index')
axes[1].set_title('Cell Cost')
axes[1].legend(loc='lower right')

# Adjust layout and save
plt.tight_layout()
plt.savefig('results/exp.png')
plt.show()
