import matplotlib.pyplot as plt
import numpy as np

def print_value_function(V, grid_size):
    for i in range(grid_size):
        row = ""
        for j in range(grid_size):
            val = V.get((i, j), 0.0)
            row += f"{val:6.2f} "
        print(row)
    print()

def plot_value_heatmap(V, grid_size):
    grid = np.zeros((grid_size, grid_size))
    for (i, j), v in V.items():
        grid[i][j] = v

    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title("Value Function Heatmap")

    # Optional: overlay values as text
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, f"{grid[i,j]:.1f}", ha='center', va='center', color='white')

    plt.show()
