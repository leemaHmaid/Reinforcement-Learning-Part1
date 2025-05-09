import random
from src.gridworld import GridWorld
from src.model_Free.td import td_prediction
from main import print_policy, print_value_table
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
    plt.show()

def test_td():
    def random_policy(state):
        return random.choice(env.get_possible_actions(state))

    env = GridWorld()
    V = td_prediction(env, random_policy, episodes=1000, alpha=0.1, gamma=0.9)

    print("Estimated Value Function:")
    print_value_function(V, env.grid_size)
    plot_value_heatmap(V, env.grid_size)
  