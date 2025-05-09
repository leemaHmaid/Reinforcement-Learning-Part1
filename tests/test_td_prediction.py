import random
from src.gridworld import GridWorld
from src.model_Free.td0 import td_predicition
import matplotlib.pyplot as plt
import numpy as np

from src.utils import plot_value_heatmap, print_value_function
 

def test_td():
    def random_policy(state):
        return random.choice(env.get_possible_actions(state))

    env = GridWorld()
    V = td_predicition(env, random_policy, episodes=1000, alpha=0.5, gamma=0.9)

    print("Estimated Value Function:")
    print_value_function(V, env.grid_size)
    plot_value_heatmap(V, env.grid_size)
  