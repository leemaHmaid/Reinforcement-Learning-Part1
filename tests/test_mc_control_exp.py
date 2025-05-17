import os
from test_mc_control import test_monte_carlo_control
from matplotlib import pyplot as plt
from src.gridworld import GridWorld

import numpy as np

# Optional smoothing
def moving_average(x, window_size=100):
    return np.convolve(x, np.ones(window_size) / window_size, mode='valid')

env = GridWorld()
epsilon_values = [0.1, 0.2, 0.3]
alpha_values = [0.05, 0.1, 0.5]

plt.figure(figsize=(10, 6))

for epsilon in epsilon_values:
        label = f"ε={epsilon}"
        _, _, returns, _ = test_monte_carlo_control(
            env=env,
            num_episodes=5000,
            gamma=0.9,
         
            epsilon=epsilon,
            decay=True,
            verbose=False  # Turn off printing + plot
        )

        smoothed_returns = moving_average(returns, 100)
        plt.plot(smoothed_returns, label=label)

plt.title("MC_Control: Episode Return for Different ε ")
plt.xlabel("Episode")
plt.ylabel("Smoothed Total Reward")
plt.legend()
plt.grid()
plt.tight_layout()
 

os.makedirs("results", exist_ok=True)
plt.savefig("results/mc_control_returns.png")   
plt.show()
