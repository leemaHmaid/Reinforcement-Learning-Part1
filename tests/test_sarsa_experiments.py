from test_SARSA import test_sarsa_control
from matplotlib import pyplot as plt
from src.gridworld import GridWorld

import numpy as np

# Optional smoothing
def moving_average(x, window_size=100):
    return np.convolve(x, np.ones(window_size) / window_size, mode='valid')

env = GridWorld(stochastic=True, stochastic_prob=0.2)
epsilon_values = [0.1, 0.2, 0.3]
alpha_values = [0.05, 0.1, 0.5]

plt.figure(figsize=(10, 6))

for epsilon in epsilon_values:
    for alpha in alpha_values:
        label = f"ε={epsilon}, α={alpha}"
        _, _, returns, _ = test_sarsa_control(
            env=env,
            num_episodes=5000,
            gamma=0.9,
            alpha=alpha,
            epsilon=epsilon,
            decay=True,
            verbose=False  # Turn off printing + plot
        )

        smoothed_returns = moving_average(returns, 100)
        plt.plot(smoothed_returns, label=label)

plt.title("SARSA: Episode Return for Different ε and α")
plt.xlabel("Episode")
plt.ylabel("Smoothed Total Reward")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
