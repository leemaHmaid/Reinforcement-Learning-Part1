from tests.test_q_learning import test_q_learning
from matplotlib import pyplot as plt
from src.gridworld import GridWorld
import numpy as np

def moving_average(x, window_size=100):
    return np.convolve(x, np.ones(window_size) / window_size, mode='valid')

env = GridWorld(stochastic=True, stochastic_prob=0.2)
epsilon_values = [0.1, 0.2, 0.3]
alpha_values = [0.05, 0.1, 0.5]

plt.figure(figsize=(10, 6))

for epsilon in epsilon_values:
        for alpha in alpha_values:
            label = f"ε={epsilon}, α={alpha}"
            _, _, returns, _ = test_q_learning(
                env=env,
                num_episodes=5000,
                gamma=0.9,
                epsilon=epsilon,
                alpha=alpha,
                decay=True,
                verbose=False
            )

            smoothed_returns = moving_average(returns, 100)
            plt.plot(smoothed_returns, label=label)


plt.title("Q-learning: Episode Return for Different ε and α")
plt.xlabel("Episode")
plt.ylabel("Smoothed Total Reward")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
