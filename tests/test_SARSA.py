from matplotlib import pyplot as plt
from src.gridworld import GridWorld
from src.model_Free.SARSA import SARSAAgent
from tests.test_mc_control import print_policy_grid  

def test_sarsa_control(env=None, num_episodes=5000, gamma=0.9, alpha=0.1, epsilon=0.2, decay=True, verbose=True):
    if env is None:
        env = GridWorld(stochastic=True, stochastic_prob=0.2)

    num_actions = env.action_space
    agent = SARSAAgent(action_space=num_actions, gamma=gamma, alpha=alpha, epsilon=epsilon)
    
    returns, episode_lengths = agent.train(env, num_episodes=num_episodes, decay=decay)
    
    
    policy = agent.get_greedy_policy(terminal_states=env.terminal_states)

    if verbose:
        print("\nLearned Greedy Policy (SARSA):")
        for state in sorted(policy.keys()):
            print(f"State {state}: Best action → {policy[state]}")
        
        # Optional: Grid of arrows
        print_policy_grid(policy, grid_size=env.grid_size)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(returns)
        plt.title("Episode Return (SARSA)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")

        plt.subplot(1, 2, 2)
        plt.plot(episode_lengths)
        plt.title("Episode Length (SARSA)")
        plt.xlabel("Episode")
        plt.ylabel("Steps Taken")

        plt.tight_layout()
        plt.show()

    return agent, policy, returns, episode_lengths
