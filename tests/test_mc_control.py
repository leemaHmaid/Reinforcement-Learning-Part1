from src.model_Free.monte_carlo_control import MonteCarloControl
def print_policy_grid(policy, grid_size=10):
    arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→', None: '⛳'}
    grid = [['⛳' for _ in range(grid_size)] for _ in range(grid_size)]

    for (i, j), action in policy.items():
        grid[i][j] = arrow_map[action]

    print("\nLearned Policy Grid:\n")
    for row in grid:
        print(' '.join(row))

action_map = ['↑', '↓', '←', '→']

def print_q_values(agent, selected_states=None):
    print("\nQ-Value Table:")
    for state in sorted(agent.Q.keys()):
        if selected_states and state not in selected_states:
            continue
        q_values = agent.Q[state]
        q_str = ' | '.join(f"{action_map[i]}: {q:.2f}" for i, q in enumerate(q_values))
        print(f"State {state} → {q_str}")



def test_monte_carlo_control(env, num_episodes=5000, gamma=0.9, epsilon=0.2, decay=True, verbose=False):
    num_actions = env.action_space
    agent = MonteCarloControl(action_space=num_actions, gamma=gamma, epsilon=epsilon)
    
    agent.train(env, num_episodes=num_episodes, decay=decay)

    policy = agent.get_greedy_policy()

    print_policy_grid(policy, grid_size=env.grid_size)
    # print("\nLearned Greedy Policy:")
    # for state in sorted(policy.keys()):
    #     print(f"State {state}: Best action → {policy[state]}")
    print_q_values(agent, selected_states=[(0,0), (5,5), (9,8)])


    return agent, policy
