def policy_evaluation_No_Discount(env, policy , gamma= 1.0 , theta = 1e-2):
    """
    Iterative policy evaluation using the Bellman expectation equation.

    Args:
        env: instance of GridWorld
        policy: dict mapping state to action
        gamma: discount factor
        theta: small threshold for stopping (convergence)

    Returns:
        value_table: dict mapping state -> value
    """
    # Initialize V(s) = 0 for all states
    value_table = {state:0.0 for state in env.states}

    iteration = 0
    while True:
        delta = 0
        new_value_table =  value_table.copy()

        for state in env.states:
            if env.is_terminal(state):
                continue
            action = policy[state]
            transitions = env.get_transition_probabilities(state, action)
            value = 0.0

            for prob, next_state, reward in transitions:
                value += prob * (reward +gamma * new_value_table[next_state])

            delta = max (delta , abs(value- new_value_table[state]))
            new_value_table[state] = value
        
        iteration += 1
        print(f"Iteration {iteration}, delta = {delta:.5f}")

        value_table = new_value_table
        if delta < theta:
            break
    # print(f"Old: {value_table[state]:.3f}, New: {value:.3f}")

    return value_table
