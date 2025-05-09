def policy_improvement(env, value_table, gamma =0.9):
    """
     Improve policy greedily w.r.t the current value function.

    Args:
        env: GridWorld instance
        value_table: dict mapping state -> V(s)
        gamma: discount factor

    Returns:
        new_policy: dict mapping state -> best action
    """
    policy = {}

    for state in env.states:
        if env.is_terminal(state):
            continue
        best_action = None
        best_value = float('-inf')

        for action in env.get_possible_actions(state):
             q =0
             transitions =  env.get_transition_probabilities(state, action)
             for prob, next_state, reward in transitions:
                 q+= prob *(reward + gamma *value_table[next_state])
             
             if q > best_value:
                 best_value = q
                 best_action = action

        policy[state]= best_action

    return policy