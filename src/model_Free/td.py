# this is to implement the TD learning algorithm
#TD(0)


def td_predicition(env, policy, episodes, alpha=0.1, gamma=0.9):

    V = {state: 0.0 for state in env.states}

    for ep in range(episodes):
        state = env.states[0]
        done = env.is_terminal(state)
        while not done:

            action =  policy[state]
            transitions = env.get_transition_probabilities(state, action)
            _, next_state, reward = transitions[0]

            td_target = reward + gamma * V[next_state]
            td_error = td_target - V[state]
            V[state] += alpha * td_error
            state = next_state
            done = env.is_terminal(state)

    return V



