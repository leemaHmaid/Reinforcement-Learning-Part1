from .policy_evaluation import policy_evaluation
from .policy_improvement import policy_improvement

def policy_iteration(env, gamma=0.9, theta=1e-2):

    """
    Performs full policy iteration:
    - Initialize a random policy
    - Evaluate and improve repeatedly until policy converges
    Returns:
    - optimal_policy: dict mapping state -> action
    - value_table: dict mapping state -> value
    """

    policy ={}
    for state in env.states:
       if env.is_terminal(state):
           continue
       actions = env.get_possible_actions(state) 
       if actions:
           policy[state] = actions[0] 

    while True:

        value_table = policy_evaluation(env, policy, gamma=gamma, theta=theta) 
        new_policy = policy_improvement(env, value_table, gamma=gamma)   
        if new_policy == policy:
            break
        policy = new_policy
    return policy, value_table 
    