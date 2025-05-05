class GridWorld:
    def __init__(self, grid_size =4 , terminal_states= [(0,0), (3,3)], default_reward = -1):
        self.grid_size = grid_size
        self.states = [(i,j) for i in range(grid_size) for j in range(grid_size)]
        self.terminal_states = terminal_states
        self.actions = ['up', 'down', 'left', 'right']
        self.default_reward = default_reward
    
    def is_terminal(self, state ):
        return state in self.terminal_states
    
    def get_next_state(self, state, action):

        if self.is_terminal(state):
            return state
        i, j = state
        if action == "up":
            i = max(0, i-1)
        if action == "down":
            i = min(i+1, self.grid_size-1)
        if action == "right":
            j =  min(j+1, self.grid_size-1)
        if action == "left":
            j = max(0, j-1)

        return (i,j)
    
    def get_transition_probabilities(self, state, action):

        """returns list of (prob, next_state, reward) tuples"""
        if self.is_terminal(state):
            return [(1.0, state , 0)]
        next_state = self.get_next_state(state, action)
        return [(1.0, next_state, self.default_reward)]
    
    def get_possible_actions (self, state):
        if self.is_terminal(state):
            return []
        return self.actions