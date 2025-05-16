import numpy as np
from collections import defaultdict
import random


class MonteCarloControl:
    def __init__(self, action_space, gamma=0.8, epsilon = 1.0):
        self.Q = defaultdict(lambda: np.zeros(action_space))
        self.return_sum = defaultdict(float)
        self.return_count = defaultdict(float)
        self.gamma = gamma
        self.epsilon = max(0.01, epsilon * 0.97)
        self.action_space = action_space
    
    def get_greedy_policy(self):
        policy = {}
        for state in self.Q:
            policy[state] = int(np.argmax(self.Q[state]))
        return policy


    def epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            return int(np.argmax(self.Q[state]))
        
    # Generate an episode using the policy ---> [(state, action, reward),...]

    def generate_episode(self, env):
        episode =[]
        state = env.reset()
        done = False
        while not done:
            action = self.epsilon_greedy_action(state)
            next_state, reward, done, _ = env.step_control(action)
            episode.append((state, action, reward))
            state = next_state
        return episode
    
    # Update Q-values first visit:

    def update_q_first_visit(self, episode):
        G = 0
        visited =set()

        for t in reversed(range(len(episode))):

            state, action, reward = episode[t]
            G = self.gamma*G +reward

            if( state, action) not in visited:
                visited.add((state, action))

                self.return_sum[(state, action)] += G
                self.return_count[(state, action)] += 1


                self.Q[state][action] = self.return_sum[(state, action)] / self.return_count[(state, action)]


    # train the agent using the Monte Carlo control algorithm
    def train(self, env, num_episodes, decay=False):
        returns_per_episode = []
        episode_lengths = []

        for episode_num in range(1, num_episodes + 1):
            episode = self.generate_episode(env)
            self.update_q_first_visit(episode)

            # Track total return and steps
            G = 0
            for i, (_, _, reward) in enumerate(reversed(episode)):
                G = self.gamma * G + reward
            returns_per_episode.append(G)
            episode_lengths.append(len(episode))

            # Decay epsilon
            if decay:
                self.epsilon = max(0.01, self.epsilon * 0.99)

            if episode_num % 1000 == 0:
                print(f"Episode {episode_num}/{num_episodes} completed.")

        print("Training completed.")
        return returns_per_episode, episode_lengths

        



    


