from src.env import GridWorld

def main():
    env = GridWorld()

    print("States:", env.states)

    state = (1,2)
    print("Actions:", env.get_possible_actions(state))

    for action in env.acrtions:
        transitions = env.get_transition_probabilities(state, action)
        print(f"Action: {action}")
        for prob, next_state, reward in transitions:
            print(f"  Transition: {prob}, {next_state}, {reward}")

if __name__ == "__main__":
    main()