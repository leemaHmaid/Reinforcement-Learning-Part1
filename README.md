# 🧠 Reinforcement Learning from Scratch (Part1)

Hello everyone! I am a Machine learning master's student curious about learning RL.
This project is my personal journey into the foundations of Reinforcement Learning (RL) as self-taught learner.  
I implemented key RL algorithms from scratch - no external libraries, just NumPy and pure curiosity.

Along the way, I experimented and visualized  
*“How does exploration affect learning?”*  
*“What happens when environments become stochastic?”*  


---

## 🌟 What You'll Find Here
- This repo is RL-part1 in a trial to cover the main concepts in the first 5 lectures of [David Silver’s RL Course](https://www.davidsilver.uk/teaching/):

- A custom GridWorld environment (with ⛳ terminal states and arrow policies)
- Step-by-step implementations of:
  - Dynamic Programming (Policy Eval, Iteration, Value Iteration)
  - Monte Carlo and TD(0) & TD (lambda) for Prediction
  - Monte Carlo Control (First-Visit, ε-greedy)
  - Temporal Difference (SARSA, Q-learning)
- Smoothed reward plots and learning comparisons
- Support for **stochastic environments**
- Parameter studies on **ε and α** — and how they affect learning

---

## 🗂️ Folder Structure

**`src/`** – Core implementation of agents and environment  
├── `gridworld.py` – Custom GridWorld environment  
├── `dynamic_programming/` – DP methods: policy/value iteration  
├── `model_Free/` – MC, TD(prediction), SARSA, Q-learning agents  

**` tests/`** – Test scripts to run agents and visualize results  
├── `test_sarsa_control.py`  
├── `test_mc_control.py`  
├── `test_q_learning.py`  
└── `...other tests`

**`results/`** – Learning curves, experiment comparisons  
├── `sarsa_vs_q_learning.png`  
└── `exploration_effects.png`  

**`main.py`** – CLI entry point to run agents  
**`README.md`** – You are here :)

## 📊 Sample Results

You'll find full plots and experiment writeups in the [`/results`](./results) folder.

Here’s a preview:

<img src="plots/sarsa_vs_q_learning.png" width="600">

---

## 🧪 Run the Agents

```bash
# Train SARSA agent
python main.py --test sarsa

# Run Monte Carlo control
python main.py --test mc_control

# Train Q-learning agent
python main.py --test q_learning


```
---
## 🙏 Credits

This project is heavily inspired by the brilliant [**David Silver’s Reinforcement Learning Course**](https://www.davidsilver.uk/teaching/), which laid the foundation for my understanding of RL (I highly recommend it if you are beginning your journey into RL don't be scared of the math!).
