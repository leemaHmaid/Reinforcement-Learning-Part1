# Reinforcement Learning: GridWorld Algorithms 🧠🌍

This repository implements foundational **Reinforcement Learning algorithms** using a custom-built GridWorld environment, inspired by **David Silver’s RL Course**.

✅ Built from scratch  
✅ No external RL libraries  
✅ Designed for learning & clarity

---

## 🔁 Algorithms Implemented

### 🔹 Dynamic Programming (Model-Based)
> Requires full knowledge of the MDP

- Policy Evaluation
- Policy Improvement
- Policy Iteration
- Value Iteration
- Optimal Policy Extraction

### 🔹 Monte Carlo (Model-Free, Episodic)
> Learns from complete sampled episodes

- First-Visit MC Control
- ε-greedy policy improvement
- Episode sampling + return averaging

### 🔹 Temporal Difference (TD) Learning
> Learns online from bootstrapped estimates

- **SARSA** (on-policy TD control)
- ε-greedy with decay (GLIE)
- Step-by-step updates after every action

### 🔹 Q-learning (off-policy TD control) — *coming soon*
> Learns from greedy future actions even if not taken

---

## 🗺️ Environment: GridWorld

- Custom grid of configurable size (default: 10x10)
- Deterministic transitions
- Custom terminal states
- Visual policy rendering with arrows
- Easy to extend with stochasticity or walls

---

## 🖼️ Features

- Modular agent design (`MC`, `SARSA`, `DP`, `Q-learning`)
- ε-greedy exploration with optional decay
- Pretty-print policy arrows (↑ ↓ ← →)
- Terminal state masking (⛳)
- Test scripts under `/tests/` for quick runs
- Designed to help you visualize learning

## 📚 Educational Use

This repo is part of an ongoing reinforcement learning learning journey  
by **Leema Hamid** — from GridWorld all the way to deep RL.

🌟 It will soon be accompanied by a **Medium article series** covering:

- Monte Carlo Control  
- SARSA  
- Q-learning (and beyond)

---

## 👩‍💻 About This Project

I'm a self-taught reinforcement learning enthusiast, currently deepening my understanding through hands-on projects like this. Everything in this repo was implemented from scratch as part of my personal learning journey — no external RL libraries, just logic, math, and curiosity.

I hope this project helps others who are learning RL the hard (and fun) way.

---

## 🧠 Credits

- Inspired by [David Silver’s RL Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)  
- All algorithms implemented manually for understanding
---

## 🚀 Run Examples

```bash
python main.py --test dp
python main.py --test mc_control
python main.py --test sarsa





