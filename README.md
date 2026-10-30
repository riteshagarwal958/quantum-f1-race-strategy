# 🏎️ F1 Race Strategy Optimization using Deep Reinforcement Learning and Quantum Reinforcement Learning

## Overview

This project presents an intelligent Formula 1 race strategy optimization system that leverages both **Deep Reinforcement Learning (DQN)** and **Quantum Reinforcement Learning (QRL)** to make pit stop decisions. Historical Formula 1 race data is collected using the **FastF1 API**, processed into a custom **Gymnasium** environment, and used to train reinforcement learning agents capable of learning effective race strategies.

The trained agents are evaluated on unseen Formula 1 races, compared with the historical race strategy, and visualized through an interactive dashboard.

---

## Features

- Historical Formula 1 race data collection using FastF1
- Automated race data generation and preprocessing
- Custom Gymnasium environment for race strategy simulation
- Deep Reinforcement Learning using Actor-Critic Architecture and trained using PPO
- Quantum Reinforcement Learning using PennyLane and PyTorch
- Hyperparameter optimization using Optuna
- Multi-track and multi-season training
- Evaluation on unseen race scenarios
- Interactive dashboard for strategy visualization
- Strategy comparison between:
  - Historical Strategy
  - Deep Reinforcement Learning (DQN)
  - Quantum Reinforcement Learning (QRL)

---

## Project Structure

```text
Project/
│
├── data/                          # Processed race scenarios
├── f1_cache/                      # FastF1 cache
├── models/                        # Trained DQN and QRL models
│
├── generate_data.py               # Generate race scenarios from FastF1
├── data_loader.py                 # Data loading and preprocessing utilities
├── f1_strategy_env.py             # Custom Gymnasium environment
│
├── train_dqn.py                   # Train the DQN agent
├── evaluate_dqn.py                # Evaluate the trained DQN agent
│
├── train_qrl.py                   # Train the Quantum RL agent
├── evaluate_qrl.py                # Evaluate the trained Quantum RL agent
│
├── compare_agents.py              # Compare Historical, DQN and QRL strategies
├── dashboard.py                   # Interactive dashboard for visualization
│
├── requirements.txt               # Project dependencies
└── README.md
```

---

## Observation Space

The environment consists of **13 normalized features** describing the current race state, including:

- Remaining laps
- Tyre compound
- Tyre age
- Lap pace difference
- Current position
- Gap to leader
- Safety car status
- Pit lane time loss
- Tyre degradation
- Compound performance characteristics
- Tyre cliff age
- Field position information
- Additional race state features

---

## Action Space

| Action | Description |
|:------:|-------------|
| 0 | Stay Out |
| 1 | Pit for Soft Tyres |
| 2 | Pit for Medium Tyres |
| 3 | Pit for Hard Tyres |

---

## Reward Function

The reward function encourages efficient race strategies by considering:

- Competitive race pace
- Effective pit stop timing
- Tyre management
- Finishing position
- Strategic decision making

---

## Dataset

Race data is collected using the **FastF1 API**.

### Driver

- Lewis Hamilton (HAM)

### Training Tracks

- Silverstone
- Monza
- Monaco

### Training Seasons

- 2022
- 2023
- 2024

### Testing Season

- 2025

---

## Models

### Deep Reinforcement Learning (DQN)

- **Algorithm:** Deep Q-Network (DQN)
- **Framework:** Stable-Baselines3

Saved model:

```text
models/dqn_f1_strategist_final.zip
```

### Quantum Reinforcement Learning (QRL)

- **Frameworks:** Qiskit + PyTorch
- **Architecture:** Hybrid Quantum-Classical PPO Actor-Critic

Saved model:

```text
models/qrl_f1_strategist_final.pt
```

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Generate Race Data

Downloads historical race data using FastF1 and generates processed race scenarios.

```bash
python generate_data.py
```

### 2. Train the Deep Reinforcement Learning Agent

```bash
python train_dqn.py
```

### 3. Evaluate the Deep Reinforcement Learning Agent

```bash
python evaluate_dqn.py
```

### 4. Train the Quantum Reinforcement Learning Agent

```bash
python train_qrl.py
```

### 5. Evaluate the Quantum Reinforcement Learning Agent

```bash
python evaluate_qrl.py
```

### 6. Compare All Strategies

```bash
python compare_agents.py
```

This script compares the performance of:

- Historical Formula 1 Strategy
- Deep Reinforcement Learning (DQN)
- Quantum Reinforcement Learning (QRL)

on unseen Formula 1 races from the 2025 season.

### 7. Launch the Dashboard

```bash
python dashboard.py
```

The dashboard provides an interactive visualization of race scenarios, agent decisions, evaluation metrics, and strategy comparisons.

---

## Technologies Used

- Python
- FastF1
- Gymnasium
- Stable-Baselines3
- PyTorch
- Qiskit
- NumPy
- Pandas
- Optuna

---
