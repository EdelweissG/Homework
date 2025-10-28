# HW2

This folder contains the implementation and experiments for Homework 2: Multi-Armed Bandit Algorithms.


## Description
This homework implements and compares three multi-armed bandit algorithms:

- **Epsilon-Greedy**
- **Thompson Sampling**
- **Bayesian UCB**

The goal is to simulate each algorithm on a 4-armed bandit problem and analyze their performance.

## Folder Structure

```text
├── algorithms.py # Implementation of the bandit algorithms
├── Bandit.py # Abstract base class for bandits
├── configs.py # Settings: bandit arms, number of trials, epsilon
├── visualizations.py # Plotting and visualization functions
├── experiment.ipynb # Notebook to run experiments and visualize results
└── results.csv # Experiment output file
```

## How to run
- Open `experiment.ipynb` in IDE with the Jupyter extension.
- Execute all cells to run the experiments for all algorithms.
- Visualize the learning curves and cumulative reward/regret plots.
- The per-trial results will be saved automatically to `results.csv`.

## Results

The experiments generate per-trial results including:
Chosen arm
Observed reward
Regret
Algorithm used
