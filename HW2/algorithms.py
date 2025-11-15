import numpy as np
import pandas as pd
from loguru import logger
from Bandit import Bandit

# -----------------------------
# Epsilon-Greedy
# -----------------------------
class EpsilonGreedy(Bandit):
    def __init__(self, p, epsilon=1.0):
        self.p = np.array(p)
        self.k = len(p)
        self.epsilon = epsilon
        self.counts = np.zeros(self.k)
        self.values = np.zeros(self.k)
        self.rewards_history = []
        self.arm_history = []

    def __repr__(self):
        return f"EpsilonGreedy(epsilon={self.epsilon:.3f})"

    def pull(self, t=None):
        current_epsilon = self.epsilon if t is None else self.epsilon / (t + 1)
        if np.random.random() < current_epsilon:
            arm = np.random.randint(self.k)
        else:
            arm = np.argmax(self.values)
        reward = np.random.normal(self.p[arm], 1)
        return arm, reward

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def experiment(self, n_trials=20000):
        logger.info("Running Epsilon-Greedy Experiment")
        for t in range(n_trials):
            arm, reward = self.pull(t)
            self.update(arm, reward)
            self.arm_history.append(arm)
            self.rewards_history.append(reward)
        regret = np.max(self.p) - np.array([self.p[a] for a in self.arm_history])
        df = pd.DataFrame({
            "Arm": self.arm_history,
            "Reward": self.rewards_history,
            "Regret": regret,
            "Algorithm": "Epsilon-Greedy"
        })
        return df

    def report(self):
        avg_reward = np.mean(self.rewards_history)
        avg_regret = np.mean(np.max(self.p) - np.array(self.rewards_history))
        logger.info(f"[Epsilon-Greedy] Avg Reward: {avg_reward:.4f}")
        logger.info(f"[Epsilon-Greedy] Avg Regret: {avg_regret:.4f}")


# -----------------------------
# Thompson Sampling
# -----------------------------
class ThompsonSampling(Bandit):
    def __init__(self, p):
        self.p = np.array(p)
        self.k = len(p)
        self.alpha = np.ones(self.k)
        self.beta = np.ones(self.k)
        self.rewards_history = []
        self.arm_history = []

    def __repr__(self):
        return "ThompsonSampling()"

    def pull(self):
        theta = np.random.beta(self.alpha, self.beta)
        arm = np.argmax(theta)
        reward = np.random.normal(self.p[arm], 1)
        return arm, reward

    def update(self, arm, reward):
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def experiment(self, n_trials=20000):
        logger.info("Running Thompson Sampling Experiment")
        for _ in range(n_trials):
            arm, reward = self.pull()
            self.update(arm, reward)
            self.arm_history.append(arm)
            self.rewards_history.append(reward)
        regret = np.max(self.p) - np.array([self.p[a] for a in self.arm_history])
        df = pd.DataFrame({
            "Arm": self.arm_history,
            "Reward": self.rewards_history,
            "Regret": regret,
            "Algorithm": "Thompson Sampling"
        })
        return df

    def report(self):
        avg_reward = np.mean(self.rewards_history)
        avg_regret = np.mean(np.max(self.p) - np.array(self.rewards_history))
        logger.info(f"[Thompson Sampling] Avg Reward: {avg_reward:.4f}")
        logger.info(f"[Thompson Sampling] Avg Regret: {avg_regret:.4f}")


# -----------------------------
# Bayesian UCB - bonus
# -----------------------------
class BayesianUCB(Bandit):
    """Bayesian Upper Confidence Bound with Gaussian likelihood."""

    def __init__(self, p):
        self.p = np.array(p)
        self.k = len(p)
        self.counts = np.zeros(self.k)
        self.means = np.zeros(self.k)
        self.rewards_history = []
        self.arm_history = []

    def __repr__(self):
        return "BayesianUCB()"

    def pull(self, t=None):
        t = t + 1 if t is not None else np.sum(self.counts) + 1
        ucb = self.means + np.sqrt(2 * np.log(t) / (self.counts + 1e-5))
        arm = np.argmax(ucb)
        reward = np.random.normal(self.p[arm], 1)
        return arm, reward

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm]

    def experiment(self, n_trials=20000):
        logger.info("Running Bayesian UCB Experiment")
        for t in range(n_trials):
            arm, reward = self.pull(t)
            self.update(arm, reward)
            self.arm_history.append(arm)
            self.rewards_history.append(reward)
        regret = np.max(self.p) - np.array([self.p[a] for a in self.arm_history])
        df = pd.DataFrame({
            "Arm": self.arm_history,
            "Reward": self.rewards_history,
            "Regret": regret,
            "Algorithm": "Bayesian UCB"
        })
        return df

    def report(self):
        avg_reward = np.mean(self.rewards_history)
        avg_regret = np.mean(np.max(self.p) - np.array(self.rewards_history))
        logger.info(f"[Bayesian UCB] Avg Reward: {avg_reward:.4f}")
        logger.info(f"[Bayesian UCB] Avg Regret: {avg_regret:.4f}")
