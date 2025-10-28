import matplotlib.pyplot as plt
from loguru import logger

class Visualization:
    def __init__(self, df_list):
        self.df_list = df_list

    def plot_rewards(self):
        logger.info("Plotting Rolling Mean Rewards")
        plt.figure()
        for df in self.df_list:
            plt.plot(df["Reward"].rolling(200).mean(), label=df["Algorithm"].iloc[0])
        plt.title("Learning Curve: Rolling Mean Reward")
        plt.xlabel("Trials")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.show()

    def plot_cumulative(self):
        logger.info("Plotting Cumulative Reward & Regret")
        plt.figure()
        for df in self.df_list:
            plt.plot(df["Reward"].cumsum(), label=df["Algorithm"].iloc[0])
        plt.title("Cumulative Reward")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.show()

        plt.figure()
        for df in self.df_list:
            plt.plot(df["Regret"].cumsum(), label=df["Algorithm"].iloc[0])
        plt.title("Cumulative Regret")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.show()
