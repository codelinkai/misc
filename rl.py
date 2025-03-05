import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

class PortfolioEnv(gym.Env):
    """
    Custom Portfolio Optimization Environment
    for Reinforcement Learning (RL).
    """

    def __init__(self, df):
        super(PortfolioEnv, self).__init__()
        self.df = df
        self.unique_dates = sorted(df["date"].unique())
        self.current_step = 0
        self.num_stocks = len(df["ticker"].unique())

        # State Space: Features include past stock returns & technical indicators
        num_features = df.shape[1] - 2  # Excluding 'date' and 'ticker'
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,))

        # Action Space: Portfolio weights (must sum to 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stocks,))

        # Track portfolio history
        self.portfolio_returns = []
        self.portfolio_value = [1.0]  # Start with $1 (normalized)

    def _get_observation(self):
        """
        Get the current observation (excluding 'date' and 'ticker').
        """
        return self.df.iloc[self.current_step].drop(["date", "ticker"]).values

    def _normalize_weights(self, action):
        """
        Normalize weights so they sum to 1.0.
        """
        action = np.maximum(action, 0)  # Ensure non-negative weights
        return action / np.sum(action)  # Normalize

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        # Normalize action to ensure total weights sum to 1.0
        action = self._normalize_weights(action)

        # Get today's returns (NO FUTURE DATA USED)
        today_returns = self.df.pivot(index="date", columns="ticker", values="daily_price").pct_change().iloc[self.current_step]
        portfolio_return = np.dot(action, today_returns.fillna(0).values)  # Compute portfolio return

        # Compute Sharpe Ratio using past returns (NO FUTURE DATA USED)
        if self.current_step >= 10:  # Use at least 10 past days for Sharpe Ratio
            past_returns = self.portfolio_returns[-10:]  # Last 10 days
            avg_return = np.mean(past_returns)
            volatility = np.std(past_returns) + 1e-8  # Avoid division by zero
            sharpe_ratio = avg_return / volatility
        else:
            sharpe_ratio = 0  # Not enough history

        # Compute Max Drawdown using past values (NO FUTURE DATA USED)
        max_drawdown = np.min(
            self.portfolio_value[: self.current_step] - np.maximum.accumulate(self.portfolio_value[: self.current_step])
        )

        # Reward function (Sharpe Ratio - 0.1 * Max Drawdown)
        reward = sharpe_ratio - 0.1 * max_drawdown

        # Store portfolio history
        self.portfolio_returns.append(portfolio_return)
        self.portfolio_value.append(self.portfolio_value[-1] * (1 + portfolio_return))

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.unique_dates) - 1  # Done if we reach the last date

        return self._get_observation(), reward, done, {}

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.current_step = 0
        self.portfolio_returns = []
        self.portfolio_value = [1.0]
        return self._get_observation()

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset (Ensure it contains 'date', 'ticker', 'daily_price', 'sector', and technical indicators)
df = pd.read_csv("stock_data.csv")

### --- Discretization for Technical Indicators --- ###

# RSI (Relative Strength Index) into 5 bins
df["RSI_bin"] = pd.qcut(df["RSI"], q=5, labels=False)

# MACD into 5 bins
df["MACD_bin"] = pd.qcut(df["MACD"], q=5, labels=False)

# Bollinger Band %B into 5 bins
df["BB_bin"] = pd.qcut(df["BB_percent"], q=5, labels=False)

# Drop original continuous columns (since we use discrete versions)
df = df.drop(columns=["RSI", "MACD", "BB_percent"])

### --- Encoding Sector Information --- ###

# Label Encoding for sector categories
le = LabelEncoder()
df["sector_encoded"] = le.fit_transform(df["sector"])

# Drop string-based sector names
df = df.drop(columns=["sector"])

# Print sample data
print(df.head())
# Load your dataset (Ensure it has 'date', 'ticker', 'daily_price', technical indicators, sector)
df = pd.read_csv("stock_data.csv")

# Initialize environment
env = PortfolioEnv(df)

# Train the RL agent using PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save trained model
model.save("ppo_portfolio")

# Load the trained model
model = PPO.load("ppo_portfolio")

# Run the agent on test data
obs = env.reset()
done = False
portfolio_values = []

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    portfolio_values.append(env.portfolio_value[-1])

# Plot portfolio growth
plt.plot(portfolio_values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Portfolio Value ($)")
plt.show()
