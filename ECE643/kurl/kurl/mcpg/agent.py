from pathlib import Path
from itertools import count
import pandas as pd
import torch

from kurl.mcpg.model import finish_episode, select_action, policy
from kurl.envs.stock import StockTradingEnv

dir_path = Path(__file__).parent.parent.absolute()
file_path = "envs/stock.csv"
df = pd.read_csv(dir_path.joinpath(file_path))
df = df.sort_values('Date')

env = StockTradingEnv(df)
def train():
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):
            action = select_action(state.flatten())
            state, reward, done, _ = env.step(action)

            env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        finish_episode()
        if i_episode % 10 == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}')
