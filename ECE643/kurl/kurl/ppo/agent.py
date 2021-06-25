from pathlib import Path
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from kurl.envs.stock import StockTradingEnv


dir_path = Path(__file__).parent.parent.absolute()
file_path = "envs/stock.csv"

df = pd.read_csv(dir_path.joinpath(file_path))
df = df.sort_values('Date')

env = make_vec_env(StockTradingEnv, env_kwargs={"df": df})

def train(is_learn=False, log=False):
    if is_learn:
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=20000)
        model.save("ppo_stock")

    else:
        model = PPO.load("ppo_stock")


    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if log:
            with open('./log/ppo.txt', 'a') as f:
                f.write(f'{env.env_method("log")[0]}\n')
        env.render()