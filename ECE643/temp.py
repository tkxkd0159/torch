import gym

from stable_baselines3 import DQN

from kurl.temp.env import SnakeEnv

env = SnakeEnv()

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
total_score = 0
record = 0
for i in range(3000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

    if info['score'] > record:
        record = info['score']

    total_score += info['score']
    mean_score = total_score / (i+1)

print(mean_score, record)

env.close()