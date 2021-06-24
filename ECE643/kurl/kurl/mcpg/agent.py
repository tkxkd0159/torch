import os
import torch
import numpy as np
from torch.distributions import Categorical

from kurl.game import SnakeGameAI, Direction, Point
from kurl.mcpg.model import Policy, Trainer
from kurl.tool import score_plot

ALPHA = 0.001

class Agent:

    def __init__(self, load):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.99

        if load == False:
            self.model = Policy(11, 256, 3)  # state : 11 / action : [straight, right, left]
        else:
            self.model = Policy(11, 256, 3)
            model_folder_path = './torch_model'
            file_name = os.path.join(model_folder_path, 'mcpg_model.pth')
            self.model.load_state_dict(torch.load(file_name))
            self.model.eval()

        self.trainer = Trainer(self.model, lr=ALPHA, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def train(self):
        self.trainer.step()

    def get_action(self, state):
        final_move = [0, 0, 0]

        state = torch.tensor(state, dtype=torch.float)
        probs = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        self.model.saved_log_probs.append(m.log_prob(action))
        move = action.item()
        final_move[move] = 1

        return final_move


def train(load=False, is_learn=True):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(load)
    game = SnakeGameAI()
    while True:
        ep_reward = 0

        for _ in range(10000):
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)

            agent.model.rewards.append(reward)
            ep_reward += reward

            if done:
                break

        # 한 에피소드 끝날 때
        game.reset()
        agent.n_games += 1
        agent.train()

        print(f'Episode {agent.n_games}, Reward : {ep_reward}')

        if score > record:
            record = score
            if is_learn == True:
                agent.model.save()
        print('Game', agent.n_games, 'Score', score, 'Record:', record)
        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)
        score_plot(plot_scores, plot_mean_scores)

        if total_score >= 50:
            break
