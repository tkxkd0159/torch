import random
import math
from collections import namedtuple
import numpy as np
import pygame
import gym
from gym import spaces
from enum import Enum

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
YELLOW = (255, 255, 0)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

BLOCK_SIZE = 20


class SnakeEnv(gym.Env):

    def __init__(self):
        self.w = 640
        self.h = 480
        self.block_size = 20
        self.speed = 30
        self.action_space = spaces.Discrete(4)

        high = np.ones(12)
        low = np.zeros(12)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.display = None


    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x - self.block_size, self.head.y),
                      Point(self.head.x - (2 * self.block_size), self.head.y)]

        self.score = 0
        self.reward = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.dist = math.sqrt((self.head.x-self.food.x)**2 + (self.head.y-self.food.y)**2)

        return self.get_state()


    def step(self, action):
        reward_given = False
        done = False
        info = {'score': self.score}

        self._move(action)
        self.snake.insert(0, self.head)

        if self.head == self.food:
            self.score += 1
            self.reward = 10
            self._place_food()
            reward_given = True
        else:
            self.snake.pop()

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            self.reward = -100
            reward_given = True


        self.measure_distance(self.food)
        if not reward_given:
            if self.dist < self.prev_dist:
                self.reward = 1
            else:
                self.reward = -1


        return self.get_state(), self.reward, done, info

    def render(self):
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake game')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Calibri', 25)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.display.fill(BLACK)
        for pt in self.snake:
            if pt == self.head:
                pygame.draw.rect(self.display, YELLOW, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
                continue
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))

        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        self.clock.tick(self.speed)

    def close(self):
        if self.display:
            pygame.quit()
            quit()

    def get_state(self):

        head = self.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        if pt.x > self.w - self.block_size or pt.x < 0 or pt.y > self.h - self.block_size or pt.y < 0:
            return True

        if pt in self.snake[1:]:
            return True

        return False

    def _place_food(self):
        x = random.randint(0, (self.w - self.block_size ) // self.block_size ) * self.block_size
        y = random.randint(0, (self.h - self.block_size ) // self.block_size ) * self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _move(self, action):
        # [straight, clockwise_turn, anti-clockwise_turn]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        elif np.array_equal(action, [0, 0, 1]):
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
        else:
            raise ValueError("Invalid action")

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def measure_distance(self, target):
        self.prev_dist = self.dist
        self.dist = math.sqrt((self.head.x-target.x)**2 + (self.head.y-target.y)**2)