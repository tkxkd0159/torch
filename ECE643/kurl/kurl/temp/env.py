import random
import math
from collections import namedtuple
import numpy as np
import pygame
import gym
from gym import spaces

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
YELLOW = (255, 255, 0)



class SnakeEnv(gym.Env):

    def __init__(self):
        self.w = 640
        self.h = 480
        self.block_size = 20
        self.speed = 60
        self.action_space = spaces.Discrete(4)

        high = np.ones(12)
        low = np.zeros(12)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.display = None


    def reset(self):
        self.direction = 1

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

    def close(self):
        if self.display:
            pygame.quit()
            quit()

    def get_state(self):

        # wall check
        if self.head.y <= self.h / 4:
            wall_up, wall_down = 1, 0
        elif self.head.y >= self.h*3 / 4:
            wall_up, wall_down = 0, 1
        else:
            wall_up, wall_down = 0, 0
        if self.head.x <= self.w / 4:
            wall_right, wall_left = 0, 1
        elif self.head.x >= self.h*3 / 4:
            wall_right, wall_left = 1, 0
        else:
            wall_right, wall_left = 0, 0

        # body close
        body_up = []
        body_right = []
        body_down = []
        body_left = []
        if len(self.snake) > 3:
            for body in self.snake[3:]:
                if self.measure_distance(body) == 20:
                    if body.y > self.head.y:
                        body_down.append(1)
                    elif body.y < self.head.y:
                        body_up.append(1)
                    if body.x < self.head.x:
                        body_left.append(1)
                    elif body.x > self.head.x:
                        body_right.append(1)

        if len(body_up) > 0: body_up = 1
        else: body_up = 0
        if len(body_right) > 0: body_right = 1
        else: body_right = 0
        if len(body_down) > 0: body_down = 1
        else: body_down = 0
        if len(body_left) > 0: body_left = 1
        else: body_left = 0


        state = [
            int(self.head.y < self.food.y),
            int(self.head.x < self.food.x),
            int(self.head.y > self.food.y),
            int(self.head.x > self.food.x),
            int(wall_up or body_up),
            int(wall_right or body_right),
            int(wall_down or body_down),
            int(wall_left or body_left),
            int(self.direction == 0),
            int(self.direction == 1),
            int(self.direction == 2),
            int(self.direction == 3)
            ]

        return np.array(state)

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
        """
        action
            0: right
            1: down
            2: left
            3: up
        """

        action = int(action) % 4
        if np.abs(self.direction - action) != 2:
            self.direction = action

        x = self.head.x
        y = self.head.y
        if self.direction == 0:
            x += self.block_size
        elif self.direction == 1:
            y += self.block_size
        elif self.direction == 2:
            x -= self.block_size
        elif self.direction == 3:
            y -= self.block_size

        self.head = Point(x, y)

    def measure_distance(self, target):
        self.prev_dist = self.dist
        self.dist = math.sqrt((self.head.x-target.x)**2 + (self.head.y-target.y)**2)