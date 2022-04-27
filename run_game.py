import sys
from time import sleep
from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame

from controller.auto_agent import AutoAgent
from controller.epsilon_profile import EpsilonProfile
from controller.keyboard import KeyboardController
from game.SpaceInvaders import SpaceInvaders

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("python run_game.py <episodes> <steps> <alpha> <gamma> <granu_x")
        sys.exit(1)

    N_EPISODES = int(sys.argv[1])
    MAX_STEPS = int(sys.argv[2])
    ALPHA = float(sys.argv[3])
    GAMMA = float(sys.argv[4])
    eps_profile = EpsilonProfile(1.0, 0.01)

    granu_x = list(range(-20, 800, int(sys.argv[5])))
    granu_y = [-20, 150, 250]

    game = SpaceInvaders(display=False, granu_x=granu_x, granu_y=granu_y)

    controller = AutoAgent(game, eps_profile, GAMMA, ALPHA)
    controller.learn(game, N_EPISODES, MAX_STEPS)
    state = game.reset()

    game.display = True
    DISPLAY_SIZE = (800, 600)
    game.screen = pygame.display.set_mode(DISPLAY_SIZE)

    is_done = False

    while not is_done:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)

    print(str(game.score_val))

    is_done = False
    game.reset()
    while not is_done:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)

    print(str(game.score_val))
