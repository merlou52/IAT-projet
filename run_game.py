import sys
from time import sleep

import pygame

from controller.auto_agent import AutoAgent
from controller.epsilon_profile import EpsilonProfile
from controller.keyboard import KeyboardController
from game.SpaceInvaders import SpaceInvaders


if __name__ == '__main__':
    granu_x = [-20, 200, 325, 425, 550]
    granu_y = [-20, 150, 250]

    game = SpaceInvaders(display=False, granu_x=granu_x, granu_y=granu_y)

    if len(sys.argv) != 5:
        print("python run_game.py <episodes> <steps> <alpha> <gamma>")
        sys.exit(1)

    N_EPISODES = int(sys.argv[1])
    MAX_STEPS = int(sys.argv[2])
    ALPHA = float(sys.argv[3])
    GAMMA = float(sys.argv[4])
    eps_profile = EpsilonProfile(1.0, 0.01)

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

    print("SCORE TOTAL 1: " + str(game.score_val))

    is_done = False
    game.reset()
    while not is_done:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)

    print("SCORE TOTAL 2 : " + str(game.score_val))
