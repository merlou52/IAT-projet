from time import sleep

from controller.auto_agent import AutoAgent
from controller.epsilon_profile import EpsilonProfile
from controller.keyboard import KeyboardController
from game.SpaceInvaders import SpaceInvaders


if __name__ == '__main__':
    granu_x = [-20, 200, 325, 425, 550]
    granu_y = [-20, 150, 250]


    game = SpaceInvaders(display=True, granu_x=granu_x, granu_y=granu_y)
    controller = KeyboardController()
    # controller = RandomAgent(game.na)

    n_episodes = 100
    max_steps = 150
    alpha = 0.01
    gamma = 0.9
    eps_profile = EpsilonProfile(1.0, 0.1)

    # controller = AutoAgent(game, eps_profile, gamma, alpha)
    # controller.learn(game, n_episodes, max_steps)
    state = game.reset()

    game.display = True
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
