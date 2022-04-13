from time import sleep

from controller.auto_agent import AutoAgent
from controller.epsilon_profile import EpsilonProfile
from game.SpaceInvaders import SpaceInvaders


if __name__ == '__main__':
    granu_x = 25
    granu_y = 50

    game = SpaceInvaders(display=True, granu_x=granu_x, granu_y=granu_y)
    # controller = KeyboardController()
    # controller = RandomAgent(game.na)

    n_episodes = 15
    max_steps = 5000
    alpha = 0.001
    gamma = 1
    eps_profile = EpsilonProfile(1.0, 0.1)

    controller = AutoAgent(game, eps_profile, gamma, alpha)
    controller.learn(game, n_episodes, max_steps)
    state = game.reset()

    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
