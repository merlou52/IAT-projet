from time import sleep

from controller.auto_agent import AutoAgent
from controller.epsilon_profile import EpsilonProfile
from game.SpaceInvaders import SpaceInvaders


def main():

    game = SpaceInvaders(display=True)
    # controller = KeyboardController()
    # controller = RandomAgent(game.na)

    n_episodes = 2000
    max_steps = 50
    alpha = 0.001
    gamma = 1
    eps_profile = EpsilonProfile(1.0, 0.1)

    controller = AutoAgent(game, eps_profile, gamma, alpha)
    state = game.reset()

    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        sleep(0.0001)


if __name__ == '__main__' :
    main()
