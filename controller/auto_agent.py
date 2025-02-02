import numpy as np


class AutoAgent():
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def select_action(self, _):
        return np.random.randint(self.num_actions)
