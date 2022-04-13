import numpy as np

from game.SpaceInvaders import SpaceInvaders
from controller.epsilon_profile import EpsilonProfile


class AutoAgent:
    """
    Agent automatique utilisant la méthode du qlearning
    """
    def __init__(self, game: SpaceInvaders, eps_profile: EpsilonProfile, gamma, alpha):
        """
        :param game: le jeu à résoudre
        :type game: SpaceInvaders

        :param eps_profile: le profil du paramètre d'exploration epsilon
        :type eps_profile: EpsilonProfile

        :param alpha: le learning rate
        :type alpha: float

        :param gamma: le discount factor
        :type gamma: float
        """

        self.num_actions = game.na
        self.eps_profile = eps_profile
        self.game = game

        self.Q = np.zeros([800//game.granu_x, 2,
                           *[800//game.granu_x]*game.NO_INVADERS,
                           *[650//game.granu_y]*game.NO_INVADERS,
                           *[2]*game.NO_INVADERS,
                           game.na])  # à compléter en fonction des états choisis

        # paramètres de l'algo d'apprentissage
        self.gamma = gamma
        self.alpha = alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

        self.ep_score = 0

    def select_action(self, state): # specifier type ?
        """
        Retourne l'action à effectuer en fonction du processus d'exploration (epsilon-greedy)

        :param state: l'état courant
        :return: l'action
        """

        if np.random.rand() < self.epsilon:
            a = np.random.randint(self.num_actions)
        else:
            a = self.select_greedy_action(state)
        return a

    def select_greedy_action(self, state): # specifier type ?
        """
        Retourne l'action gloutonne

        :param state: l'état courant
        :return: l'action gloutonne
        """
        mx = np.max(self.Q[state])
        return np.random.choice(np.where(self.Q[state] == mx)[0])

    def learn(self, env, n_episodes, max_steps):
        """Cette méthode exécute l'algorithme de q-learning.
        Il n'y a pas besoin de la modifier. Simplement la comprendre et faire le parallèle avec le cours.
        :param env: L'environnement
        :type env: gym.Envselect_action
        :param num_episodes: Le nombre d'épisode
        :type num_episodes: int
        :param max_num_steps: Le nombre maximum d'étape par épisode
        :type max_num_steps: int

        # Visualisation des données
        Elle doit proposer l'option de stockage de (i) la fonction de valeur & (ii) la Q-valeur
        dans un fichier de log
        """
        n_steps = np.zeros(n_episodes) + max_steps

        # Execute N episodes
        for episode in range(n_episodes):
            # Reinitialise l'environnement
            state = env.reset()
            # Execute K steps
            for step in range(max_steps):
                # Selectionne une action
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = env.step(action)
                if reward:
                    self.ep_score+=reward
                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)

                if terminal:
                    n_steps[episode] = step + 1
                    break

                state = next_state
            # Mets à jour la valeur du epsilon
            self.epsilon = max(self.epsilon - self.eps_profile.dec_episode / (n_episodes - 1.), self.eps_profile.final)
            print(f"End episode {episode} : score {self.ep_score}, epsilon {self.epsilon}")

    def updateQ(self, state, action, reward, next_state):
        self.Q[state][action] = (1. - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))
