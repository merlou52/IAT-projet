import numpy as np

from game.SpaceInvaders import SpaceInvaders
from epsilon_profile import EpsilonProfile


class AutoAgent():
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
        self.game = game

        self.Q = np.zeros([game.na])  # à compléter en fonction des états choisis

        # paramètres de l'algo d'apprentissage
        self.gamma = gamma
        self.alpha = alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

    def select_action(self, _):
        return np.random.randint(self.num_actions)

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
            state = env.reset_using_existing_maze()
            # Execute K steps
            for step in range(max_steps):
                # Selectionne une action
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = env.step(action)
                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)

                if terminal:
                    n_steps[episode] = step + 1
                    break

                state = next_state
            # Mets à jour la valeur du epsilon
            self.epsilon = max(self.epsilon - self.eps_profile.dec_episode / (n_episodes - 1.), self.eps_profile.final)

    def updateQ(self, state, action, reward, next_state):
        return "not implemented"