import numpy as np
from game.SpaceInvaders import SpaceInvaders
from controller.epsilon_profile import EpsilonProfile
import pandas as pd

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

        self.Q = np.zeros([2, *[len(game.granu_x)] * game.NO_INVADERS,
                          *[len(game.granu_y)] * game.NO_INVADERS,
                          *[2] * game.NO_INVADERS,
                          game.na])  # à compléter en fonction des états choisis

        # paramètres de l'algo d'apprentissage
        self.gamma = gamma
        self.alpha = alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

        self.ep_score = 0

        # Visualisation des données
        self.qvalues = pd.DataFrame(data={'episode': [], 'value': []})
        self.values = pd.DataFrame(data={'nx': [*[len(game.granu_x)] * game.NO_INVADERS], 'ny': [*[len(game.granu_y)] * game.NO_INVADERS], 'no_invaders':[*[2] * game.NO_INVADERS]})
	
    def select_action(self, state):  # specifier type ?
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

    def select_greedy_action(self, state):
        """
        Retourne l'action gloutonne

        :param state: l'état courant
        :return: l'action gloutonne
        """
        try:
            mx = np.max(self.Q[state])
            # if mx != 0:
            #     print(mx, np.where(self.Q[state] == mx))
            return np.random.choice(np.where(self.Q[state] == mx)[0])
        except IndexError:
            print("INDEX ERROR")
            print(state, self.Q.shape)

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
            self.ep_score = 0
            # Reinitialise l'environnement
            state = env.reset()
            # Execute K steps
            for step in range(max_steps):
                # Selectionne une action
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = env.step(action)

                if terminal:
                    n_steps[episode] = step + 1
                    break

                if reward:
                    self.ep_score += reward
                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)

                state = next_state
            # Mets à jour la valeur du epsilon
            self.epsilon = max(self.epsilon - self.eps_profile.dec_episode / (n_episodes - 1.), self.eps_profile.final)
            # print(f"End episode {episode} : score {self.ep_score}, epsilon {self.epsilon}")

            if n_episodes >= 0:
                state = env.reset()
                print("\r#> Ep. {}/{} Value {}".format(episode, n_episodes,
                                                       self.Q[state][self.select_greedy_action(state)]), end=" ")
                self.save_log(env, episode)

            # self.values.to_csv('logV.csv')
            self.qvalues.to_csv('logQ.csv')

    def updateQ(self, state, action, reward, next_state):
        try:
            new_value = (1. - self.alpha) * self.Q[state][action] + self.alpha * (
                    reward + self.gamma * np.max(self.Q[next_state]))
            # if new_value:
            
            #     print(state, action, reward, new_
            
            self.Q[state][action] = new_value
        except IndexError:
            print("INDEX ERROR")
            print(state, action, self.Q.shape)

    def save_log(self, env, episode):
        """Sauvegarde les données d'apprentissage.
        :warning: Vous n'avez pas besoin de comprendre cette méthode
        """
        state = env.reset()
        # Construit la fonction de valeur d'état associée à Q
        # V = np.zeros([2, *[len(self.game.granu_x)] * self.game.NO_INVADERS,
        #                   *[len(self.game.granu_y)] * self.game.NO_INVADERS,
        #                   *[2] * self.game.NO_INVADERS])
        # for state in self.game.get_state():
        #     val = self.Q[state][self.select_action(state)]
        #     # V[state] = val

        self.qvalues = self.qvalues.append(
            {'episode': episode, 'value': self.Q[state][self.select_greedy_action(state)]}, ignore_index=True)
        # self.values = self.values.append(
        #     {'episode': episode, 'value': np.reshape(V, (1, *[len(self.game.granu_x)] * self.game.NO_INVADERS *
        #                   [len(self.game.granu_y)] * self.game.NO_INVADERS * [2] * self.game.NO_INVADERS))[0]}, ignore_index=True)