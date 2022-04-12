from dataclasses import dataclass
from typing import List, Tuple

import pygame
import random
from math import pow, sqrt
import numpy as np
import os

# encodes action as integer :
# 0 : gauche
# 1 : droite
# 2 : shoot
# 3 : pass

# encodes state as np.array(np.array(pixels))

class SpaceInvaders:
    NO_INVADERS = 2  # Nombre d'aliens

    def __init__(self, display: bool = False, granu_x=25, granu_y=25):
        self.granu_x=25
        self.granu_y=25

        self.display = display

        # nombre d'actions (left, right, fire, no_action)
        self.na = 4

        # initializing pygame
        pygame.init()

        DISPLAY_SIZE = (800, 600)

        if self.display:
            self.screen = pygame.display.set_mode(DISPLAY_SIZE)
        else:
            self.screen = pygame.display.set_mode(DISPLAY_SIZE, flags=pygame.HIDDEN)

        # caption and icon
        pygame.display.set_caption("(´｡• ω •｡`) Space invaders")

        # Score
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.player_Xchange = 0

        # Game Over
        self.game_over_font = pygame.font.Font('freesansbold.ttf', 64)

        # player
        self.playerImage = pygame.image.load(os.path.abspath('game/data/spaceship.png'))
        self.reset()

    def get_state(self):
        """ A COMPLETER AVEC VOTRE ETAT
        Cette méthode doit renvoyer l'état du système comme vous aurez choisi de
        le représenter. Vous pouvez utiliser les accesseurs ci-dessus pour cela. 
        """

        bullet = 1 if self.bullet_state == "fire" else 0
        enemy_x = [self.invader_X[i]//self.granu_x for i in range(self.NO_INVADERS)]
        enemy_y = [self.invader_Y[i]//self.granu_y for i in range(self.NO_INVADERS)]
        enemy_direction = [1 if self.invader_Xchange[i] > 0 else 0 for i in range(self.NO_INVADERS)]
        player_x = self.player_X//self.granu_x

        return [player_x, bullet, *enemy_x, *enemy_y, *enemy_direction]

    def reset(self):
        """Reset the game at the initial state.
        """
        self.score_val = 0

        self.player_X = 370
        self.player_Y = 523

        # Invader
        self.invaderImage = []
        self.invader_X = []
        self.invader_Y = []
        self.invader_Xchange = []
        self.invader_Ychange = []

        for _ in range(SpaceInvaders.NO_INVADERS):
            self.invaderImage.append(pygame.image.load(os.path.abspath('game/data/alien.png')))
            self.invader_X.append(random.randint(64, 737))
            self.invader_Y.append(random.randint(30, 180))
            self.invader_Xchange.append(1.2)
            self.invader_Ychange.append(50)

        # Bullet
        # rest - bullet is not moving
        # fire - bullet is moving
        self.bulletImage = pygame.image.load(os.path.abspath('game/data/bullet.png'))

        self.bullet_X = 0
        self.bullet_Y = 500
        self.bullet_Xchange = 0
        self.bullet_Ychange = 3
        self.bullet_state = "rest"

        if self.display:
            self.render()

        return self.get_state()

    def step(self, action):
        """Execute une action et renvoir l'état suivant, la récompense perçue 
        et un booléen indiquant si la partie est terminée ou non.
        """
        is_done = False
        reward = 0

        # RGB
        self.screen.fill((0, 0, 0))

        # Controling the player movement from the arrow keys
        if action == 0:  # GO LEFT
            self.player_Xchange = -1.7
        if action == 1:  # GO RIGHT
            self.player_Xchange = 1.7
        if action == 2:  # FIRE
            self.player_Xchange = 0
            # Fixing the change of direction of bullet
            if self.bullet_state == "rest":
                self.bullet_X = self.player_X
                self.move_bullet(self.bullet_X, self.bullet_Y)
        if action == 3:  # NO ACTION
            self.player_Xchange = 0

        # adding the change in the player position
        self.player_X += self.player_Xchange
        for i in range(SpaceInvaders.NO_INVADERS):
            self.invader_X[i] += self.invader_Xchange[i]

        # bullet movement
        if self.bullet_Y <= 0:
            self.bullet_Y = 600
            self.bullet_state = "rest"
        if self.bullet_state == "fire":
            self.move_bullet(self.bullet_X, self.bullet_Y)
            self.bullet_Y -= self.bullet_Ychange

        # movement of the invader
        for i in range(SpaceInvaders.NO_INVADERS):
            if self.invader_Y[i] >= 450:
                if abs(self.player_X - self.invader_X[i]) < 80:
                    for j in range(SpaceInvaders.NO_INVADERS):
                        self.invader_Y[j] = 2000
                    is_done = True
                    break

            if self.invader_X[i] >= 735 or self.invader_X[i] <= 0:
                self.invader_Xchange[i] *= -1
                self.invader_Y[i] += self.invader_Ychange[i]

            self.invader_X[i] = max(self.invader_X[i], 0)
            self.invader_X[i] = min(self.invader_X[i], 800)

            # Collision
            collision = self.is_collision(self.bullet_X, self.invader_X[i], self.bullet_Y, self.invader_Y[i])
            if collision:
                reward = 1
                self.score_val += 1
                self.bullet_Y = 600
                self.bullet_state = "rest"
                self.invader_X[i] = random.randint(64, 736)
                self.invader_Y[i] = random.randint(30, 200)
                self.invader_Xchange[i] *= -1

            self.move_invader(self.invader_X[i], self.invader_Y[i], i)


        # restricting the spaceship so that it doesn't go out of screen
        self.player_X = max(self.player_X, 16)
        self.player_X = min(self.player_X, 750)

        self.move_player(self.player_X, self.player_Y)

        if self.display:
            self.render()

        return self.get_state(), reward, is_done

    def render(self):
        pygame.display.update()

    def move_player(self, x, y):
        self.screen.blit(self.playerImage, (x - 16, y + 10))

    def move_invader(self, x, y, i):
        self.screen.blit(self.invaderImage[i], (x, y))

    def move_bullet(self, x, y):
        self.screen.blit(self.bulletImage, (x, y))
        self.bullet_state = "fire"

    def show_score(self):
        score = self.font.render("Points: " + str(self.score_val), True, (255, 255, 255))
        self.screen.blit(score, (5, 5))

    def game_over(self):
        game_over_text = self.game_over_font.render("GAME OVER", True, (255, 255, 255))
        self.screen.blit(game_over_text, (190, 250))

    def is_collision(self, x1, x2, y1, y2):
        distance = pow(x1 - x2, 2) + pow(y1 - y2, 2)
        return distance <= 2500
