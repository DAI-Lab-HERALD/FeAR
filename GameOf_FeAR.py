import numpy as np

np.random.seed(0)
rng = np.random.default_rng(seed=0)
import seaborn as sns

sns.set_theme(style="ticks")
import matplotlib.pyplot as plt
import copy
import json
from json.decoder import JSONDecodeError
import pprint
import datetime
import os
import glob

import GWorld
import Agent
import Responsibility
import Emergence
import PlotGWorld

import Plot_cFeAR_ as plotcf

plotgw = PlotGWorld.PlotGWorld();  # Object for accessing plotters
import AnalysisOf_FeAR_Sims as FeARUI

import matplotlib.backends.backend_agg as agg
import pygame
from pygame.locals import *

# -------------------------------------------------------------------------------------------------------------------- #
TIMED = True
CHEAT_TIMED = True
RESET_SCORES = False
RESET_SCORES = True

REMOVE_COLLIDING_AGENTS = False

# N_TRIALS = 10
N_TRIALS = 10
FEAR_POINTS = 5000
CRASH_POINTS = 1000
APPLE_POINTS = 500
# N_APPLES = 20
N_APPLES = 25


TIME_ACTION_SELECTION = 5000
# TIME_WINDOW = 5000
TIME_WINDOW = 500
FEAR_TIME_WINDOW = 1000

pygame.init()
pygame_icon = pygame.image.load('Game_of_FeAR_icon.png')
pygame.display.set_icon(pygame_icon)
pygame.display.set_caption('Game of FeAR   :   Feasible Action-Space Reduction')

# GAME_FONT = pygame.font.Font('DragonHunter.otf', 30, bold=False)
GAME_FONT = pygame.font.Font('Jaro_36pt-Regular.ttf', 30)

# DISPLAY_FONT = pygame.font.Font('LinLibertine_RB.ttf', 20, bold=True)
# DISPLAY_FONT = pygame.font.Font('DragonHunter.otf', 20, bold=False)
# DISPLAY_FONT = pygame.font.SysFont('couriernew', 20, bold=True)
# DISPLAY_FONT = pygame.font.SysFont('sitkabanner', 15, bold=True)
DISPLAY_FONT = pygame.font.SysFont('bahnschrift', 28, bold=False)

DISPLAY_FONT_SMALL = pygame.font.SysFont('consolas', 20, bold=False)

# DISPLAY_FONT_TINY = pygame.font.Font('LinLibertine_RB.ttf', 15, bold=False)
# DISPLAY_FONT_TINY = pygame.font.SysFont('sitkabanner', 15, bold=False)
DISPLAY_FONT_TINY = pygame.font.SysFont('consolas', 16, bold=False)
# DISPLAY_FONT_TINY = pygame.font.SysFont('bahnschrift', 15, bold=False)
DISPLAY_FONT_MICRO = pygame.font.SysFont('consolas', 12, bold=False)

RECORD_FOLDER = 'Results'
SAVE_IMAGES_TO = 'Plots'
OVERWRITE_IMAGES = True

JSON_4_PLOTS = 'Instances2plot.json'
# JSON_4_PLOTS = 'Instances2plot_finer.json'

N_TOP = 10

# CMAP_4COUNT = sns.cubehelix_palette(as_cmap=True, light=1)
hue_validMoves = 190
CMAP_4COUNT = sns.diverging_palette(365 - hue_validMoves, hue_validMoves, l=60, as_cmap=True)

ANNOTATE_FONT_SIZE = 12

BLACK = (100, 100, 100)
RED = (150, 10, 30)
BLUE = (0, 128, 255)
WHITE = (255, 255, 255)
GOLD = (255, 215, 0)
TEXT_PADDING = 15
PYGAME_WIN_SIZE = (1080, 600)

FPS = 5


def main():
    # ---------------------------------------------------------------------------------------------------------------- #
    ui = ui_FeAR()
    ui.run_manager()

    pass


class ui_FeAR:

    def __init__(self):
        self.run = True
        self.run_next = 'start_window'
        self.game_over = False
        self.trials = 1
        self.n_trials = N_TRIALS
        self.score = 0

        # self.game_levels = ['GameMap_8', 'GameMap']
        self.game_levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
        self.game_level_id = 0
        self.game_level = self.game_levels[self.game_level_id]
        self.n_game_levels = len(self.game_levels)

        if RESET_SCORES:
            self.highscores = {
                'Level 1': 0,
                'Level 2': 0,
                'Level 3': 0,
                'Level 4': 0,
                'Level 5': 0
            }
        else:
            with open('highscores.json') as json_file:
                self.highscores = json.load(json_file)

        print(f'{pprint.pformat(self.highscores, width=50)}')

        self.font = DISPLAY_FONT
        self.small_font = DISPLAY_FONT_SMALL
        self.tiny_font = DISPLAY_FONT_TINY
        self.micro_font = DISPLAY_FONT_MICRO
        self.game_font = GAME_FONT

        window = pygame.display.set_mode(PYGAME_WIN_SIZE, pygame.RESIZABLE, DOUBLEBUF)
        self.screen = pygame.display.get_surface()
        pygame.display.flip()
        self.clock = pygame.time.Clock()
        self.w, self.h = pygame.display.get_surface().get_size()

        self.run_gworld = []
        self.ego_action = 0
        self.ego_action_RL = 0
        self.ego_action_UD = 0
        self.progress_bar_h = 0

        return

    def run_manager(self):
        while self.run:
            if self.run_next == 'start_window':
                self.run_start_window()
            elif self.run_next == 'instruction_window':
                self.run_instruction_window()
            elif self.run_next == 'action_selection_window':
                self.run_action_selection_window()
            elif self.run_next == 'selected_actions_window':
                self.run_selected_actions_window()
            elif self.run_next == 'fear_score_window':
                self.run_fear_score_window()
            elif self.run_next == 'update_score_window':
                self.run_update_score_window()
            elif self.run_next == 'game_over_window':
                self.run_game_over_window()
            else:
                self.run_start_window()

        print('Exiting Game!')
        pygame.display.quit()
        pygame.quit()

    def run_start_window(self):
        self.game_over = False
        self.score = 0
        self.ego_action = 0
        self.ego_action_RL = 0
        self.ego_action_UD = 0

        self.trials = 1

        name_text = self.game_font.render(('Game of FeAR'), True, RED)
        tw_name, th_name = name_text.get_size()
        instruction_text = self.tiny_font.render(('Press ENTER to begin.'), True, BLACK)
        tw_instruction, th_instruction = instruction_text.get_size()

        while self.run and self.run_next == 'start_window':

            level_text = self.small_font.render(f'{self.game_level}', True, BLACK)
            tw_level, th_level = level_text.get_size()

            self.clock.tick(FPS)
            self.screen.fill((255, 255, 255))  # Fill White
            self.w, self.h = pygame.display.get_surface().get_size()
            self.screen.fill((255, 255, 255))  # Fill White
            self.screen.blit(name_text, ((self.w - tw_name) // 2, (self.h - th_name) // 3))
            self.screen.blit(level_text, ((self.w - tw_level) // 2, ((self.h - th_name) // 3) + 50))
            self.screen.blit(instruction_text, ((self.w - tw_instruction) // 2,
                                                ((self.h - th_name) // 3) + 150 + th_level))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.event.clear()
                        self.run = False
                    if event.key == pygame.K_RETURN:
                        pygame.event.clear()
                        # self.run_next = 'action_selection_window'
                        self.run_next = 'instruction_window'
                    if event.key == pygame.K_LSHIFT:
                        pygame.event.clear()
                        self.game_level_id = (self.game_level_id - 1) % self.n_game_levels
                        self.game_level = self.game_levels[self.game_level_id]
                    if event.key == pygame.K_RSHIFT:
                        pygame.event.clear()
                        self.game_level_id = (self.game_level_id + 1) % self.n_game_levels
                        self.game_level = self.game_levels[self.game_level_id]

            pygame.display.update()

        self.run_gworld = runGWorld(game_level=self.game_level)

        return self.run

    def run_instruction_window(self):

        while self.run and self.run_next == 'instruction_window':
            self.clock.tick(FPS)
            self.w, self.h = pygame.display.get_surface().get_size()
            self.screen.fill((255, 255, 255))  # Fill White
            heading_text = self.font.render(('Instructions'), True, BLACK)
            tw_heading, th_heading = heading_text.get_size()
            self.screen.blit(heading_text, ((self.w - tw_heading) // 2, 200))
            self.draw_description(content='Use the arrow keys to select the velocity of ego agent for each trial.', h=-self.h // 2 + th_heading - 50)
            self.draw_description(content='Collect stars without colliding with other agents.', h=-self.h // 2 + th_heading - 25)
            self.draw_description(content='Collisions = - 1000 points, Stars = + 500 points', h=-self.h // 2 + th_heading - 0)
            self.draw_description(content='Bonus points based on FeAR', h=-self.h // 2 + th_heading + 25)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.event.clear()
                        self.run = False
                    if event.key == pygame.K_RETURN:
                        pygame.event.clear()
                        self.run_next = 'action_selection_window'
            pygame.display.update()

        return self.run



    def run_action_selection_window(self):
        update_surf = True
        if TIMED:
            time_now = 0
        while self.run and self.run_next == 'action_selection_window':
            if TIMED:
                time_now = time_now + self.clock.get_time()
                if time_now >= TIME_ACTION_SELECTION:
                    self.run_next = 'selected_actions_window'
            self.clock.tick(FPS)
            self.screen.fill((255, 255, 255))  # Fill White
            self.w, self.h = pygame.display.get_surface().get_size()
            self.screen.fill((255, 255, 255))  # Fill White

            if update_surf:
                update_surf = False
                ax = self.run_gworld.view_ego_action(ego_action=self.ego_action)
                world_surf = FeARUI.get_pygame_surf_from_ax(ax)
            scaled_world_surf = FeARUI.rescale_photo(world_surf, self.w, self.h - 20)
            self.screen.blit(scaled_world_surf, (0, 10))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.event.clear()
                        self.run = False
                    if event.key == pygame.K_RETURN:
                        pygame.event.clear()
                        self.run_next = 'selected_actions_window'
                    # ------------------------------------------------------
                    if event.key == pygame.K_UP:
                        pygame.event.clear()
                        self.ego_action_UD = np.clip(self.ego_action_UD + 1, -4, 4)
                        self.ego_action_RL = 0
                        # self.ego_action = 1
                        update_surf = True
                    if event.key == pygame.K_DOWN:
                        pygame.event.clear()
                        self.ego_action_UD = np.clip(self.ego_action_UD - 1, -4, 4)
                        self.ego_action_RL = 0
                        # self.ego_action = 2
                        update_surf = True
                    if event.key == pygame.K_LEFT:
                        pygame.event.clear()
                        self.ego_action_UD = 0
                        self.ego_action_RL = np.clip(self.ego_action_RL - 1, -4, 4)
                        # self.ego_action = 3
                        update_surf = True
                    if event.key == pygame.K_RIGHT:
                        pygame.event.clear()
                        self.ego_action_UD = 0
                        self.ego_action_RL = np.clip(self.ego_action_RL + 1, -4, 4)
                        # self.ego_action = 4
                        update_surf = True
                    # ------------------------------------------------------
                    if event.key == pygame.K_x:
                        pygame.event.clear()
                        self.game_over = True
                        self.run_next = 'fear_score_window'
                    if event.key == pygame.K_SPACE:
                        pygame.event.clear()
                        if CHEAT_TIMED:
                            time_now = 0

                    if update_surf:
                        if self.ego_action_UD > 0:  # UP
                            self.ego_action = 1 + (self.ego_action_UD - 1) * 4
                        elif self.ego_action_UD < 0:  # DOWN
                            self.ego_action = 2 + (-self.ego_action_UD - 1) * 4
                        elif self.ego_action_RL < 0:  # LEFT
                            self.ego_action = 3 + (-self.ego_action_RL - 1) * 4
                        elif self.ego_action_RL > 0:  # RIGHT
                            self.ego_action = 4 + (self.ego_action_RL - 1) * 4
                        else:
                            self.ego_action = 0

                        print(f'{self.ego_action_UD =}')
                        print(f'{self.ego_action_RL =}')

            self.draw_score()
            self.draw_highscore()
            if TIMED:
                self.draw_time_bar(time_now=time_now, time_limit=TIME_ACTION_SELECTION, text='Select Action!')
            self.draw_description('Select the action (velocity) using the arrow keys. '
                                  '| Moves de Rigueur if selected, are shown in blue. '
                                  '| Press ENTER to finalise selection.')
            pygame.display.update()

        return self.run

    def run_selected_actions_window(self):
        ax = self.run_gworld.view_selected_actions(ego_action=self.ego_action)
        world_surf = FeARUI.get_pygame_surf_from_ax(ax)
        scaled_world_surf = FeARUI.rescale_photo(world_surf, self.w, self.h - 20)

        if TIMED:
            time_now = 0

        while self.run and self.run_next == 'selected_actions_window':

            if TIMED:
                time_now = time_now + self.clock.get_time()
                if time_now >= TIME_WINDOW:
                    self.run_next = 'fear_score_window'

            self.clock.tick(FPS)
            self.screen.fill((255, 255, 255))  # Fill White
            self.w, self.h = pygame.display.get_surface().get_size()
            self.screen.fill((255, 255, 255))  # Fill White
            self.screen.blit(scaled_world_surf, (0, 10))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.event.clear()
                        self.run = False
                    if event.key == pygame.K_RETURN:
                        pygame.event.clear()
                        self.run_next = 'fear_score_window'
                    if event.key == pygame.K_x:
                        pygame.event.clear()
                        self.game_over = True
                        self.run_next = 'fear_score_window'
                    if event.key == pygame.K_SPACE:
                        pygame.event.clear()
                        if CHEAT_TIMED:
                            time_now = 0

            self.draw_score()
            self.draw_highscore()
            if TIMED:
                self.draw_time_bar(time_now=time_now, time_limit=TIME_WINDOW,
                                   colour=BLUE, time_bar_h=2)
            self.draw_description('Your selected action has been recorded. '
                                  '| The actions chosen by other are also shown. '
                                  '| Press ENTER to view FeAR score.')
            pygame.display.update()

        # self.run_next = 'fear_score_window'
        return self.run

    def run_fear_score_window(self):
        self.run_gworld.calculate_FeAR()
        ego_fear_values = self.run_gworld.FeAR[0]
        fear_score = np.round(-np.sum(ego_fear_values) * FEAR_POINTS).astype(int)
        ax = self.run_gworld.view_selected_actions(ego_action=self.ego_action, fear_values=ego_fear_values)
        pygame.event.clear() # So that inputs while computing FeAR are ignored.

        agent_locations = np.array(self.run_gworld.World.AgentLocations)
        x = agent_locations[:, 1] + 0.5
        y = agent_locations[:, 0] + 0.5

        ax = plotcf.plot_directed_fear_graph(fear=self.run_gworld.FeAR, x0=x, y0=y, fear_threshold=0.05,
                                             ax=ax, game_mode=True)
        world_surf = FeARUI.get_pygame_surf_from_ax(ax)
        scaled_world_surf = FeARUI.rescale_photo(world_surf, self.w, self.h - 20)

        if TIMED:
            time_now = 0

        while self.run and self.run_next == 'fear_score_window':

            if TIMED:
                time_now = time_now + self.clock.get_time()
                if time_now >= FEAR_TIME_WINDOW:
                    self.run_next = 'update_score_window'

            self.clock.tick(FPS)
            self.screen.fill((255, 255, 255))  # Fill White
            self.w, self.h = pygame.display.get_surface().get_size()
            self.screen.fill((255, 255, 255))  # Fill White
            scaled_world_surf = FeARUI.rescale_photo(world_surf, self.w, self.h - 20)
            self.screen.blit(scaled_world_surf, (0, 10))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.event.clear()
                        self.run = False
                    if event.key == pygame.K_RETURN:
                        pygame.event.clear()
                        if self.game_over:
                            self.run_next = 'update_score_window'
                        else:
                            self.run_next = 'update_score_window'
                    if event.key == pygame.K_x:
                        pygame.event.clear()
                        self.game_over = True
                        self.run_next = 'update_score_window'
                    if event.key == pygame.K_SPACE:
                        pygame.event.clear()
                        if CHEAT_TIMED:
                            time_now = 0

            self.draw_score(fear_score)
            self.draw_highscore()
            if TIMED:
                self.draw_time_bar(time_now=time_now, time_limit=FEAR_TIME_WINDOW,
                                   colour=BLUE, time_bar_h=2)
            self.draw_description('FeAR Scores. | Press ENTER to update GWorld. '
                                  '| Press X to quit game.')
            pygame.display.update()

        self.score = self.score + fear_score
        return self.run

    def run_update_score_window(self):
        n_crashes, n_apples_caught = self.run_gworld.gworld_update()
        crash_score = np.round(-n_crashes * CRASH_POINTS).astype(int)
        apple_score = np.round(n_apples_caught * APPLE_POINTS).astype(int)
        ax = self.run_gworld.view_updated_gworld()
        world_surf = FeARUI.get_pygame_surf_from_ax(ax)

        if TIMED:
            time_now = 0

        while self.run and self.run_next == 'update_score_window':

            if TIMED:
                time_now = time_now + self.clock.get_time()
                if time_now >= TIME_WINDOW:
                    if self.game_over:
                        self.run_next = 'game_over_window'
                    else:
                        self.run_next = 'action_selection_window'

            self.clock.tick(FPS)
            self.screen.fill((255, 255, 255))  # Fill White
            self.w, self.h = pygame.display.get_surface().get_size()
            self.screen.fill((255, 255, 255))  # Fill White
            scaled_world_surf = FeARUI.rescale_photo(world_surf, self.w, self.h - 20)
            self.screen.blit(scaled_world_surf, (0, 10))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.event.clear()
                        self.run = False
                    if event.key == pygame.K_RETURN:
                        pygame.event.clear()
                        if self.game_over:
                            self.run_next = 'game_over_window'
                        else:
                            self.run_next = 'action_selection_window'
                    if event.key == pygame.K_x:
                        pygame.event.clear()
                        self.game_over = True
                        self.run_next = 'game_over_window'
                    if event.key == pygame.K_SPACE:
                        pygame.event.clear()
                        if CHEAT_TIMED:
                            time_now = 0

            self.draw_score(crash_score + apple_score)
            self.draw_highscore()
            if TIMED:
                self.draw_time_bar(time_now=time_now, time_limit=TIME_WINDOW,
                                   colour=BLUE, time_bar_h=2)
            self.draw_description('Update Scores (from collisions and stars). | Press ENTER to continue playing. '
                                  '| Press X to quit game.')
            pygame.display.update()

        self.score = self.score + crash_score + apple_score

        self.trials = self.trials + 1
        if self.trials >= self.n_trials:
            print('Game Over !')
            self.game_over = True
            self.trials = self.n_trials

        return self.run

    def run_game_over_window(self):
        name_text = self.game_font.render(('Game of FeAR'), True, RED)
        tw_name, th_name = name_text.get_size()

        if self.highscores[self.game_level] < self.score:
            self.highscores[self.game_level] = self.score
            pretty_print_highscores = pprint.pformat(self.highscores, width=50).replace("'", '"')
            with open('highscores.json', 'w') as f:
                f.write(pretty_print_highscores)

        while self.run and self.run_next == 'game_over_window':

            self.clock.tick(FPS)
            self.screen.fill((255, 255, 255))  # Fill White
            self.w, self.h = pygame.display.get_surface().get_size()
            self.screen.fill((255, 255, 255))  # Fill White
            self.screen.blit(name_text, ((self.w - tw_name) // 2, (self.h - th_name) // 3))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.event.clear()
                        self.run = False
                    if event.key == pygame.K_RETURN:  # Enter Slider UI for Sequence of Iterations
                        pygame.event.clear()
                        self.run_next = 'start_window'

            self.draw_score(h_offset=((self.h - th_name) // 3) + th_name)
            self.draw_highscore()
            self.draw_description('GAME OVER | Press ENTER to restart.')
            pygame.display.update()

        # self.run_next = 'leaderboard_window'
        return self.run

    def leaderboard_window(self):
        return self.run

    def draw_score(self, fear_score=None, h_offset=0):
        score_text = self.font.render(('SCORE : ' + str(self.score)), True, BLACK)
        tw_score, th_score = score_text.get_size()
        self.screen.blit(score_text, ((self.w - tw_score) // 2, 20 + h_offset))
        if fear_score is not None:
            if fear_score > 0:
                fear_score_text = self.font.render('+ ' + str(np.abs(fear_score)), True, BLUE)
            elif fear_score < 0:
                fear_score_text = self.font.render('- ' + str(np.abs(fear_score)), True, RED)
            else:
                fear_score_text = self.font.render('+ ' + str(np.abs(fear_score)), True, BLACK)
            self.screen.blit(fear_score_text, (20 + (self.w - tw_score) // 2 + tw_score, 20 + h_offset))

        self.draw_progress_bar()

    def draw_highscore(self, h_offset=0):
        score_text = self.tiny_font.render(f'HIGH SCORE : {self.highscores[self.game_level]}', True, BLACK)
        tw_score, th_score = score_text.get_size()
        self.screen.blit(score_text, ((self.w - tw_score)-20, 20 + h_offset))

    def draw_description(self, content='', h=-22):
        text = self.tiny_font.render(content, True, BLACK)
        tw_, th_ = text.get_size()
        self.screen.blit(text, ((self.w - tw_) // 2, (self.h - th_) + h))

    def draw_progress_bar(self):
        pb_padding = 3
        pb_content = f'Trial ( {self.trials} / {self.n_trials} )'
        pb_text = self.micro_font.render(pb_content, True, WHITE)
        tw_pb, th_pb = pb_text.get_size()
        self.progress_bar_h = th_pb + 2 * pb_padding
        progress_bar_w = (self.trials / self.n_trials) * self.w
        pygame.draw.rect(self.screen, BLUE, pygame.Rect(0, self.h - self.progress_bar_h,
                                                        progress_bar_w, self.progress_bar_h))
        self.screen.blit(pb_text, (10, self.h - self.progress_bar_h + pb_padding + 1))

    def draw_time_bar(self, time_now, time_limit, colour=GOLD, time_bar_h=10, text=None):
        if TIMED:
            if text is not None:
                tb_padding = 3
                time_bar_text = self.micro_font.render(text, True, WHITE)
                tw_time_bar, th_time_bar = time_bar_text.get_size()
                time_bar_h = max(time_bar_h, th_time_bar + 2 * tb_padding)
            time_bar_w = 20 + (time_now / time_limit ) * (self.w-20)
            # pygame.draw.rect(self.screen, GOLD, pygame.Rect(0, self.h - self.progress_bar_h - time_bar_h,
            #                                                 time_bar_w, time_bar_h))
            pygame.draw.rect(self.screen, colour, pygame.Rect(0, 0, time_bar_w, time_bar_h))

            if text is not None:
                self.screen.blit(time_bar_text, (10, 0 + tb_padding + 1))


class runGWorld:
    def __init__(self, game_level='GameMap_8'):
        self.scenario_name = game_level

        self.Scenario = GWorld.LoadJsonScenario(json_filename='Scenarios4GameOfFeAR_.json',
                                                scenario_name=self.scenario_name)
        self.N_Agents = self.Scenario['N_Agents']

        self.ActionNames, self.ActionMoves = Agent.DefineActions()

        self.region = np.array(self.Scenario['Map']['Region'])

        # Dictionary of Policies
        self.policy_map = np.zeros(np.shape(self.region), dtype=int)
        self.policies = self.Scenario['Policies']
        print(f'policies = \n{pprint.pformat(self.policies)}')

        # Update PolicyMap
        policy_keys = self.policies.keys()
        print(f'{policy_keys =}')
        for key in policy_keys:
            slicex = self.policies[key]['slicex']
            slicey = self.policies[key]['slicey']
            self.policy_map[slicex, slicey] = key
        print(f'Region =\n {self.region}')
        print(f'policyMap =\n {self.policy_map}')

        # Dictionary of MdRs
        self.mdr_map = np.zeros(np.shape(self.region), dtype=int)
        self.mdrs = self.Scenario['MdRs']
        print(f'mdrs = \n{pprint.pformat(self.mdrs)}')

        # Update MdRMap
        mdrs_keys = self.mdrs.keys()
        print(f'{mdrs_keys =}')
        for key in mdrs_keys:
            slicex = self.mdrs[key]['slicex']
            slicey = self.mdrs[key]['slicey']
            self.mdr_map[slicex, slicey] = key
        print(f'Region =\n {self.region}')
        print(f'mdr_map =\n {self.mdr_map}')

        # Running Simulation Cases !

        # Initialising World Map
        Walls = self.Scenario['Map']['Walls']
        OneWays = self.Scenario['Map']['OneWays']

        self.World = GWorld.GWorld(self.region, Walls=Walls, OneWays=OneWays)  # Initialising GWorld

        self.AgentLocations = []
        for location in self.Scenario['AgentLocations']:
            self.AgentLocations.append(tuple(location))

        # Adding nn Agents at sorted random positions
        if len(self.AgentLocations) < self.N_Agents:
            [locX, locY] = np.where(self.region == 1)
            LocIdxs = rng.choice(locX.shape[0], size=(self.N_Agents - len(self.AgentLocations)), replace=False,
                                 shuffle=False)
            LocIdxs.sort()
            for Idx in LocIdxs:
                self.AgentLocations.append((locX[Idx], locY[Idx]))

        # Adding Agents
        PreviousAgentAdded = True
        for location in self.AgentLocations:
            # Adding new Agents if Previous Agent was Added to the World
            if PreviousAgentAdded:
                Ag_i = Agent.Agent()
            PreviousAgentAdded = self.World.AddAgent(Ag_i, location, printStatus=False)

        PreviousAgentAdded = True
        while len(self.World.AgentList) < self.N_Agents:
            # Adding new Agents if Previous Agent was Added to the World
            if PreviousAgentAdded:
                Ag_i = Agent.Agent()
            Loc_i = (np.random.randint(self.region.shape[0]), np.random.randint(self.region.shape[1]))
            PreviousAgentAdded = self.World.AddAgent(Ag_i, Loc_i, printStatus=False)

        # Action Selection for Agents
        self.defaultAction = self.Scenario['defaultAction']
        self.SpecificAction4Agents = self.Scenario['SpecificAction4Agents']

        print('SpecificAction4Agents :', self.SpecificAction4Agents)

        # ------------------------------------------------------------------------------------------------------------------

        # Setting Policy for all Agents

        # Updating Agent Policies in World
        for ii, agent in enumerate(self.World.AgentList):
            agent_location = self.World.AgentLocations[ii]
            agent_policy = str(self.policy_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_stepWeights = self.policies[agent_policy]['stepWeights']
            agent_directionWeights = self.policies[agent_policy]['directionWeights']

            print(f'{agent_location =}, {agent_policy =}')
            print(f'{agent_stepWeights = }')
            print(f'{agent_directionWeights = }')

            policy = Agent.GeneratePolicy(StepWeights=agent_stepWeights, DirectionWeights=agent_directionWeights)
            agent.UpdateActionPolicy(policy)

        # ------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        # Move de Rigueur

        self.MdR4Agents = []

        for ii in range(len(self.World.AgentList)):
            agent_location = self.World.AgentLocations[ii]
            agent_mdr_key = str(self.mdr_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_mdr = self.mdrs[agent_mdr_key]['mdr']
            self.MdR4Agents.append([ii, agent_mdr])
            print(f'{agent_location =}, {agent_mdr =}')

        print('MdR4Agents : ', self.MdR4Agents)
        mdr_string = FeARUI.get_mdr_string(self.MdR4Agents, return_names=True)
        print('MdRs: ', mdr_string)

        self.Action4Agents = self.World.SelectActionsForAll(defaultAction='stay')
        self.FeAR = []
        # self.FeAL = []

        valid_locations = np.transpose(np.where(self.region > 0))
        random_indices = np.random.choice(len(valid_locations), N_APPLES, replace=False)
        self.apples = valid_locations[random_indices]

        fig, self.ax = plt.subplots()

    def gworld_iteration(self):

        # Iterations

        # ------------------------------------------------------------------------------------------------------------------

        self.MdR4Agents = []  # Resetting MdR4Agents

        for ii, agent in enumerate(self.World.AgentList):
            agent_location = self.World.AgentLocations[ii]

            # Updating Policies of Agents
            agent_policy = str(self.policy_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_stepWeights = self.policies[agent_policy]['stepWeights']
            agent_directionWeights = self.policies[agent_policy]['directionWeights']

            print(f'{agent_location =}, {agent_policy =}')
            print(f'{agent_stepWeights = }')
            print(f'{agent_directionWeights = }')

            # Updating MdRs of Agents
            agent_mdr_key = str(self.mdr_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_mdr = self.mdrs[agent_mdr_key]['mdr']
            self.MdR4Agents.append([ii, agent_mdr])
            print(f'{agent_location =}, {agent_mdr =}')
            print('MdR4Agents : ', self.MdR4Agents)
            mdr_string = FeARUI.get_mdr_string(self.MdR4Agents, return_names=True)
            print('MdRs: ', mdr_string)

            policy = Agent.GeneratePolicy(StepWeights=agent_stepWeights, DirectionWeights=agent_directionWeights)
            agent.UpdateActionPolicy(policy)

        # Select Actions for Agents based on defaultAction and SpecificAction4Agents
        self.Action4Agents = self.World.SelectActionsForAll(defaultAction=self.defaultAction,
                                                            InputActionID4Agents=self.SpecificAction4Agents)
        print('SpecificAction Inputs 4Agents :', self.SpecificAction4Agents)
        print('Actions chosen for Agents :', self.Action4Agents)

        # ------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

    def calculate_FeAR(self, use_panic=False):
        # Responsibility

        if use_panic:
            # PANIC for FeAR
            self.FeAR, _, _, _, _ = Responsibility.panic_4_fear(self.World, self.Action4Agents, self.MdR4Agents)

        else:
            # Calculate Responsibility Metric for the chosen Actions
            self.FeAR, _, _, _, _ = Responsibility.FeAR(self.World, self.Action4Agents, self.MdR4Agents)

            # self.FeAR, _, _, _, _ = Responsibility.FeAR_4_one_actor(self.World, self.Action4Agents, self.MdR4Agents,
            #                                                         actor_ii=0)
            # self.FeAL, _, _, _, _ = Responsibility.FeAL(self.World, self.Action4Agents, self.MdR4Agents)



    def gworld_update(self):
        # Update World with Selected Steps
        agent_crashes, restricted_moves, self.apples, apples_caught = \
            self.World.UpdateGWorld(ActionID4Agents=self.Action4Agents, apples=self.apples, apple_eaters=[0])

        if REMOVE_COLLIDING_AGENTS:
            agents_to_remove = []
            for agent in range(1, len(agent_crashes)):  # Skip the ego agent !
                if agent_crashes[agent] == 1:
                    agents_to_remove.append(agent)

            # Reverse the order to prevent messing the ids
            agents_to_remove.sort(reverse=True)
            for agent in agents_to_remove:
                self.World.RemoveAgent(agentID=agent)

        if agent_crashes[0]:
            n_crashes = 1
        elif restricted_moves[0]:
            n_crashes = 1
        else:
            n_crashes = 0

        n_apples_caught = len(apples_caught)

        print(f'{n_apples_caught=}')

        return n_crashes, n_apples_caught

    def view_ego_action(self, ego_action=0):
        # Specify actions for all agents - only ego moves
        SpecificAction4Agents = [[0, ego_action]]
        self.SpecificAction4Agents = self.World.SelectActionsForAll(defaultAction='stay',
                                                                    InputActionID4Agents=SpecificAction4Agents)
        self.gworld_iteration()
        ego_mdr = self.MdR4Agents[0][1]

        if ego_mdr == ego_action:
            ego_mdr_colour = True
        else:
            ego_mdr_colour = False

        # Plotting the State of the World and Chosen Actions for the next iteration
        ax = plotgw.ViewGWorld(self.World, ViewNextStep=True, ViewActionTrail=False, ax=self.ax,
                               Animate=True, mdr_colour=ego_mdr_colour,
                               game_mode=True, apples=self.apples)
        return ax

    def view_selected_actions(self, ego_action=0, fear_values=None):
        # For all the actions
        # Specify ego actions only
        self.SpecificAction4Agents = [[0, ego_action]]  # Only specify the ego action
        # Plotting the State of the World and Chosen Actions for the next iteration

        if fear_values is None: # Don't update actions if actions were already chosen.
            self.gworld_iteration()

        if fear_values is not None and len(fear_values) == self.N_Agents:
            ax = plotgw.ViewGWorld(self.World, ViewNextStep=True, ViewActionTrail=False, apples=self.apples,
                                   colour_by_fear=True, fear_values=fear_values, Animate=True,
                                   ax=self.ax, game_mode=True)
        elif fear_values is not None:
            print('NUmber of FeAR values passed in do not match the number of agents')
            ax = plotgw.ViewGWorld(self.World, ViewNextStep=True, ViewActionTrail=False, ax=self.ax,
                                   game_mode=True, apples=self.apples, Animate=True)
        else:
            ax = plotgw.ViewGWorld(self.World, ViewNextStep=True, ViewActionTrail=False,
                                   game_mode=True,  ax=self.ax, Animate=True,
                                   apples=self.apples)

        return ax

    def view_updated_gworld(self):
        ax = plotgw.ViewGWorld(self.World, ViewActionArrows=False, ViewActionTrail=False,
                               game_mode=True, ax=self.ax, Animate=True, apples=self.apples)
        return ax


if __name__ == "__main__":
    main()
