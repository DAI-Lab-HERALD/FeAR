import numpy as np

np.random.seed(0)

import seaborn as sns
sns.set_theme(style="ticks")
import matplotlib.pyplot as plt

import json
from json.decoder import JSONDecodeError
import pprint
import os
import glob

import GWorld
import Agent
import Responsibility
import Emergence
import PlotGWorld

rng = np.random.default_rng(seed=0)
plotgw = PlotGWorld.PlotGWorld();  # Object for accessing plotters

import matplotlib.backends.backend_agg as agg
import pygame
from pygame.locals import *

# -------------------------------------------------------------------------------------------------------------------- #
# Reading Records stored as JSONs
RECORD_NAME = '22-07-23 18-44-43 SingleLane10-2Agents-AgentsAt-3-5_S0-S0_ex.json'
# The first record to be plotted if it exists.
# -------------------------------------------------------------------------------------------------------------------- #

pygame.init()

# DISPLAY_FONT = pygame.font.Font('LinLibertine_RB.ttf', 20, bold=True)
# DISPLAY_FONT = pygame.font.SysFont('couriernew', 20, bold=True)
DISPLAY_FONT = pygame.font.SysFont('sitkabanner', 20, bold=True)

# DISPLAY_FONT_TINY = pygame.font.Font('LinLibertine_RB.ttf', 15, bold=False)
# DISPLAY_FONT_TINY = pygame.font.SysFont('sitkabanner', 15, bold=False)
DISPLAY_FONT_TINY = pygame.font.SysFont('consolas', 15, bold=False)
# DISPLAY_FONT_TINY = pygame.font.SysFont('bahnschrift', 15, bold=False)


RECORD_FOLDER = 'Results'
SAVE_IMAGES_TO = 'Plots'
OVERWRITE_IMAGES = True


JSON_4_PLOTS = 'Instances2plot.json'
# JSON_4_PLOTS = 'Instances2plot_finer.json'

FINER = True

N_TOP = 10

# CMAP_4COUNT = sns.cubehelix_palette(as_cmap=True, light=1)
hue_validMoves = 190
CMAP_4COUNT = sns.diverging_palette(365 - hue_validMoves, hue_validMoves, l=60, as_cmap=True)

ANNOTATE_FONT_SIZE = 12

BLACK = (100, 100, 100)
RED = (150, 10, 30)
BLUE = (0, 128, 255)
TEXT_PADDING = 15
PYGAME_WIN_SIZE = (1080, 600)

FPS = 5


def main():
    list_of_record_names = get_list_of_records(RECORD_FOLDER)

    if not RECORD_NAME:
        record_name_id = 0
    elif RECORD_NAME in list_of_record_names:
        record_name_id = list_of_record_names.index(RECORD_NAME)
    else:
        record_name_id = 0

    # ---------------------------------------------------------------------------------------------------------------- #
    ui = ui_FeAR(list_of_record_names, record_name_id)
    ui.run_ui_for_viewing_results()

    pass


def find_instances_with_extreme_ssq_of_FeAR(N_Cases, N_iterations, ReadRecord):
    maxCaseDigits = np.ceil(np.log10(N_Cases * N_iterations)).astype(int)
    print('N_Cases: ', N_Cases)
    print('N_iterations: ', N_iterations)
    print('maxCaseDigits: ', maxCaseDigits)

    # Sum of Squares of FeAR
    SSq_FeAR = {}
    SSq_FeAR_minusDiag = {}
    SSq_FeAR_positive = {}
    SSq_FeAR_negative = {}

    for jj in range(N_Cases * N_iterations):
        InstanceID = 'Instance_' + str(jj).zfill(maxCaseDigits)
        FeAR = np.array(ReadRecord[InstanceID]['FeAR'])

        ReadRecord[InstanceID]['SSq_FeAR'] = np.sum(FeAR ** 2)
        #     print(ReadRecord[InstanceID]['SSq_FeAR'])
        SSq_FeAR[InstanceID] = ReadRecord[InstanceID]['SSq_FeAR']

        # Finding Elements close to the Diagonal
        Diag_plus1 = np.where(np.eye(FeAR.shape[0], k=1) == 1)
        Diag_minus1 = np.where(np.eye(FeAR.shape[0], k=-1) == 1)

        # Removing the elements close to the diagonals to study indirect influences
        FeAR_minusDiag = FeAR.copy()
        FeAR_minusDiag[Diag_plus1] = 0
        FeAR_minusDiag[Diag_minus1] = 0

        ReadRecord[InstanceID]['SSq_FeAR_minusDiag'] = np.sum(FeAR_minusDiag ** 2)
        SSq_FeAR_minusDiag[InstanceID] = ReadRecord[InstanceID]['SSq_FeAR_minusDiag']

        FeAR_positive = np.clip(FeAR, a_min=0, a_max=None)
        FeAR_negative = np.clip(FeAR, a_min=None, a_max=0)

        ReadRecord[InstanceID]['SSq_FeAR_positive'] = np.sum(FeAR_positive ** 2)
        SSq_FeAR_positive[InstanceID] = ReadRecord[InstanceID]['SSq_FeAR_positive']

        ReadRecord[InstanceID]['SSq_FeAR_negative'] = np.sum(FeAR_negative ** 2)
        SSq_FeAR_negative[InstanceID] = ReadRecord[InstanceID]['SSq_FeAR_negative']

    Max_N_SSqs = sorted(SSq_FeAR, key=SSq_FeAR.get, reverse=True)[:N_TOP]
    print('Max_N_SSqs', Max_N_SSqs)
    # for ii in Max_N_SSqs:
    #     print(SSq_FeAR[ii])

    Min_N_SSqs = sorted(SSq_FeAR, key=SSq_FeAR.get, reverse=False)[:N_TOP]
    print('Min_N_SSqs', Min_N_SSqs)
    # for ii in Min_N_SSqs:
    #     print(SSq_FeAR[ii])

    Max_N_SSqs_positive = sorted(SSq_FeAR_positive, key=SSq_FeAR_positive.get, reverse=True)[:N_TOP]
    print('Max_N_SSqs_positive', Max_N_SSqs_positive)
    # for ii in Max_N_SSqs_positive:
    #     print(SSq_FeAR_positive[ii])

    Min_N_SSqs_positive = sorted(SSq_FeAR_positive, key=SSq_FeAR_positive.get, reverse=False)[:N_TOP]
    # Min_N_SSqs_positive = sorted(SSq_FeAR_positive, reverse=False)[-N_TOP:]
    print('Min_N_SSqs_positive', Min_N_SSqs_positive)
    # for ii in Min_N_SSqs_positive:
    #     print(SSq_FeAR_positive[ii])

    Max_N_SSqs_negative = sorted(SSq_FeAR_negative, key=SSq_FeAR_negative.get, reverse=True)[:N_TOP]
    print('Max_N_SSqs_negative', Max_N_SSqs_negative)
    Min_N_SSqs_negative = sorted(SSq_FeAR_negative, key=SSq_FeAR_negative.get, reverse=False)[:N_TOP]
    print('Min_N_SSqs_negative', Min_N_SSqs_negative)

    Max_N_SSqs_minusDiag = sorted(SSq_FeAR_minusDiag, key=SSq_FeAR_minusDiag.get, reverse=True)[:N_TOP]
    print('Max_N_SSqs_minusDiag', Max_N_SSqs_minusDiag)
    Min_N_SSqs_minusDiag = sorted(SSq_FeAR_minusDiag, key=SSq_FeAR_minusDiag.get, reverse=False)[:N_TOP]
    print('Min_N_SSqs_minusDiag', Min_N_SSqs_minusDiag)

    list_of_instance_id_lists = [Max_N_SSqs, Min_N_SSqs, Max_N_SSqs_positive, Min_N_SSqs_positive,
                                 Max_N_SSqs_negative, Min_N_SSqs_negative, Max_N_SSqs_minusDiag, Min_N_SSqs_minusDiag]
    instance_id_set_names = ['Max_N_SSqs', 'Min_N_SSqs', 'Max_N_SSqs_positive', 'Min_N_SSqs_positive',
                             'Max_N_SSqs_negative', 'Min_N_SSqs_negative',
                             'Max_N_SSqs_minusDiag', 'Min_N_SSqs_minusDiag']

    return list_of_instance_id_lists, instance_id_set_names


def find_list_of_instances_per_case(N_Cases, N_iterations):
    maxCaseDigits = np.ceil(np.log10(N_Cases)).astype(int)
    maxInstanceDigits = np.ceil(np.log10(N_Cases * N_iterations)).astype(int)

    instance_id_set_names = []
    list_of_instance_id_lists = []

    for jj in range(N_Cases):
        caseID = 'Case_' + str(jj).zfill(maxCaseDigits)
        instance_id_set_names.append(caseID)

        instance_id_lists = []
        for kk in range(N_iterations):
            InstanceID = 'Instance_' + str(jj * N_iterations + kk).zfill(maxInstanceDigits)
            instance_id_lists.append(InstanceID)

        list_of_instance_id_lists.append(instance_id_lists.copy())

    return list_of_instance_id_lists, instance_id_set_names


# Function to Plot Results for Instances from a List
def plot_results_for_instances(InstanceID, ReadRecord, mdr_string='', saveFolder=None, record_name='Record',
                               overwrite_images='False'):
    record_instance_name = record_name + '_mdr-' + mdr_string + '_' + InstanceID

    Scenario = ReadRecord['Scenario']

    finer = FINER

    if saveFolder is not None:
        for_print = True
    else:
        for_print = False

    print(InstanceID)

    # Initialising World Map

    Region = np.array(Scenario['Map']['Region'])
    Walls = Scenario['Map']['Walls']
    OneWays = Scenario['Map']['OneWays']

    # Initialising GWorld from World Map
    World = GWorld.GWorld(Region, Walls=Walls, OneWays=OneWays)

    # Adding Agents
    AgentLocations = ReadRecord[InstanceID]["AgentLocations"]

    PreviousAgentAdded = True
    for locationArray in AgentLocations:
        # Adding new Agents if Previous Agent was Added to the World
        location = tuple(locationArray)  # Converting to Tuples
        if PreviousAgentAdded:
            Ag_i = Agent.Agent()
        PreviousAgentAdded = World.AddAgent(Ag_i, location, printStatus=False)

    # Move de Rigueur
    MdR4Agents = ReadRecord[InstanceID]["MdR4Agents"]
    print('MdR4Agents : ', MdR4Agents)

    # Select Actions for Agents based on defaultAction and SpecificAction4Agents
    SpecificAction4Agents = ReadRecord[InstanceID]["Action4Agents"]
    Action4Agents = World.SelectActionsForAll(InputActionID4Agents=SpecificAction4Agents)
    print('SpecificAction Inputs 4Agents :', SpecificAction4Agents)
    print('Actions chosen for Agents :', Action4Agents)

    FeAR = np.array(ReadRecord[InstanceID]['FeAR'])
    ValidMoves_MdR = np.array(ReadRecord[InstanceID]['ValidMoves_MdR'])
    ValidMoves_action1 = np.array(ReadRecord[InstanceID]['ValidMoves_action1'])

    world_ax = plotgw.ViewGWorld(World, ViewNextStep=True, ViewActionArrows=True, ViewActionTrail=False,
                                 annot_font_size=ANNOTATE_FONT_SIZE, saveFolder=saveFolder, Animate=True,
                                 imageName=record_instance_name + '_GW', overwrite_image=overwrite_images);

    map_size_x = np.size(World.WorldState, 0)
    if map_size_x == 1:
        only_horizontal = True
    else:
        only_horizontal = False

    if saveFolder:
        fear_title = 'FeAR'
    else:
        fear_title = (InstanceID + ' FeAR')

    N_Agents = Scenario['N_Agents']

    if N_Agents == 2:
        fmt = '0.2f'
    else:
        fmt = '0.1f'

    if N_Agents >= 4:
        finer = True

    if ReadRecord[InstanceID].get('FeAL'):
        fear_ax = PlotGWorld.plotResponsibility(FeAR, FeAL=np.array(ReadRecord[InstanceID]['FeAL']),
                                                annot_font_size=ANNOTATE_FONT_SIZE, plot_feal_separately=False,
                                                title=fear_title, for_print=for_print, fmt=fmt, finer=finer,
                                                saveFolder=saveFolder, imageName=record_instance_name + '_FeAR',
                                                overwrite_image=overwrite_images)
    else:
        fear_ax = PlotGWorld.plotResponsibility(FeAR,
                                                annot_font_size=ANNOTATE_FONT_SIZE,
                                                title=fear_title, for_print=for_print, fmt=fmt, finer=finer,
                                                saveFolder=saveFolder, imageName=record_instance_name + '_FeAR',
                                                overwrite_image=overwrite_images)

    if (N_Agents == 2) and ReadRecord[InstanceID].get('ValidityOfMoves_Mdr'):
        validity_of_moves_mdr = np.array(ReadRecord[InstanceID]['ValidityOfMoves_Mdr'])

        if for_print:
            fig, count_mdr_axs = plt.subplots(2, 1, layout="constrained")
        else:
            fig, count_mdr_axs = plt.subplots(1, 2)

        print('validity_of_moves_mdr', validity_of_moves_mdr)

        if for_print:
            title12_mdr = 'Valid actions of 1\n for MdR of 2'
            title21_mdr = 'Valid actions of 2\n for MdR of 1'
            title12_move = 'Valid actions of 1\n for Move of 2'
            title21_move = 'Valid actions of 2\n for Move of 1'
        else:
            title12_mdr = 'Valid actions of 1 for MdR of 2'
            title21_mdr = 'Valid actions of 2 for MdR of 1'
            title12_move = 'Valid actions of 1 for Move of 2'
            title21_move = 'Valid actions of 2 for Move of 1'

        PlotGWorld.plot_valid_actions(validity_of_moves_mdr[1][0], ax=count_mdr_axs[0],
                                      title=title12_mdr, only_horizontal=only_horizontal,
                                      overwrite_image=overwrite_images)
        PlotGWorld.plot_valid_actions(validity_of_moves_mdr[0][1], ax=count_mdr_axs[1], for_print=for_print,
                                      title=title21_mdr, only_horizontal=only_horizontal,
                                      saveFolder=saveFolder, imageName=record_instance_name + '_validActions_mdr',
                                      overwrite_image=overwrite_images)

        count_mdr_ax = count_mdr_axs[0]

        # ---

        validity_of_moves_action = np.array(ReadRecord[InstanceID]['ValidityOfMoves_action1'])

        if for_print:
            fig, count_move_axs = plt.subplots(2, 1, layout="constrained")
        else:
            fig, count_move_axs = plt.subplots(1, 2)

        PlotGWorld.plot_valid_actions(validity_of_moves_action[1][0], ax=count_move_axs[0],
                                      title=title12_move, only_horizontal=only_horizontal,
                                      overwrite_image=overwrite_images)
        PlotGWorld.plot_valid_actions(validity_of_moves_action[0][1], ax=count_move_axs[1], for_print=for_print,
                                      title=title21_move, only_horizontal=only_horizontal,
                                      saveFolder=saveFolder, imageName=record_instance_name + '_validActions_move',
                                      overwrite_image=overwrite_images)

        count_move_ax = count_move_axs[0]

    else:
        if ReadRecord[InstanceID].get('FeAL'):
            ValidMoves_action_FeAL = np.array(ReadRecord[InstanceID]['ValidMoves_action_FeAL']).astype(int)
            ValidMoves_MdR_FeAL = np.array(ReadRecord[InstanceID]['ValidMoves_moveDeRigueur_FeAL']).astype(int)
            count_mdr_ax = PlotGWorld.plotCounts(ValidMoves_MdR, counts_feal=ValidMoves_MdR_FeAL,
                                                 title='Number of valid moves \nof Affected if \nActor does MdR ',
                                                 cmap=CMAP_4COUNT, annot_font_size=ANNOTATE_FONT_SIZE,
                                                 saveFolder=saveFolder, for_print=for_print, finer=finer,
                                                 imageName=record_instance_name + '_validActions_mdr',
                                                 overwrite_image=overwrite_images);
            count_move_ax = PlotGWorld.plotCounts(ValidMoves_action1, counts_feal=ValidMoves_action_FeAL,
                                                  title='Number of valid moves \nof Affected for \nActor\'s chosen move.',
                                                  cmap=CMAP_4COUNT, annot_font_size=ANNOTATE_FONT_SIZE,
                                                  saveFolder=saveFolder, for_print=for_print, finer=finer,
                                                  imageName=record_instance_name + '_validActions_move',
                                                  overwrite_image=overwrite_images);
        else:
            count_mdr_ax = PlotGWorld.plotCounts(ValidMoves_MdR,
                                                 title='Number of valid moves \nof Affected if \nActor does MdR ',
                                                 cmap=CMAP_4COUNT, annot_font_size=ANNOTATE_FONT_SIZE,
                                                 saveFolder=saveFolder, for_print=for_print,
                                                 imageName=record_instance_name + '_validActions_mdr',
                                                 overwrite_image=overwrite_images);
            count_move_ax = PlotGWorld.plotCounts(ValidMoves_action1,
                                                  title='Number of valid moves \nof Affected for \nActor\'s chosen move.',
                                                  cmap=CMAP_4COUNT, annot_font_size=ANNOTATE_FONT_SIZE,
                                                  saveFolder=saveFolder, for_print=for_print,
                                                  imageName=record_instance_name + '_validActions_move',
                                                  overwrite_image=overwrite_images);

    del World

    return world_ax, fear_ax, count_mdr_ax, count_move_ax


def plot_results_for_instanceList(List_of_InstanceIDs, ReadRecord):
    for instance_id in List_of_InstanceIDs:
        plot_results_for_instances(instance_id, ReadRecord)


class ui_FeAR:

    def __init__(self, list_of_record_names=None, record_name_id=None):
        if list_of_record_names is None:
            print("list_of_record_names not found!")
            return

        if record_name_id is None:
            print("record_name_id not found!")
            return

        self.list_of_record_names = list_of_record_names
        self.record_name_id = record_name_id

        self.ReadRecord, self.record_name, self.N_Cases, self.N_iterations = \
            read_json_record(self.list_of_record_names, self.record_name_id)
        # ----------------------------------------------------------------------------------------------------------- #

        self.list_of_instance_id_lists, self.instance_id_set_names = \
            find_instances_with_extreme_ssq_of_FeAR(N_Cases=self.N_Cases, N_iterations=self.N_iterations,
                                                    ReadRecord=self.ReadRecord)

        if self.list_of_instance_id_lists is None:
            print("list_of_instance_id_lists not found!")
            return
        if self.instance_id_set_names is None:
            print("instance_id_set_names not found!")
            return
        if self.ReadRecord is None:
            print("ReadRecord not found!")
            return

        self.instance_id_set = 0
        self.instance_id_ii = 0
        self.InstanceID = self.list_of_instance_id_lists[self.instance_id_set][self.instance_id_ii]

        self.MdR4Agents = self.ReadRecord[self.InstanceID]["MdR4Agents"]
        MdR_id = self.MdR4Agents[0][1]  # Assuming that all agents have the same MdR
        ActionList, _ = Agent.DefineActions()
        self.MdR = ActionList[MdR_id]

        self.mdr_string = get_mdr_string(self.MdR4Agents, return_names=True)
        print('MdR String : ', self.mdr_string)

        self.font = DISPLAY_FONT
        self.tiny_font = DISPLAY_FONT_TINY

        window = pygame.display.set_mode(PYGAME_WIN_SIZE, pygame.RESIZABLE, DOUBLEBUF)
        self.screen = pygame.display.get_surface()

        pygame.display.flip()

        self.clock = pygame.time.Clock()
        self.w, self.h = pygame.display.get_surface().get_size()
        self.count_mdr_surf, self.count_move_surf, self.fear_surf, self.world_surf = get_new_axs(
            self.list_of_instance_id_lists, self.ReadRecord, self.instance_id_set, self.instance_id_ii, self.screen)

        self.old_record_name_id = self.record_name_id
        self.old_instance_id_set = self.instance_id_set
        self.old_instance_id_ii = self.instance_id_ii

        return

    def run_ui_instances_per_case(self):
        self.instance_id_ii = 0  # Reset
        self.instance_id_set = 0  # Reset

        self.list_of_instance_id_lists, self.instance_id_set_names = \
            find_list_of_instances_per_case(N_Cases=self.N_Cases, N_iterations=self.N_iterations)
        self.draw_text_and_update_changes_iteration_sequence(update_now=True)
        self.draw_surfaces()

        run = True
        while run:

            self.draw_surfaces()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        run = False

                    if event.key == pygame.K_s:  # Go back to showing extreme results
                        print('Returning to Extremes Window!')
                        return run

                    self.key_event_manager_4_navigation(event)

            self.draw_text_and_update_changes_iteration_sequence()

            pygame.display.update()

        return run

    def run_ui_for_viewing_results(self):
        self.instance_id_ii = 0  # Reset
        self.instance_id_set = 0  # Reset

        run = True
        while run:

            self.draw_surfaces()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        run = False

                    if event.key == pygame.K_s:  # Enter Slider UI for Sequence of Iterations
                        pygame.event.clear()
                        print('Going to Sequence Window')
                        run = self.run_ui_instances_per_case()
                        print('Returned from Sequence Window')

                        self.instance_id_ii = 0  # Reset
                        self.instance_id_set = 0  # Reset
                        self.list_of_instance_id_lists, self.instance_id_set_names = \
                            find_instances_with_extreme_ssq_of_FeAR(self.N_Cases, self.N_iterations, self.ReadRecord)
                        self.draw_text_and_update_changes(update_now=True)
                        self.draw_surfaces()

                    self.key_event_manager_4_navigation(event)

            self.draw_text_and_update_changes()

            pygame.display.update()

        pygame.quit()

    def key_event_manager_4_navigation(self, event):
        if event.key == pygame.K_UP:
            self.instance_id_ii -= 1
            if self.instance_id_ii < 0:
                self.instance_id_ii = 0
            pygame.event.clear()
            # count_mdr_surf, count_move_surf, fear_surf, world_surf = get_new_axs(
            #     list_of_instance_id_lists, ReadRecord, instance_id_set, instance_id_ii, screen)
        if event.key == pygame.K_DOWN:
            self.instance_id_ii += 1
            if self.instance_id_ii >= len(self.list_of_instance_id_lists[self.instance_id_set]):
                self.instance_id_ii = len(self.list_of_instance_id_lists[self.instance_id_set]) - 1
            pygame.event.clear()
            # count_mdr_surf, count_move_surf, fear_surf, world_surf = get_new_axs(
            #     list_of_instance_id_lists, ReadRecord, instance_id_set, instance_id_ii, screen)
        if event.key == pygame.K_LEFT:
            self.instance_id_ii = 0  # Reset
            self.instance_id_set -= 1
            if self.instance_id_set < 0:
                self.instance_id_set = 0
            pygame.event.clear()
            # count_mdr_surf, count_move_surf, fear_surf, world_surf = get_new_axs(
            #     list_of_instance_id_lists, ReadRecord, instance_id_set, instance_id_ii, screen)
        if event.key == pygame.K_RIGHT:
            self.instance_id_ii = 0  # Reset
            self.instance_id_set += 1
            if self.instance_id_set >= len(self.list_of_instance_id_lists):
                self.instance_id_set = len(self.list_of_instance_id_lists) - 1
            pygame.event.clear()
            # count_mdr_surf, count_move_surf, fear_surf, world_surf = get_new_axs(
            #     list_of_instance_id_lists, ReadRecord, instance_id_set, instance_id_ii, screen)
        if event.key == pygame.K_PAGEDOWN or event.key == pygame.K_RSHIFT:
            self.instance_id_ii = 0  # Reset
            self.instance_id_set = 0  # Reset
            self.record_name_id += 1
            if self.record_name_id >= len(self.list_of_record_names):
                self.record_name_id = len(self.list_of_record_names) - 1
            pygame.event.clear()
        if event.key == pygame.K_PAGEUP or event.key == pygame.K_LSHIFT:
            self.instance_id_ii = 0  # Reset
            self.instance_id_set = 0  # Reset
            self.record_name_id -= 1
            if self.record_name_id < 0:
                self.record_name_id = 0
            pygame.event.clear()
        if event.key == pygame.K_p:  # Add Record and Instance to plots
            add_record_instance_to_plot(new_record_name=self.list_of_record_names[self.record_name_id],
                                        new_instance=
                                        self.list_of_instance_id_lists[self.instance_id_set][self.instance_id_ii])
            pygame.event.clear()
        if event.key == pygame.K_g:  # Generate Plots
            generate_plots_for_record_instances()
            pygame.event.clear()

    def draw_surfaces(self):
        self.old_record_name_id = self.record_name_id
        self.old_instance_id_set = self.instance_id_set
        self.old_instance_id_ii = self.instance_id_ii
        # print('Run Data:', instance_id_set, instance_id_ii)
        self.clock.tick(FPS)
        self.screen.fill((255, 255, 255))  # Fill White
        self.w, self.h = pygame.display.get_surface().get_size()
        # w_offset = (w - scaled_surf.get_width()) // 2  # To center the Image Horizontally
        # h_offset = (h - scaled_surf.get_height()) // 2  # To center the Image Vertically
        # screen.blit(scaled_surf, (w_offset, h_offset))
        self.screen.fill((255, 255, 255))  # Fill White
        scaled_world_surf = rescale_photo(self.world_surf, self.w // 2, self.h // 2)
        scaled_fear_surf = rescale_photo(self.fear_surf, self.w // 2, self.h // 2)
        scaled_count_mdr_surf = rescale_photo(self.count_mdr_surf, self.w // 2, self.h // 2)
        scaled_count_move_surf = rescale_photo(self.count_move_surf, self.w // 2, self.h // 2)
        self.screen.blit(scaled_world_surf, (0, 0))
        self.screen.blit(scaled_fear_surf, (0, self.h // 2))
        self.screen.blit(scaled_count_mdr_surf, (self.w // 2, 0))
        self.screen.blit(scaled_count_move_surf, (self.w // 2, self.h // 2))

    def draw_text_and_update_changes(self, update_now=False):
        if self.instance_id_set_names is not None:
            instance_text = self.instance_id_set_names[self.instance_id_set] + ' : ' + str(self.instance_id_ii + 1)
            if (not self.instance_id_set == self.old_instance_id_set) or \
                    (not self.instance_id_ii == self.old_instance_id_ii):
                instance_text_rendered = self.font.render(instance_text, True, RED)
            else:
                instance_text_rendered = self.font.render(instance_text, True, BLACK)
            tw_instance, th_instance = instance_text_rendered.get_size()
            self.screen.blit(instance_text_rendered, ((self.w - tw_instance) // 2, 0))
        else:
            th_instance = 0

        # MdR_text = self.tiny_font.render(('Move de Rigueur : ' + self.MdR), True, BLACK)
        MdR_text = self.tiny_font.render(('Move de Rigueur : ' + self.mdr_string), True, BLACK)

        tw_MdR, th_MdR = MdR_text.get_size()
        self.screen.blit(MdR_text, ((self.w - tw_MdR) // 2, th_instance + TEXT_PADDING // 2))
        record_name_text = self.tiny_font.render(self.record_name, True, BLACK, (255, 255, 255))
        record_name_text = pygame.transform.rotate(record_name_text, 90)
        tw, th = record_name_text.get_size()
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(
            self.w - tw - TEXT_PADDING, self.h - th - TEXT_PADDING, tw, th))
        self.screen.blit(record_name_text, ((self.w - tw) - TEXT_PADDING, self.h - th - TEXT_PADDING))
        # screen.blit(record_name_text, ((w-tw)//2, h - th))
        if not self.record_name_id == self.old_record_name_id:
            self.ReadRecord, self.record_name, self.N_Cases, self.N_iterations = read_json_record(
                self.list_of_record_names, self.record_name_id)
            self.list_of_instance_id_lists, self.instance_id_set_names = \
                find_instances_with_extreme_ssq_of_FeAR(self.N_Cases, self.N_iterations, self.ReadRecord)

            # ---
            self.InstanceID = self.list_of_instance_id_lists[self.instance_id_set][self.instance_id_ii]

            self.MdR4Agents = self.ReadRecord[self.InstanceID]["MdR4Agents"]
            MdR_id = self.MdR4Agents[0][1]  # Assuming that all agents have the same MdR
            ActionList, _ = Agent.DefineActions()
            self.MdR = ActionList[MdR_id]

            self.mdr_string = get_mdr_string(self.MdR4Agents, return_names=True)
            print('MdR String : ', self.mdr_string)
            # ---

            text_img = self.font.render('Loading...', True, RED)
            tw, th = text_img.get_size()
            self.screen.blit(text_img, (0, self.h - th))
            pygame.display.update()
            self.clock.tick(FPS)

            self.count_mdr_surf, self.count_move_surf, self.fear_surf, self.world_surf = get_new_axs(
                self.list_of_instance_id_lists, self.ReadRecord, self.instance_id_set,
                self.instance_id_ii, self.screen)

        elif (not self.instance_id_set == self.old_instance_id_set) or (
                not self.instance_id_ii == self.old_instance_id_ii) or (update_now == True):
            text_img = self.font.render('Loading...', True, RED)
            tw, th = text_img.get_size()
            self.screen.blit(text_img, (0, self.h - th))
            pygame.display.update()
            self.clock.tick(FPS)

            self.count_mdr_surf, self.count_move_surf, self.fear_surf, self.world_surf = get_new_axs(
                self.list_of_instance_id_lists, self.ReadRecord, self.instance_id_set,
                self.instance_id_ii, self.screen)

    def draw_text_and_update_changes_iteration_sequence(self, update_now=False):
        text_img = self.tiny_font.render('Sequences of Instances per Case', True, BLUE)
        # tw, th = text_img.get_size()
        self.screen.blit(text_img, (0, 0))

        if self.instance_id_set_names is not None:
            instance_text = self.instance_id_set_names[self.instance_id_set] + ' : ' + str(self.instance_id_ii + 1)
            if (not self.instance_id_set == self.old_instance_id_set) or \
                    (not self.instance_id_ii == self.old_instance_id_ii):
                instance_text_rendered = self.font.render(instance_text, True, RED)
            else:
                instance_text_rendered = self.font.render(instance_text, True, BLACK)
            tw_instance, th_instance = instance_text_rendered.get_size()
            self.screen.blit(instance_text_rendered, ((self.w - tw_instance) // 2, 0))
        else:
            th_instance = 0
        # MdR_text = self.tiny_font.render(('Move de Rigueur : ' + self.MdR), True, BLACK)
        MdR_text = self.tiny_font.render(('Move de Rigueur : ' + self.mdr_string), True, BLACK)

        tw_MdR, th_MdR = MdR_text.get_size()
        self.screen.blit(MdR_text, ((self.w - tw_MdR) // 2, th_instance + TEXT_PADDING // 2))
        record_name_text = self.tiny_font.render(self.record_name, True, BLACK, (255, 255, 255))
        record_name_text = pygame.transform.rotate(record_name_text, 90)
        tw, th = record_name_text.get_size()
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(
            self.w - tw - TEXT_PADDING, self.h - th - TEXT_PADDING, tw, th))
        self.screen.blit(record_name_text, ((self.w - tw) - TEXT_PADDING, self.h - th - TEXT_PADDING))
        # screen.blit(record_name_text, ((w-tw)//2, h - th))
        if not self.record_name_id == self.old_record_name_id:
            self.ReadRecord, self.record_name, self.N_Cases, self.N_iterations = read_json_record(
                self.list_of_record_names, self.record_name_id)
            self.list_of_instance_id_lists, self.instance_id_set_names = \
                find_list_of_instances_per_case(N_Cases=self.N_Cases, N_iterations=self.N_iterations)
            # ---
            self.InstanceID = self.list_of_instance_id_lists[self.instance_id_set][self.instance_id_ii]

            self.MdR4Agents = self.ReadRecord[self.InstanceID]["MdR4Agents"]
            MdR_id = self.MdR4Agents[0][1]  # Assuming that all agents have the same MdR
            ActionList, _ = Agent.DefineActions()
            self.MdR = ActionList[MdR_id]

            self.mdr_string = get_mdr_string(self.MdR4Agents, return_names=True)
            print('MdR String : ', self.mdr_string)
            # ---
            text_img = self.font.render('Loading...', True, RED)
            tw, th = text_img.get_size()
            self.screen.blit(text_img, (0, self.h - th))
            pygame.display.update()
            self.clock.tick(FPS)

            self.count_mdr_surf, self.count_move_surf, self.fear_surf, self.world_surf = get_new_axs(
                self.list_of_instance_id_lists, self.ReadRecord, self.instance_id_set,
                self.instance_id_ii, self.screen)

        elif (not self.instance_id_set == self.old_instance_id_set) or (
                not self.instance_id_ii == self.old_instance_id_ii) or (update_now == True):
            text_img = self.font.render('Loading...', True, RED)
            tw, th = text_img.get_size()
            self.screen.blit(text_img, (0, self.h - th))
            pygame.display.update()
            self.clock.tick(FPS)

            self.count_mdr_surf, self.count_move_surf, self.fear_surf, self.world_surf = get_new_axs(
                self.list_of_instance_id_lists, self.ReadRecord, self.instance_id_set,
                self.instance_id_ii, self.screen)


def get_new_axs(list_of_instance_id_lists, ReadRecord, instance_id_set, instance_id_ii, screen):
    world_ax, fear_ax, count_mdr_ax, count_move_ax = \
        plot_results_for_instances(list_of_instance_id_lists[instance_id_set][instance_id_ii], ReadRecord)
    world_surf = get_pygame_surf_from_ax(world_ax)
    fear_surf = get_pygame_surf_from_ax(fear_ax)
    count_mdr_surf = get_pygame_surf_from_ax(count_mdr_ax)
    count_move_surf = get_pygame_surf_from_ax(count_move_ax)

    return count_mdr_surf, count_move_surf, fear_surf, world_surf


def get_pygame_surf_from_ax(ax):
    fig = ax.get_figure()
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    raw_data = canvas.buffer_rgba()
    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    return surf


def read_json_record(list_of_record_names, record_name_id):
    record_name = list_of_record_names[record_name_id]
    Record_Path = os.path.join(RECORD_FOLDER, record_name)
    print('Reading Record : ', Record_Path)
    with open(Record_Path) as json_file:
        ReadRecord = json.load(json_file)

    # Get Scenarios from the read record.
    Scenario = ReadRecord['Scenario']

    N_Agents = Scenario['N_Agents']
    N_Cases = Scenario['N_Cases']
    N_iterations = Scenario['N_iterations']

    # Just a check - Minimum one iteration
    if N_iterations <= 0:
        N_iterations = 1

    print('N_Agents : ', N_Agents)
    print('N_Cases : ', N_Cases)
    print('N_iterations : ', N_iterations)

    return ReadRecord, record_name, N_Cases, N_iterations


def get_mdr_string(MdR4Agents, return_names=False):
    if return_names:

        mdr_name_list = []
        action_names, _ = Agent.DefineActions()
        for agent, mdr in MdR4Agents:
            mdr_name = action_names[mdr]
            if mdr_name == 'Stay':
                mdr_name_list.append('S0')
            else:
                # First Letter and Last Number (of steps)
                mdr_name_list.append(mdr_name[0] + mdr_name[-1])
        mdr_string_names = '-'.join(mdr_name_list)
        return mdr_string_names

    else:

        mdr_list = []
        for agent, mdr in MdR4Agents:
            mdr_list.append(mdr)
        mdr_string_num = '-'.join(map("{:02d}".format, mdr_list))
        return mdr_string_num


def get_list_of_records(pathname):
    print('Loading Records from : ', pathname)
    record_paths = (glob.glob(pathname + '/**/*.json', recursive=True))

    print('List of Records :')

    record_names = []

    for ii in range(len(record_paths)):
        record_names.append(os.path.basename(record_paths[ii]))
        print(record_names[ii])

    return record_names


def add_record_instance_to_plot(json_filename=JSON_4_PLOTS, new_record_name=None, new_instance=None):
    # Appending a new instance to the JSON file
    print('Adding Record:', new_record_name, ' and Instance: ', new_instance)

    # Reading Dictionary from JSON file
    with open(json_filename) as json_file:
        # data = json.load(json_file)
        try:
            data = json.load(json_file)
        except JSONDecodeError:
            print('JSONDecodeError')
            data = {}

    # Adding new scenario
    if new_record_name is None:
        print('Error ! - No New Record')
        return False
    elif new_instance is None:
        print("Error - No  - Aborted")
        return False
    else:
        new_record_instance = {'RecordName': new_record_name, 'Instance': new_instance}
        if new_record_instance in data.values():
            print('Instance already added!')
        else:
            keys = list(data.keys())
            n = len(keys)
            data[str(n)] = new_record_instance

    pretty_print_json = pprint.pformat(data, width=150).replace("'", '"')

    with open(json_filename, 'w') as f:
        f.write(pretty_print_json)

    return True


def generate_plots_for_record_instances(json_filename=JSON_4_PLOTS):
    print('Generating Plots')

    # Reading Dictionary from JSON file
    with open(json_filename) as json_file:
        # data = json.load(json_file)
        try:
            data = json.load(json_file)
        except JSONDecodeError:
            print('JSON File Loading Error')
            return False

    for record_instance_id in data:
        record_instance = data[record_instance_id]
        print('record_instance', record_instance)
        record_name = record_instance['RecordName']
        instance_id = record_instance['Instance']
        list_of_record_names = get_list_of_records(RECORD_FOLDER)
        record_name_id = list_of_record_names.index(record_name)
        read_record, _, _, _ = read_json_record(list_of_record_names=list_of_record_names,
                                                record_name_id=record_name_id)
        MdR4Agents = read_record[instance_id]["MdR4Agents"]
        mdr_string = get_mdr_string(MdR4Agents, return_names=True)
        plot_results_for_instances(instance_id, read_record, mdr_string=mdr_string, overwrite_images=OVERWRITE_IMAGES,
                                   saveFolder=SAVE_IMAGES_TO, record_name=record_name[17:-5])
        print('Plots Generated!')

    print('All plots generated!')

    return True


def rescale_photo(photo, new_width, new_height):
    old_height = photo.get_height()
    old_width = photo.get_width()

    if new_height / new_width <= old_height / old_width:
        scale_by = new_height / old_height
    else:
        scale_by = new_width / old_width

    scaled_height = scale_by * old_height
    scaled_width = scale_by * old_width

    scaled_photo = pygame.transform.smoothscale(photo, (scaled_width, scaled_height))

    return scaled_photo


if __name__ == "__main__":
    main()
