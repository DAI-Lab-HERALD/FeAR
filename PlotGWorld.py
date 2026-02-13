# Functions for plotting GWorld
import numpy as np

np.random.seed(0)
import seaborn as sns
import matplotlib.colors as mcolors

sns.set_theme(style="ticks")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import cmcrameri.cm as cmc

import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import glob
from PIL import Image
import os
from tqdm import tqdm
import GWorld
import Agent
import Emergence
from itertools import permutations



VerboseFlag = False
# VerboseFlag = True

ColourPalette = "bone_r"
# CMAP_VALIDMOVES = sns.cubehelix_palette(as_cmap=True, light=1, dark=0.3, gamma=0.5)
hue_validMoves = 190
CMAP_VALIDMOVES = sns.diverging_palette(360 - hue_validMoves, hue_validMoves, l=60, as_cmap=True)
CMAP_FeAR = sns.diverging_palette(220, 20, as_cmap=True)
cmap_FeAR_normalize = mcolors.Normalize(vmin=-1, vmax=1)
CMAP_FeAR_SCALAR_MAPPABLE = plt.cm.ScalarMappable(cmap=CMAP_FeAR, norm=cmap_FeAR_normalize)

# MOVE_ARROW_COLOUR = 'tab:blue'
# MOVE_ARROW_COLOUR = 'tab:grey'
MOVE_ARROW_COLOUR = 'dimgrey'
MOVE_ARROW_COLOUR_MDR = 'deepskyblue'
MOVE_ARROW_WIDTH = 0.05
AGENT_BOX_OFFSET = 0.15

# PRINT_MATRIX_SIZE = [9, 3]
PRINT_MATRIX_SIZE = [4, 4] # OG
PRINT_MATRIX_SIZE_FINER = [6, 6]  # Finer

# PRINT_GWORLD_SIZE = [8, 4]
PRINT_GWORLD_SIZE = [12, 6]
GWORLD_FONT_SIZE = 8

PRINT_VALIDMOVES_SIZE = [3, 2]
PRINT_DPI = 800

FIG_SIZE = [9, 5]
FIG_DPI = 200

DECIMALS_FMT = '.1f'

# plt.rcParams['figure.figsize'] = [8, 5];
plt.rcParams['figure.figsize'] = FIG_SIZE
# plt.rcParams['figure.dpi'] = 600;
plt.rcParams['figure.dpi'] = FIG_DPI

plt.rcParams['savefig.bbox'] = 'tight'


# plt.rcParams['savefig.pad_inches'] = 0.1 // Default = 0.1


class PlotGWorld:

    def __init__(self):
        #         #--- For Visualisation----
        self.Gfig, self.Gax = plt.subplots(figsize=(3, 3), dpi=FIG_DPI);
        # #         line, = self.Gax.plot([])     # A tuple unpacking to unpack the only plot
        # plt.clf();
        # plt.axis('equal');
        pass

    # ----------------------------------------------------------------------------------------------- #

    def ViewGWorld(self, World, ViewNextStep=False, ViewActionTrail=True, ViewActionArrows=True, Animate=False, ax=None,
                   saveFolder=None, imageName='GW_Snap', extension='png',
                   mark_agent_entropy=True,
                   highlight_actor=None, highlight_affected=None, highlight_affected_action=False,
                   grayscale=False,
                   annot_font_size=GWORLD_FONT_SIZE, overwrite_image=False, colour_by_fear=False, fear_values=None,
                   mdr_colour=False, game_mode=False, ego_id=0, apples=[],
                   annot_rox=None):

        if ViewActionTrail:
            WorldState = World.WorldState
            # This has the trail for the last Action
        else:
            WorldState = np.where(World.WorldState > 0, 0, World.WorldState)
            for idx, agent in enumerate(World.AgentList):
                # Updating WorldState with latest AgentLocations
                WorldState[(World.AgentLocations[idx])] = idx + 1

        if mark_agent_entropy:
            map_of_agent_entropy = Emergence.GetMapofAgentEntropies(World, WorldState)
            map_values = map_of_agent_entropy
        else:
            map_values = WorldState

        if ax is None:
            ax = self.Gax

        Annotations = WorldState
        Annotations = np.where(Annotations == 0, 0, Annotations)
        # mask = np.where(Annotations == 0, 1, 0)
        mask = np.where(WorldState >= 0, 0, np.nan)
        mask_c = np.where(mask == 0, 1, 0)

        plt.clf()
        len_x, len_y = WorldState.shape
        axis_length_max = max(len_x, len_y)

        # Plot Agent IDs and Locations of the Map

        # Plotting Grey blocks for valid regions of the Map
        cmap_for_grey = sns.color_palette("Greys", as_cmap=True)
        ax = sns.heatmap(np.zeros_like(WorldState), linewidths=max(25 // axis_length_max, 3), square=True, mask=mask,
                         linecolor='whitesmoke',
                         cbar=False, cmap=cmap_for_grey, vmax=10, vmin=-1.5)
        ax = sns.heatmap(np.zeros_like(WorldState), linewidths=max(25 // axis_length_max, 3) - 2, square=True,
                         mask=mask_c,
                         cbar=False, cmap=cmap_for_grey, vmax=10, vmin=0)
        # Plotting Agent Locations with Annotations
        # ax = sns.heatmap(map_values, linewidths=1, annot=Annotations, mask=mask,
        #                  square=True,
        #                  cbar=False, cmap=ColourPalette, annot_kws={"size": annot_font_size})
        # plt.title('State of GWorld: ')
        # plt.axis('equal')
        plt.axis('off')
        xlim_heatmap = ax.get_xlim()
        ylim_heatmap = ax.get_ylim()

        #  Plotting boxes around agents
        if colour_by_fear and fear_values is not None:
            if len(fear_values) == len(World.AgentList):
                agent_colours = CMAP_FeAR_SCALAR_MAPPABLE.to_rgba(fear_values)
            else:
                print('Number of FeAR values passed in does not match the number of agents.')
        else:
            agent_colours = special_spectral_cmap(n_colours=len(World.AgentList), game_mode=game_mode, ego_id=ego_id,
                                                  highlight_actor=highlight_actor, highlight_affected=highlight_affected,
                                                  grayscale=grayscale)

        for xx in range(len_x):
            for yy in range(len_y):
                if WorldState[xx][yy] > 0:
                    agentIdxxyy = WorldState[xx][yy].astype(int) - 1
                    if agentIdxxyy == ego_id and game_mode:
                        agent_annotation = '$\U0001F60C$'
                        # agent_annotation = '$\U0001F604$'
                    else:
                        agent_annotation = str(agentIdxxyy + 1)

                    # ax = plot_rect_on_matrix(yy, xx, ax=ax, offset=-AGENT_BOX_OFFSET, color=MOVE_ARROW_COLOUR,
                    #                          linewidth=3)

                    if highlight_actor is not None or highlight_affected is not None:
                        if agentIdxxyy in highlight_affected:
                            edgecolor=MOVE_ARROW_COLOUR
                        elif agentIdxxyy in highlight_actor:
                            edgecolor=MOVE_ARROW_COLOUR
                        else:
                            edgecolor= 'darkgrey'
                    else:
                        edgecolor = MOVE_ARROW_COLOUR

                    ax = plot_rect_on_matrix(yy, xx, ax=ax, offset=-AGENT_BOX_OFFSET,
                                             linewidth=max(20 // axis_length_max, 1),
                                             color=agent_colours[agentIdxxyy],
                                             edgecolor=edgecolor,
                                             fill=True, zorder=5)
                    ax.text(yy + 0.5, xx + 0.5, agent_annotation, zorder=10, size=annot_font_size,
                            horizontalalignment='center', verticalalignment='center_baseline')
        for apple in apples:
            ax.text(apple[1] + 0.5, apple[0] + 0.5, '$\U0001F604$', zorder=5, size=annot_font_size,
                    horizontalalignment='center', verticalalignment='center_baseline', color='red')


        MaxSteps = World.MaxSteps
        # MaX_ArrowOffsets = np.ceil(MaxSteps/5)*5 # So as to get a multiple of 5
        MaX_ArrowOffsets = MaxSteps + 1

        # Offset for plotting arrows
        ArrowOffset = 0.5
        Margin_OneWayArrow = 0.2
        IndividualArrowOffset_Span = 1 - 2 * (AGENT_BOX_OFFSET + MOVE_ARROW_WIDTH)
        IndividualArrowOffset_Margin = (AGENT_BOX_OFFSET + MOVE_ARROW_WIDTH)
        IndividualArrowOffset_Delta = IndividualArrowOffset_Span / MaX_ArrowOffsets

        # Plot Actions Selected by the agents
        if ViewActionArrows:
            for idx, agent in enumerate(World.AgentList):
                dx = 0
                dy = 0
                for stepx, stepy in agent.SelectedAction:
                    dx += stepx
                    dy += stepy

                ArrowOffset_x = ArrowOffset
                ArrowOffset_y = ArrowOffset

                if ViewNextStep:  # Start Arrow from Current Location
                    x0 = World.AgentLocations[idx][0]
                    y0 = World.AgentLocations[idx][1]
                else:  # Start Arrow from Previous Location
                    x0 = World.PreviousAgentLocations[idx][0]
                    y0 = World.PreviousAgentLocations[idx][1]

                if abs(dx) > 0:
                    #  ArrowOffset_y = (x % MaxOffset) times the Delta
                    ArrowOffset_y = (((x0 % MaX_ArrowOffsets)) * IndividualArrowOffset_Delta) \
                                    + IndividualArrowOffset_Margin
                    if dx > 0:
                        ArrowOffset_x = ArrowOffset_x + (0.5 - AGENT_BOX_OFFSET)  # Start from agent box
                        dx = dx - 2 * (0.5 - AGENT_BOX_OFFSET)  # Reduce arrow length to stay between agent boxes
                    else:
                        ArrowOffset_x = ArrowOffset_x - (0.5 - AGENT_BOX_OFFSET)  # Start from agent box
                        dx = dx + 2 * (0.5 - AGENT_BOX_OFFSET)  # Reduce arrow length to stay between agent boxes
                    # print(ArrowOffset_y)
                elif abs(dy) > 0:
                    #  ArrowOffset_x = (y % MaxOffset) times the Delta
                    ArrowOffset_x = (((y0 % MaX_ArrowOffsets)) * IndividualArrowOffset_Delta) \
                                    + IndividualArrowOffset_Margin
                    if dy > 0:
                        ArrowOffset_y = ArrowOffset_y + (0.5 - AGENT_BOX_OFFSET)  # Start from agent box
                        dy = dy - 2 * (0.5 - AGENT_BOX_OFFSET)  # Reduce arrow length to stay between agent boxes
                    else:
                        ArrowOffset_y = ArrowOffset_y - (0.5 - AGENT_BOX_OFFSET)  # Start from agent box
                        dy = dy + 2 * (0.5 - AGENT_BOX_OFFSET)  # Reduce arrow length to stay between agent boxes

                # Adding the ArrowOffsets to x0 and y0
                x = x0 + ArrowOffset_x
                y = y0 + ArrowOffset_y

                if not mdr_colour:
                    if highlight_actor is None and highlight_affected is None:
                        arrow_colour = MOVE_ARROW_COLOUR
                    else: # Use lighter arrows for non-highlighted agents
                        if idx in highlight_actor:
                            arrow_colour = MOVE_ARROW_COLOUR
                        elif idx in highlight_affected:
                            if highlight_affected_action:
                                arrow_colour=MOVE_ARROW_COLOUR
                            else:
                                arrow_colour='darkgrey'
                        else:
                            arrow_colour = 'darkgray'

                    plt.arrow(y, x, dy, dx, ls='-', color=arrow_colour, zorder=4,
                              width=MOVE_ARROW_WIDTH, head_width=MOVE_ARROW_WIDTH * 3,
                              length_includes_head=True)
                else:
                    plt.arrow(y, x, dy, dx, ls='-', color=MOVE_ARROW_COLOUR_MDR, zorder=4,
                              width=MOVE_ARROW_WIDTH, head_width=MOVE_ARROW_WIDTH * 3,
                              length_includes_head=True)

                    if dx == 0 and dy == 0 and idx == ego_id:  # Draw MdR square in case of stay
                        ax = plot_rect_on_matrix(y0, x0, ax=ax, offset=-AGENT_BOX_OFFSET,
                                                 linewidth=max(20 // axis_length_max, 2),
                                                 color=MOVE_ARROW_COLOUR_MDR, fill=False, zorder=5)

        # Plot OneWays
        for path in World.WorldOneWays:
            dx = path[1][0] - path[0][0]
            dy = path[1][1] - path[0][1]

            if dy == 0:  # Vertical Arrow
                dx -= 2 * Margin_OneWayArrow * np.sign(dx)  # Subtract Margin
                x = path[0][0] + ArrowOffset + Margin_OneWayArrow * np.sign(dx)  # Add Margin
                y = path[0][1] + ArrowOffset
            elif dx == 0:  # Horizontal Arrow
                dy -= 2 * Margin_OneWayArrow * np.sign(dy)  # Subtract Margin
                x = path[0][0] + ArrowOffset
                y = path[0][1] + ArrowOffset + Margin_OneWayArrow * np.sign(dy)  # Add Margin

            plt.arrow(y, x, dy, dx, ls='-', color='gold', width=.08,
                      lw=0.5, length_includes_head=True)

        # Plot Walls
        for wall in World.WorldWalls:
            x1 = wall[0][0]
            y1 = wall[0][1]
            x2 = wall[1][0]
            y2 = wall[1][1]

            if x1 == x2:  # Vertical Wall
                x = x1
                y = max(y1, y2)
                dx = 1
                dy = 0
            elif y1 == y2:  # Horizontal Wall
                x = max(x1, x2)
                y = y1
                dx = 0
                dy = 1

            plt.plot([y, y + dy], [x, x + dx], ls='-', color='tab:red', linewidth=2)


        if annot_rox:

            if 'rects' in annot_rox.keys():
                rects = annot_rox['rects']
                for rect_color in rects.keys():
                    xx_s= rects[rect_color]['x_s']
                    yy_s= rects[rect_color]['y_s']

                    for xx,yy in zip(xx_s, yy_s):
                        plot_rect_on_matrix(xx, yy, ax=ax, color=rect_color,
                                            zorder=6, offset=-0.02, linewidth=2)

            if 'circs' in annot_rox.keys():
                circs = annot_rox['circs']
                for circ_color in circs.keys():
                    xx_s = circs[circ_color]['x_s']
                    yy_s = circs[circ_color]['y_s']

                    for xx, yy in zip(xx_s, yy_s):
                        plot_circ_on_matrix(xx, yy, ax=ax, color='white',linewidth=7, zorder=6, offset=-0.2)
                        plot_circ_on_matrix(xx, yy, ax=ax, color=circ_color, zorder=6, offset=-0.2)

            if 'crosses' in annot_rox.keys():
                crosses = annot_rox['crosses']
                for cross_color in crosses.keys():
                    xx_s = crosses[cross_color]['x_s']
                    yy_s = crosses[cross_color]['y_s']

                    for xx, yy in zip(xx_s, yy_s):
                        plot_xcross_on_matrix(xx, yy, ax=ax, color=cross_color, zorder=6,
                                              offset=-0.3, linewidth=7)


        #         # Plot Restricted Paths
        #         for path in self.RestrictedPaths:
        #             x = path[0][0] + ArrowOffset
        #             y = path[0][1] + ArrowOffset
        #             dx = path[1][0] - path[0][0]
        #             dy = path[1][1] - path[0][1]
        #             plt.arrow(y,x,dy,dx,ls='-',color='tab:red',width=.05,length_includes_head=True)

        ax.set_ylim(ylim_heatmap)
        ax.set_xlim(xlim_heatmap)

        fig = plt.gcf()

        if saveFolder is not None:
            fig.set_size_inches(PRINT_GWORLD_SIZE[0], PRINT_GWORLD_SIZE[1])
            fig.set_dpi(PRINT_DPI)
            plt.title('')

            save_plot(imageName, overwrite_image, saveFolder, extension=extension)
        else:
            fig.set_size_inches(FIG_SIZE[0], FIG_SIZE[1])

        if not Animate:
            plt.show()

        return ax

    # ----------------------------------------------------------------------------------------------- #


def plotMatrix(Matrix, xlabel=None, ylabel=None, cmap=None, mask=None, linecolor='white',
               xticklabels='auto', yticklabels='auto', ax=None, annot=True,
               vmin=-1, vmax=1, center=0, title=None, cbar=None, annot_colour=None,
               annot_font_size=5, fmt=DECIMALS_FMT, for_print=False):
    if cmap is None:
        cmap = sns.diverging_palette(220, 20, as_cmap=True, sep=1)

    if ax is None:
        fig, ax = plt.subplots()

    if for_print:
        square = False

        if cbar is None:
            cbar = False
    else:
        square = True

    if annot_colour:
        annot_kws = {"size": annot_font_size,
                     'color': annot_colour}
    else:
        annot_kws = {"size": annot_font_size}

    # To get manage the rounding error caused by eps in the denominator and
    # string formatting to one decimal place
    Matrix = np.around(Matrix, decimals=2)

    sns.heatmap(Matrix.T, linewidths=1, cbar=cbar, cmap=cmap, square=square, center=center, vmin=vmin, vmax=vmax,
                xticklabels=xticklabels, yticklabels=yticklabels, ax=ax, annot=annot, fmt=fmt, linecolor=linecolor,
                mask=mask, annot_kws=annot_kws)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def plotResponsibility(Resp, FeAL=None, ax=None, cbar=None, annot_font_size=8, title='Responsibility',
                       plot_feal_separately=False, saveFolder=None, imageName='FeAR_', fmt=DECIMALS_FMT,
                       skip_title=False, skip_xlabel=False, add_hatches=True, gray_feal=True,
                       exclude_diag=True, extension='pdf',
                       overwrite_image=False, for_print=False, finer=False):
    maskDiag, ticklabels = get_mask_n_ticks(Resp, ExcludeDiag=exclude_diag)
    xlabel = 'Actor'
    ylabel = 'Affected'

    if skip_title:
        title = ''
        imageName = imageName + '_noTitle'
    if skip_xlabel:
        xlabel = ''
        imageName = imageName + '_noXlabel'

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))

    if (plot_feal_separately is True) and (FeAL is not None):
        if ax is not None:
            print("The axs passed in is not considered since new subplots have to be made.")

        fig, fear_axs = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [len(FeAL) + 1, 1]})

        fear_axs[0] = plotMatrix(Resp, xlabel=xlabel, ylabel=ylabel, title='FeAR', xticklabels=ticklabels,
                                 yticklabels=ticklabels, for_print=for_print, fmt=fmt,
                                 mask=maskDiag, ax=fear_axs[0], annot_font_size=annot_font_size, cbar=cbar)
        fear_axs[1] = plotMatrix(np.array([FeAL]), xlabel=None, ylabel=ylabel, title='FeAL',
                                 yticklabels=ticklabels, for_print=for_print, fmt=fmt,
                                 ax=fear_axs[1], annot_font_size=annot_font_size, cbar=False)

        fig.suptitle(title)
        ax = fear_axs[0]

    else:
        if ax is not None:
            ax = plotMatrix(Resp, xlabel=xlabel, ylabel=ylabel, title=title, xticklabels=ticklabels,
                            yticklabels=ticklabels, for_print=for_print, fmt=fmt,
                            mask=maskDiag, ax=ax, annot_font_size=annot_font_size, cbar=cbar);

        if FeAL is not None:
            N_Agents = len(FeAL)
            FeAL_matrix = np.zeros((N_Agents, N_Agents))
            for xx in range(N_Agents):
                FeAL_matrix[xx][xx] = FeAL[xx]

            mask_feal = np.where(maskDiag == 0, 1, 0)
            ax = plotMatrix(FeAL_matrix, mask=mask_feal, xlabel=xlabel, ylabel=ylabel, title=title, vmin=0, fmt=fmt,
                            xticklabels=ticklabels, yticklabels=ticklabels, ax=ax,
                            annot_font_size=annot_font_size, annot_colour='black',
                            cbar=False, for_print=for_print);

            for xx in range(N_Agents):
                # ax = plot_rect_on_matrix(xx, xx, ax=ax, offset=-0.08, color=sm.to_rgba(0), linewidth=0,
                #                          fill=True)
                if gray_feal:
                    ax = plot_rect_on_matrix(xx, xx, ax=ax, offset=0, color='lightgray', linewidth=0,
                                             fill=True)
                ax = plot_rect_on_matrix(xx, xx, ax=ax, offset=-0.9, color=sm.to_rgba(0), linewidth=0,
                                         fill=True)

    if add_hatches:

        # Overlay hatching for negative and positive values
        tt = 0.000
        # c_del = 0.95
        c_del = 0.3
        for jj in range(Resp.shape[0]):
            for ii in range(Resp.shape[1]):
                if ii == jj:
                    continue

                if Resp[jj, ii] < 0:
                    ax.add_patch(plt.Rectangle((jj + tt / 2, ii + tt / 2), 1 - tt, 1 - tt, fill=False, hatch='....',
                                               edgecolor=sm.to_rgba(-c_del), lw=0))
                    ax.add_patch(
                        plt.Rectangle((jj + tt / 2, ii + tt / 2), 1 - tt, 1 - tt, fill=False, edgecolor='white', lw=1))

                elif Resp[jj, ii] > 0:
                    ax.add_patch(plt.Rectangle((jj + tt / 2, ii + tt / 2), 1 - tt, 1 - tt, fill=False, hatch='//',
                                               edgecolor=sm.to_rgba(1 - c_del), lw=0))
                    ax.add_patch(
                        plt.Rectangle((jj + tt / 2, ii + tt / 2), 1 - tt, 1 - tt, fill=False, edgecolor='white', lw=1))

    if for_print:
        # fig = plt.gcf()
        fig = ax.figure
        if finer:
            fig.set_size_inches(PRINT_MATRIX_SIZE_FINER[0], PRINT_MATRIX_SIZE_FINER[1])
        else:
            fig.set_size_inches(PRINT_MATRIX_SIZE[0], PRINT_MATRIX_SIZE[1])
        fig.set_dpi(PRINT_DPI)

        save_plot(imageName, overwrite_image, saveFolder, ax=ax, extension=extension)

    return ax


def save_plot(imageName, overwrite_image, saveFolder, extension='png', ax=None):
    if ax:
        fig = ax.figure
    else:
        fig = plt.gcf()

    if saveFolder is not None:
        filepath = os.path.join(saveFolder, imageName)
        if not overwrite_image:
            i = 0
            while os.path.exists(('{}{:03d}.' + extension).format(filepath, i)):
                i += 1
            fig.savefig(('{}{:03d}.' + extension).format(filepath, i))
        else:
            fig.savefig(filepath + '.' + extension)

    # To remove the size and dpi settings for the next plot
    # fig, ax = plt.subplots()


def plotEntropy(Entropy, ax=None):
    xticklabels = np.arange(Entropy.shape[0]) + 1
    title = 'Entropy'
    xlabel = 'Agent'
    ylabel = None
    cbar = False
    # cmap = 'Blues'
    cmap = 'bone_r'
    vmin = None
    vmax = None
    center = None
    annot_font_size = 10

    ax = plotMatrix(np.array([Entropy]).T, xlabel=xlabel, ylabel=ylabel, title=title, xticklabels=xticklabels,
                    ax=ax, cbar=cbar, cmap=cmap, vmin=vmin, vmax=vmax, center=center, annot_font_size=annot_font_size);

    ax.get_yaxis().set_visible(False)

    return ax


def get_mask_n_ticks(Matrix, ExcludeDiag=True):
    if ExcludeDiag is True:
        maskDiag = np.diag(np.diag(np.ones_like(Matrix)))
    else:
        maskDiag = None

    ticklabels = np.arange(Matrix.shape[0]) + 1

    return maskDiag, ticklabels


def plotCounts(Counts, ax=None, title=None, vmin=None, vmax=None, cmap=CMAP_VALIDMOVES,
               annot_font_size=8, center=0, saveFolder=None, imageName='PlotCounts_', overwrite_image=False,
               cbar=True, counts_feal=None, for_print=False, finer=False):
    maskDiag, ticklabels = get_mask_n_ticks(Counts)
    xlabel = 'Actor'
    ylabel = 'Affected'

    fmt = 'd'

    ax = plotMatrix(Counts, xlabel=xlabel, ylabel=ylabel, title=title, xticklabels=ticklabels, yticklabels=ticklabels,
                    mask=maskDiag, ax=ax, vmin=vmin, vmax=vmax, center=center, cmap=cmap, cbar=cbar,
                    annot_font_size=annot_font_size, for_print=for_print, fmt=fmt);

    if counts_feal is not None:
        N_Agents = len(counts_feal)
        counts_feal_matrix = np.zeros((N_Agents, N_Agents)).astype(int)
        for xx in range(N_Agents):
            counts_feal_matrix[xx][xx] = counts_feal[xx]

        mask_feal = np.where(maskDiag == 0, 1, 0)
        ax = plotMatrix(counts_feal_matrix, mask=mask_feal, xlabel=xlabel, ylabel=ylabel, title=title,
                        xticklabels=ticklabels, yticklabels=ticklabels, ax=ax, vmin=vmin, vmax=vmax,
                        annot_font_size=annot_font_size, center=center, cmap=CMAP_VALIDMOVES, for_print=for_print,
                        cbar=False, fmt=fmt);

        for xx in range(N_Agents):
            ax = plot_rect_on_matrix(xx, xx, ax=ax, offset=-0.08, color='white', linewidth=2)

    if for_print:
        fig = plt.gcf()
        if finer:
            fig.set_size_inches(PRINT_MATRIX_SIZE_FINER[0], PRINT_MATRIX_SIZE_FINER[1])
        else:
            fig.set_size_inches(PRINT_MATRIX_SIZE[0], PRINT_MATRIX_SIZE[1])
        fig.set_dpi(PRINT_DPI)

        save_plot(imageName, overwrite_image, saveFolder)

    return ax


def plot_valid_actions(validity_of_actions=None, ax=None, title=None, only_horizontal=False, for_print=False,
                       saveFolder=None, imageName='ValidActions_', overwrite_image=False):
    if validity_of_actions is None:
        print('validity_of_actions not passed!')
        return False

    # cmap = 'bone_r'
    cmap = CMAP_VALIDMOVES
    vmin = 0
    vmax = 1
    # center = 0

    action_list = validity_of_actions

    # print('action_list :',action_list, len(action_list))

    # ActionNames, ActionMoves = Agent.DefineActions()
    # for ii in range(len(action_list)):
    #     print('Action:',ActionNames[ii],' , Validity: ', action_list[ii])

    # Putting the Up, Down , Left and Right Actions into an Matrix
    n_list = action_list[1:17]
    # print('n_list :',n_list, len(n_list))
    n_list = n_list.reshape(4, 4)
    n_list = n_list.transpose()

    matrix_for_plot = np.zeros((9, 9)).astype(int)

    # Moving the actions in n_list into matrix_for_plot
    matrix_for_plot[4, 4] = action_list[0]  # stay
    matrix_for_plot[0:4, 4] = np.flip(n_list[0, :])  # Up
    matrix_for_plot[5:10, 4] = (n_list[1, :])  # Down
    matrix_for_plot[4, 0:4] = np.flip(n_list[2, :])  # Left
    matrix_for_plot[4, 5:10] = n_list[3, :]  # Right

    # Somehow Left-Right was mixed up with Up-Down - So to solve that the transpose was taken
    matrix_for_plot = matrix_for_plot.transpose()
    matrix_for_plot = np.flip(matrix_for_plot, axis=1)

    # Masking out the irrelevant cells
    mask = np.zeros_like(matrix_for_plot)
    mask[4, :] = 1
    mask[:, 4] = 1
    mask = np.where(mask == 0, 1, 0)

    # print(matrix_for_plot)

    if only_horizontal == False:

        ax = plotMatrix(matrix_for_plot, annot=False, mask=mask, xticklabels=False, yticklabels=False,
                        fmt='d', cbar=False, ax=ax, title=title, cmap=cmap, vmin=vmin, vmax=vmax)

        ax = plot_rect_on_matrix(4, 4, offset=0.05, ax=ax, color='gainsboro', linewidth=3)
        ax = plot_rect_on_matrix(4, 4, offset=0.05, ax=ax, color='gray', linewidth=0.5)

        # ax.plot([3.95,3.95,5.05,5.05,3.95],[3.95,5.05,5.05,3.95,3.95], color='gainsboro', linewidth=3)
        # ax.plot([3.95, 3.95, 5.05, 5.05, 3.95], [3.95, 5.05, 5.05, 3.95, 3.95], color='gray', linewidth=0.5)
        # rect = patches.Rectangle((4, 4), 1, 1)
        # ax.add_patch(rect)

    else:  # Plot only horizontal moves - Left, Stay, Right

        ax = plotMatrix((matrix_for_plot[:, 4:5]), annot=False,
                        xticklabels=False, yticklabels=False, cmap=cmap, vmin=vmin, vmax=vmax,
                        fmt='d', cbar=False, ax=ax, title=title)

        ax = plot_rect_on_matrix(4, 0, offset=0.05, ax=ax, color='gainsboro', linewidth=3)
        ax = plot_rect_on_matrix(4, 0, offset=0.05, ax=ax, color='gray', linewidth=0.5)

        ax.set_ylim(0 - 0.1, 1 + 0.1)

    if for_print:
        fig = plt.gcf()
        fig.set_size_inches(PRINT_VALIDMOVES_SIZE[0], PRINT_VALIDMOVES_SIZE[1])
        fig.set_dpi(PRINT_DPI)

        save_plot(imageName, overwrite_image, saveFolder)

    return ax


def plot_rect_on_matrix(x, y, ax=None, offset=0.05, color='gainsboro', linewidth=3, zorder=1, fill=False,
                        x_len=1, y_len=1, edgecolor=MOVE_ARROW_COLOUR):
    if ax is None:
        print('No Axis Passed!')
        return False

    if fill == False:
        ax.plot([x - offset, x - offset, x + x_len + offset, x + x_len + offset, x - offset],
                [y - offset, y + y_len + offset, y + y_len + offset, y - offset, y - offset], color=color, linewidth=linewidth,
                zorder=zorder)
    else:  # Using patches for fill
        rect = patches.Rectangle((x - offset, y - offset), (1 + 2 * offset), (1 + 2 * offset), zorder=zorder,
                                 linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor=color)
        ax.add_patch(rect)

    return ax


def plot_circ_on_matrix(x, y, r=0.5, ax=None, offset=0.05,
                        color='gainsboro', linewidth=5, zorder=1,
                        fill=False,
                        ):
    if ax is None:
        print('No Axis Passed!')
        return False
    circle = plt.Circle((x+0.5, y+0.5), r+offset, color=color, linewidth=linewidth,fill=fill, zorder=zorder)
    ax.add_patch(circle)
    return ax


def plot_xcross_on_matrix(x, y, ax=None, offset=0.05,
                        color='gainsboro', linewidth=7, zorder=1,
                        x_len=1, y_len=1,
                        ):
    if ax is None:
        print('No Axis Passed!')
        return False
    ax.plot([x - offset,  x + x_len + offset],
            [y - offset,  y + y_len + offset], color=color,
            linewidth=linewidth,
            zorder=zorder)
    ax.plot([x + x_len + offset, x - offset],
            [y - offset, y + y_len + offset], color=color,
            linewidth=linewidth,
            zorder=zorder)

    return ax
# -------------------------------------------------------------------------------------------------------------------#


def make_gif_from_folder(folder, GifName='_NewGif', duration=500):
    frames = [Image.open(image) for image in tqdm(glob.glob(f"{folder}/*.png"), colour="blue", ncols=100)]
    frame_one = frames[0]
    frame_one.save(GifName + ".gif", format="GIF", append_images=frames,
                   save_all=True, duration=duration, loop=0)
    return None


def special_spectral_cmap(n_colours=5, game_mode=False, ego_id=0,
                          highlight_affected=None, highlight_actor=None,
                          grayscale=False):
    if highlight_actor or highlight_affected:
        colours = ['lightgray' for i in range(n_colours)]
        for i in highlight_actor:
            colours[i] = (0.9155324875048059, 0.6192233756247598, 0.65440215301807)
        for j in highlight_affected:
            colours[j] = (0.5820686065676477, 0.7683811347467602, 0.8872124391839864)

        return colours

    if grayscale and not game_mode:
        return ['lightgray' for i in range(n_colours)]

    if game_mode:
        ego_colour = (0.5820686065676477, 0.7683811347467602, 0.8872124391839864)
        others_colour = (0.996078431372549, 0.8784313725490196, 0.5450980392156862)
        colours = []
        for ii in range(n_colours):
            if ii == ego_id:
                colours.append(ego_colour)
            else:
                colours.append(others_colour)
        return colours

    # Function to get the set of spectral colours I like
    if n_colours >= 5:
        return sns.color_palette("Spectral", n_colors=n_colours)
    elif n_colours == 4:
        colours = sns.color_palette("Spectral", n_colors=9)
        return [(0.9155324875048059, 0.6192233756247598, 0.65440215301807),
                colours[3], colours[5],
                (0.5820686065676477, 0.7683811347467602, 0.8872124391839864)]
        # return list( colours[i] for i in [0, 2, 5, 8])
    elif n_colours == 3:
        colours = sns.color_palette("Spectral", n_colors=9)
        return [(0.9155324875048059, 0.6192233756247598, 0.65440215301807),
                colours[3],
                (0.5820686065676477, 0.7683811347467602, 0.8872124391839864)]
        # return list( colours[i] for i in [0, 3, 8])
    elif n_colours == 2:
        colours = sns.color_palette("Spectral", n_colors=9)
        return [colours[3],
                (0.5820686065676477, 0.7683811347467602, 0.8872124391839864)]
        # return list(colours[i] for i in [0, 8])
    else:
        colours = sns.color_palette("Spectral", n_colors=9)
        return [(0.9155324875048059, 0.6192233756247598, 0.65440215301807)]
        # return colours[0]


def plotGroup(i, n, resp, overwrite_image=False, for_print=False,
              saveFolder=None, scenario_name=' ', fmt='0.2f', cbar=False,
              annot_font_size = 12, actors_on_x=True,
              add_hatches=True):

    cmap = sns.diverging_palette(220, 20, as_cmap=True, sep=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))

    if actors_on_x:
        resp = resp.T
        y_values = np.arange(1, n + 1)
        x_values = [perm for perm in permutations(y_values, i) if sorted(perm) == list(perm)]
    else:
        x_values = np.arange(1, n + 1)
        y_values = [perm for perm in permutations(x_values, i) if sorted(perm) == list(perm)]


    _, ax = plt.subplots(figsize=(max(len(x_values),2), max(len(y_values)/2,1.7)))

    sns.heatmap(resp.T, ax=ax, cmap=cmap, center=0, vmin=-1, vmax=1, linecolor='white', linewidths=1, cbar=cbar,
                annot=True, annot_kws={"size": annot_font_size},
                fmt=fmt)

    if actors_on_x:
        y_labels = y_values
        x_labels = [
            '{'+ ','.join(
                [ str(x_values[ix][ia]) for ia in range(len(x_values[ix])) ]
            ) + '}' for ix in range(len(x_values))
                   ]
    else:
        x_labels = x_values
        y_labels = [
            '{' + ','.join(
                [str(y_values[iy][ia]) for ia in range(len(y_values[iy]))]
            ) + '}' for iy in range(len(y_values))
        ]
    ax.set_xticks(ticks=np.arange(len(x_values)) + 0.5, labels=x_labels)
    ax.set_yticks(ticks=np.arange(len(y_values)) + 0.5, labels=y_labels, rotation=0)

    if actors_on_x:
        ax.set_xlabel("Actors")
        ax.set_ylabel("Affected")
        ax.invert_yaxis()
    else:
        ax.set_ylabel("Actors")
        ax.set_xlabel("Affected")
    ax.set_title(f'{i}-Agent Group FeAR')

    if add_hatches:

        # Overlay hatching for negative and positive values
        tt = 0.000
        # c_del = 0.95
        c_del = 0.3
        for ii in range(resp.shape[1]):
            for jj in range(resp.shape[0]):

                if resp[jj, ii] < 0:
                    ax.add_patch(plt.Rectangle((jj + tt / 2, ii + tt / 2), 1 - tt, 1 - tt, fill=False, hatch='....',
                                               edgecolor=sm.to_rgba(-c_del), lw=0))
                    ax.add_patch(
                        plt.Rectangle((jj + tt / 2, ii + tt / 2), 1 - tt, 1 - tt, fill=False, edgecolor='white', lw=1))

                elif resp[jj,ii] > 0:
                    ax.add_patch(plt.Rectangle((jj + tt / 2, ii + tt / 2), 1 - tt, 1 - tt, fill=False, hatch='//',
                                               edgecolor=sm.to_rgba(1 - c_del), lw=0))
                    ax.add_patch(
                        plt.Rectangle((jj + tt / 2, ii + tt / 2), 1 - tt, 1 - tt, fill=False, edgecolor='white', lw=1))

    if for_print:
        save_plot(imageName=f'{scenario_name}_{i}-gFeAR', overwrite_image=overwrite_image, saveFolder=saveFolder)


    plt.show()


def plot_scenario(scenario_name=None, scenario_source_file='Scenarios4FeARSims.json',
                  save_folder='GW_Snaps',  extension='pdf', annot_font_size=24, random_seed=0,):
    assert scenario_name is not None, 'scenario_name must be provided'
    rng = np.random.default_rng(seed=random_seed)
    plotgw = PlotGWorld()

    Scenario = GWorld.LoadJsonScenario(json_filename=scenario_source_file, scenario_name=scenario_name)

    # GWorld Preview

    Region = np.array(Scenario['Map']['Region'])
    Walls = Scenario['Map']['Walls']
    OneWays = Scenario['Map']['OneWays']

    World = GWorld.GWorld(Region, Walls=Walls, OneWays=OneWays)  # Initialising GWorld from Scenario

    N_Agents = Scenario['N_Agents']

    AgentLocations = []
    for location in Scenario['AgentLocations']:
        AgentLocations.append(tuple(location))

    # Adding N Agents at sortedandom positions
    if len(AgentLocations) < N_Agents:
        [locX, locY] = np.where(Region == 1)

        LocIdxs = rng.choice(locX.shape[0], size=(N_Agents - len(AgentLocations)), replace=False, shuffle=False)
        LocIdxs.sort()

        for Idx in LocIdxs:
            AgentLocations.append((locX[Idx], locY[Idx]))

    # Adding Agents
    PreviousAgentAdded = True
    for location in AgentLocations:
        # Adding new Agents if Previous Agent was Added to the World
        if PreviousAgentAdded:
            Ag_i = Agent.Agent()
        PreviousAgentAdded = World.AddAgent(Ag_i, location, printStatus=False)

    PreviousAgentAdded = True
    while len(World.AgentList) < N_Agents:
        # Adding new Agents if Previous Agent was Added to the World
        if PreviousAgentAdded:
            Ag_i = Agent.Agent()
        Loc_i = (np.random.randint(Region.shape[0]), np.random.randint(Region.shape[1]))
        PreviousAgentAdded = World.AddAgent(Ag_i, Loc_i, printStatus=False)

    # -------------------------------------------------------------------------------
    # Selecting actions for agents
    # -------------------------------------------------------------------------------

    defaultAction = Scenario['defaultAction']
    SpecificAction4Agents = Scenario['SpecificAction4Agents']

    # Setting Policy for all Agents

    # The default Step and Direction Weights
    StepWeights = Scenario['StepWeights']
    DirectionWeights = Scenario['DirectionWeights']

    ListOfStepWeights = []
    ListOfDirectionWeights = []

    for ii in range(len(World.AgentList)):
        ListOfStepWeights.append(StepWeights)
        ListOfDirectionWeights.append(DirectionWeights)

    # Updating the list of stepweights based on specific weights for agents
    for agentIDs, stepweights4agents in Scenario['SpecificStepWeights4Agents']:
        for agentID in agentIDs:
            ListOfStepWeights[agentID] = stepweights4agents

    # Updating the list of directionweights based on specific weights for agents
    for agentIDs, directionweights4agents in Scenario['SpecificDirectionWeights4Agents']:
        for agentID in agentIDs:
            ListOfDirectionWeights[agentID] = directionweights4agents

        # Updating Agent Policies in World
        for ii, agent in enumerate(World.AgentList):
            policy = Agent.GeneratePolicy(StepWeights=ListOfStepWeights[ii],
                                          DirectionWeights=ListOfDirectionWeights[ii])
            agent.UpdateActionPolicy(policy)

    if 'Policies' in (Scenario.keys()):
        # Dictionary of Policies
        if Scenario['Policies']:
            policies = Scenario['Policies']
            policy_map = np.zeros(np.shape(Region), dtype=int)

            policy_keys = Scenario['Policies'].keys()
            # print(f'{policy_keys =}')
            for key in policy_keys:
                slicex = Scenario['Policies'][key]['slicex']
                slicey = Scenario['Policies'][key]['slicey']
                policy_map[slicex, slicey] = key

            # print(f'Region =\n {Region}')
            print(f'policyMap =\n {policy_map}')

            # Setting Policy for all Agents

            # Updating Agent Policies in World
            for ii, agent in enumerate(World.AgentList):
                agent_location = World.AgentLocations[ii]
                agent_policy = str(policy_map[agent_location[0], agent_location[1]]).zfill(2)
                if 'policyWeights' in policies[agent_policy].keys():
                    policy_weights = policies[agent_policy]['policyWeights']
                    agent_stepWeights = None
                    agent_directionWeights = None
                else:
                    policy_weights = None
                    agent_stepWeights = policies[agent_policy]['stepWeights']
                    agent_directionWeights = policies[agent_policy]['directionWeights']

                policy = Agent.GeneratePolicy(StepWeights=agent_stepWeights, DirectionWeights=agent_directionWeights,
                                              policy_weights=policy_weights)
                agent.UpdateActionPolicy(policy)

    Action4Agents = World.SelectActionsForAll(defaultAction=defaultAction, InputActionID4Agents=SpecificAction4Agents)

    # Plotting the State of the World and Chosen Actions for the next iteration
    ax = plotgw.ViewGWorld(World, ViewActionArrows=True, ViewNextStep=True, ViewActionTrail=False,
                           saveFolder=save_folder,
                           imageName=f'{scenario_name}_scenario', extension=extension,
                           annot_font_size=annot_font_size,
                           overwrite_image=True)



def make_axes_gray(ax):

    fig = ax.get_figure()

    for ax in fig.axes:
        plt.setp(ax.spines.values(), color='lightgray')
        ax.tick_params(labelcolor='dimgray', colors='lightgray')

        ax.xaxis.label.set_color('dimgray')
        ax.yaxis.label.set_color('dimgray')

