import numpy as np

# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx

SIZE = (10, 12)

BLUE = '#407e9c'
RED = '#c3553a'

GRAY = '#999999'
LIGHTGRAY = '#cccccc'
DARKGRAY = '#333333'
YELLOW = '#ffcc33'
GREEN = '#339933'
BLACK = '#000000'
WHITE = '#ffffff'

CMAP = cm.get_cmap('Spectral')


def main():
    pass


if __name__ == "__main__":
    main()


def make_axes_gray(ax):
    plt.setp(ax.spines.values(), color='lightgray')
    ax.tick_params(labelcolor='dimgray', colors='lightgray')
    ax.xaxis.label.set_color('dimgray')
    ax.yaxis.label.set_color('dimgray')

    ax.grid(True, linestyle=':')


def plot_directed_fear_graph(fear, x0, y0, fear_threshold=0.05, ax=None,
                             hide_cbar=False, horizontal_cbar=False,
                             zorder_lift=0, game_mode=False):
    """Plot a directed graph with edges colored and sized based on the fear values.

    Parameters
    ----------
    game_mode: bool
        Whether to plot for the Game of FeAR
    zorder_lift: int
        Layers to lift the plot by
    hide_cbar: bool
        Whether to hide to colorbar
    horizontal_cbar : bool
        If True, the colorbar will have horizontal orientation. Else, it will be vertical.
    fear : ndarray
        Array containing Feasible Action-Space Reduction values
    x0 : ndarray
        x coordinate of the starting locations of agents
    y0 : ndarray
        y coordinate of the starting locations of agents
    fear_threshold : float, optional
        If the absolute value of fear is below the fear_threshold, then these are not plotted
    ax : Matplotlib Axes object, optional

    Returns
    -------
    ax : Matplotlib Axes object
        Plot of the directed fear graph.
    """

    if game_mode:
        hide_cbar = True
        zorder_lift = 10

    x0 = x0.squeeze()
    y0 = y0.squeeze()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    fear_min = -1
    fear_max = 1

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    n_agents = len(x0)
    for i in range(n_agents):
        G.add_node(i, pos=(x0[i], y0[i]))

    # Add edges to the graph based on fear values
    for actor in range(n_agents):
        for affected in range(n_agents):
            if actor != affected:
                fear_value = fear[actor, affected]
                if abs(fear_value) > fear_threshold:
                    # Only add edge if fear values with abs() greater than a threshold
                    G.add_edge(actor, affected, weight=fear_value)

    # Get positions of nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Create colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True, sep=1)

    # Draw node labels as i + 1
    labels = {i: str(i + 1) for i in G.nodes()}

    # Draw the graph
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='moccasin', labels=labels,
            edge_color=weights, width=2, ax=ax, alpha=0.8,
            edge_cmap=cmap, edge_vmin=fear_min, edge_vmax=fear_max,
            connectionstyle='arc3, rad = 0.1')

    # nodes_drawn = nx.draw_networkx_nodes(G, pos, node_size=300, node_color='moccasin', ax=ax, alpha=0.75)

    #     nx.draw_networkx_labels(G, pos, labels=labels, alpha=0.75, ax=ax)
    edges_drawn = nx.draw_networkx_edges(G, pos, edge_color=weights, width=2, ax=ax,
                                         edge_cmap=cmap, edge_vmin=fear_min, edge_vmax=fear_max,
                                         connectionstyle='arc3, rad = 0.1')

    #     nx.draw_networkx_labels(G, pos, labels=labels)

    if zorder_lift > 0:
        nodes_drawn = nx.draw_networkx_nodes(G, pos, node_size=300, node_color='moccasin',
                                             ax=ax, alpha=0.75)
        labels_drawn = nx.draw_networkx_labels(G, pos, labels=labels)

        nodes_drawn.set_zorder(zorder_lift + 2)

        for _, label_drawn in labels_drawn.items():
            label_drawn.set_zorder(zorder_lift + 4)

        for edge in edges_drawn:
            edge.set_zorder(zorder_lift)

    # Add colorbar for edge colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=fear_min, vmax=fear_max))
    sm.set_array([])
    if not hide_cbar:
        divider = make_axes_locatable(ax)
        if horizontal_cbar:
            cax = divider.append_axes('bottom', size='5%', pad=1)
            cbar = fig.colorbar(sm, orientation='horizontal', cax=cax)
            cbar.ax.axvline(x=fear_threshold, ymin=0.1, ymax=0.9, c='grey')
            cbar.ax.axvline(x=-fear_threshold, ymin=0.1, ymax=0.9, c='grey')
            cbar.ax.axvspan(xmin=-fear_threshold, xmax=fear_threshold, ymin=0, ymax=1, color='w')
        else:  # Vertical Colour Bar
            cax = divider.append_axes('right', size='5%', pad=1)
            cbar = fig.colorbar(sm, orientation='vertical', cax=cax)
            cbar.ax.axhline(y=fear_threshold, xmin=0.1, xmax=0.9, c='grey')
            cbar.ax.axhline(y=-fear_threshold, xmin=0.1, xmax=0.9, c='grey')
            cbar.ax.axhspan(ymin=-fear_threshold, ymax=fear_threshold, xmin=0, xmax=1, color='w')
        cbar.outline.set_visible(False)  # Remove the black outline
        make_axes_gray(cbar.ax)
        cbar.set_label('Fear Value')
    ax.set_aspect(1)

    if not game_mode:
        ax.set_title('Directed Graph with Fear Values')

    return ax
