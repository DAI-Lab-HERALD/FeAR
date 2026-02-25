import itertools
import pandas as pd
import numpy as np
import os

import Responsibility


class GroupSortingAlgorithm:

    def __init__(self, world_in, action_id_4agents, mdr4agents):
        self.func_world, self.mdr, self.actions = Responsibility.prepare_world(world_in, action_id_4agents, mdr4agents)
        self.action_id_4agents = action_id_4agents

    def calc_responsibility(self, group, affected_index):
        fear, _, _ = Responsibility.calculateSpecificGroupResponsibility(self.func_world, group,
                                                                         affected_index,
                                                                         self.mdr, self.actions, self.action_id_4agents)
        return fear

    def find_tier_structure(self, sorted_fears, sorted_indices, affected_index):

        # If there are no assertive agents
        if len(sorted_fears) == 0 or sorted_fears[0] < 0:
            return [], 0

        # If there is exactly one assertive agent
        if len(sorted_fears) == 1:
            if sorted_fears[0] == 0: # return null set if there is only one agent which is not assertive
                return [], 0
            return [[sorted_indices]], [sorted_fears[0]]

        min_r_group = []
        tier_groups = []
        tier_fear = []

        remaining_agents = sorted_indices[sorted_fears >= 0]
        remaining_agents = np.array(remaining_agents, dtype=int)

        r_group_fear = 0
        all_agent_fear = self.calc_responsibility(remaining_agents, affected_index)

        while len(remaining_agents) > 0:
            current_tier = []
            for n in range(1, len(remaining_agents)+1):
                if n > len(remaining_agents):
                    break
                combinations = list(itertools.combinations(remaining_agents, n))
                while len(combinations) > 0:
                    combination = np.array(combinations.pop(0), dtype=int)
                    test_group = np.array(np.concatenate([min_r_group, combination]), dtype=int)
                    test_group_fear = self.calc_responsibility(test_group, affected_index)

                    if test_group_fear > r_group_fear:
                        # Test overlapping combinations
                        overlapping_combinations = [sublist for sublist in combinations if
                                                    any(elem in combination for elem in sublist)]

                        '''
                        Test if any of the ovelapping comnbinations also meet the criteria.
                        This is necessary to ensure that the algorithm is not affected by the indexing of agents.
                        If these are not identified now, they will be excluded when we prune combinations later.
                        
                        MAYBE THIS IS NOT NECESSARY (FOR THE GRID WORLD)! BUT KEEPING IT IN JUST TO BE SAFE.!
                        
                        TODO: Prove that:
                         if an assertive combo is identified, then no other overlapping combos can be assertive
                        '''
                        consolidated_combination = combination
                        for combo in overlapping_combinations:
                            combo_group = np.array(np.concatenate([min_r_group, combo]), dtype=int)
                            combo_group_fear = self.calc_responsibility(combo_group, affected_index)

                            '''If any of the ovelapping comnbinations also met the criteria,
                             add it the combination to be appended to current_tier'''
                            if combo_group_fear>r_group_fear:
                                consolidated_combination = np.union1d(consolidated_combination,combo)

                        # Adding combination to tier.
                        current_tier.append(sorted(consolidated_combination))

                        # Pruning ovelapping combinations
                        combinations = [sublist for sublist in combinations if
                                        not any(elem in consolidated_combination for elem in sublist)]

                        # Update remaining_agents: remove the elements in the combination
                        remaining_agents = np.setdiff1d(remaining_agents, consolidated_combination)

            if len(current_tier) == 0:
                break
            min_r_group = np.concatenate([min_r_group, pd.Series(current_tier).explode().to_numpy()])
            tier_groups.append(current_tier)
            r_group_fear = self.calc_responsibility(min_r_group, affected_index)
            tier_fear.append(r_group_fear)

        return tier_groups, tier_fear

    def FindCourteousGroup(sorted_fears, sorted_indices, affected_index, world_in, action_id_4agents, mdr4agents=[]):
        if len(sorted_fears) == 0 or sorted_fears[-1] >= 0:
            return []
        func_world, mdr, actions = Responsibility.prepare_world(world_in, action_id_4agents, mdr4agents)

        list_of_actions_for_agents = []
        for agentID in func_world.AgentList:
            list_of_actions_for_agents.append(len(agentID.Actions))

        group = [(sorted_indices[-1])]
        current_fear = sorted_fears[-1]
        for i in range(len(sorted_indices), 1):
            if sorted_fears[i] >= 0:
                break
            if sorted_indices[i] == affected_index:
                continue
            test_group = group.copy()
            test_group.append(sorted_indices[i])
            resp, num_v_mdr, num_v_a, = Responsibility.calculateSpecificGroupResponsibility(func_world, test_group,
                                                                                            affected_index,
                                                                                            mdr, actions,
                                                                                            action_id_4agents)
            if resp > current_fear:
                group.append(sorted_indices[i])
                current_fear = resp
        return group


def generate_tier_tikz(tier_group, affected=None, scenario_name='', save_tex=False, results_folder='Tiers_tikz'):
    """
    Generate a TikZ diagram from tier_groups structure.

    Args:
        tier_group: List of tiers, where each tier contains groups of agents
                     Structure: tier_groups[tier][group][agent_index] = agent_name

    Returns:
        str: TikZ diagram code for embedding in LaTeX document
    """
    lines = [
        "\\footnotesize",
        "\\begin{tikzpicture}[",
        "    affected/.style={circle, draw=Dandelion!70, fill=white, ultra thick, minimum size=0.6cm, font=\\scriptsize},",
        "    agent/.style={circle, draw=red!70, fill=white, very thick, minimum size=0.4cm, font=\\scriptsize},",
        "    group/.style={rectangle, rounded corners=10pt, draw=black!40, fill=black!5, very thick, inner sep=3pt},",
        "    tier/.style={rectangle, draw=black!30, fill=black!10, inner sep=5pt},",
        "    arrow/.style={->, >=stealth, ultra thick, draw=black!40, line cap=round},",
        "    link/.style={-, >=stealth, ultra thick, draw=black!40, line cap=round},",
        "]",
        ""
    ]

    # Track node positions and names
    node_map = {}  # Maps (tier, group, agent) -> node_name
    group_nodes = {}  # Maps (tier, group) -> list of node_names
    tier_nodes = {}  # Maps tier -> list of all node_names in tier

    # Calculate total height needed based on max agents per tier
    max_agents_in_tier = max(sum(len(group) for group in tier) for tier in tier_group) if tier_group else 0
    agent_spacing = 0.8 # vertical spacing between agents

    # Generate nodes for each tier (right to left, so reverse)
    x_position = 0
    tier_width = 1.1  # horizontal spacing between tiers

    # Collect all node positions first
    all_nodes = {}  # Store node info for later drawing

    for t, tier in enumerate(tier_group):

        agents_in_tier = 0
        for group in tier:
            if isinstance(group, str): # Treat strings as a single agent
                agents_in_tier += 1
            else:
                for _ in group:
                    agents_in_tier += 1

        y_offset = (agents_in_tier - 1) / 2 * -agent_spacing

        for g, group in enumerate(tier):

            if isinstance(group, str):
                agents = [group]
            else:
                agents = group

            for a, agent in enumerate(agents):
                node_name = f"T{t}G{g}A{a}"
                y_pos = -y_offset

                if isinstance(agent, str):
                    all_nodes[(t, g, a)] = {
                        'name': node_name,
                        'label': agent,
                        'x': x_position,
                        'y': y_pos
                    }
                else:
                    all_nodes[(t, g, a)] = {
                        'name': node_name,
                        'label': agent + 1,
                        'x': x_position,
                        'y': y_pos
                    }

                if (t, g) not in group_nodes:
                    group_nodes[(t, g)] = []
                group_nodes[(t, g)].append(node_name)

                if t not in tier_nodes:
                    tier_nodes[t] = []
                tier_nodes[t].append(node_name)

                y_offset += agent_spacing

        x_position += tier_width

    # Layer 1: Draw agent nodes first (on main layer, top)
    lines.append("    % Layer 1: Agent nodes (top)")
    if affected is not None:
        if isinstance(affected, int):
            lines.append(f"    \\node[affected] (AFFECTED) at ({-tier_width},0) {{{affected + 1}}};")
        else:
            lines.append(f"    \\node[affected] (AFFECTED) at ({-tier_width},0) {{{affected}}};")

    for (t, g, a), node_info in all_nodes.items():
        lines.append(
            f"    \\node[agent] ({node_info['name']}) at ({node_info['x']},{node_info['y']}) {{{node_info['label']}}};")
        node_map[(t, g, a)] = node_info['name']

    lines.append("")

    lines.append("    \\begin{pgfonlayer}{background}")

    # Layer 3: Draw tier boundaries on background (bottom layer, drawn after groups)
    lines.append("        % Tier boundaries (bottom)")
    for t in range(len(tier_group)):
        if tier_nodes[t]:
            node_list_str = "(" + ")(".join(tier_nodes[t]) + ")"
            lines.append(
                f"        \\node[tier, fit={node_list_str}, label={{[anchor=south]above:$\\tier{{{affected + 1}}}{{{t + 1}}}$ }}] (tier{t}) {{}};")

    lines.append("")
    # Layer 2: Draw group boundaries on background (middle layer)
    lines.append("    % Layer 2: Group boundaries (middle)")
    for t, tier in enumerate(tier_group):
        for g, group in enumerate(tier):
            if len(group) > 1 and not isinstance(group, str):
                group_node_list = group_nodes[(t, g)]
                node_list_str = "(" + ")(".join(group_node_list) + ")"
                group_name = f"group{t}_{g}"
                lines.append(f"        \\node[group, fit={node_list_str}] ({group_name}) {{}};")

    lines.append("    \\end{pgfonlayer}")
    lines.append("")

    # Layer 4: Draw arrows (top layer, on main)
    lines.append("    % Layer 4: Arrows (top)")
    for t in range(len(tier_group) - 1, 0, -1):
        for g in range(len(tier_group[t])):
            if len(tier_group[t][g]) == 1:
                src = group_nodes[(t, g)][0]  # For singleton groups
            else:
                src = f"group{t}_{g}"  # For non-singleton groups
            lines.append(f"    \\draw[link] ({src}) -- (tier{t - 1});")


    if affected is not None:  # Add arrows from tier 1 to affected agent.
        for g in range(len(tier_group[0])):
            if len(tier_group[0][g]) == 1 or isinstance(tier_group[0][g], str):
                src = group_nodes[(0, g)][0]  # For singleton groups
            else:
                src = f"group{0}_{g}"  # For non-singleton groups
            lines.append(f"    \\draw[arrow] ({src}) -- (AFFECTED);")

    lines.append("\\end{tikzpicture}")
    lines.append("\\normalsize")

    if save_tex:
        tex_file = f'{scenario_name}_{affected}_tiers.tex'
        tex_path = os.path.join(results_folder,tex_file)
        # Save to a .tex file
        with open(tex_path, 'w') as f:
            f.write("\n".join(lines))
        print(f'Saved tier_tex to {tex_path}')

    return "\n".join(lines)


def generate_tier_mermaid(tier_group):
    """
    Generate a Mermaid flowchart from tier_groups structure.

    Args:
        tier_group: List of tiers, where each tier contains groups of agents
                     Structure: tier_groups[tier][group][agent_index] = agent_name

    Returns:
        str: Mermaid flowchart diagram as text
    """
    lines = ["%%{init: {'theme': 'neutral' } }%%", "flowchart RL"]

    # Track node IDs for connections
    node_map = {}  # Maps (tier, group) -> node_id or list of node_ids

    # Generate subgraphs for each tier
    for t, tier in enumerate(tier_group):
        lines.append(f"    subgraph Tier{t}[Tier {t}]")
        lines.append(f"        direction LR")

        for g, group in enumerate(tier):
            if len(group) == 1:
                # Single agent - just show as circle
                agent = group[0]
                node_id = f"T{t}G{g}A0"
                if isinstance(agent, int):
                    lines.append(f"        {node_id}({agent + 1})")
                else:
                    lines.append(f"        {node_id}({agent})")
                node_map[(t, g)] = node_id
            else:
                # Multiple agents - group in ellipse
                group_nodes = []
                lines.append(f"        subgraph T{t}G{g}[\" \"]")
                lines.append(f"            direction LR")

                for a, agent in enumerate(group):
                    node_id = f"T{t}G{g}A{a}"
                    if isinstance(agent, int):
                        lines.append(f"            {node_id}({agent + 1})")
                    else:
                        lines.append(f"            {node_id}({agent})")

                    group_nodes.append(node_id)

                lines.append(f"        end")
                node_map[(t, g)] = group_nodes

        lines.append(f"    end")

    # Generate connections from tier n+1 to tier n
    for t in range(len(tier_group) - 1, 0, -1):
        # For each group in tier t, connect to each group in tier t-1
        for g in range(len(tier_group[t])):
            source_nodes = node_map.get((t, g))
            if isinstance(source_nodes, str):
                source_nodes = [source_nodes]

            target_nodes = f'Tier{t - 1}'
            if isinstance(target_nodes, str):
                target_nodes = [target_nodes]

                # Connect each agent in current tier to the target group
                for src in source_nodes:
                    # Connect to first agent of target group (representing the group)
                    lines.append(f"    {src} --> {target_nodes[0]}")

    # Style for group subgraphs (ellipse-like appearance) and tier heights
    lines.append("")
    lines.append("    %% Styling")
    for t, tier in enumerate(tier_group):
        # Set uniform height for all tier subgraphs
        lines.append(f"    style Tier{t} stroke:#aaaaaa, fill:#dddddd,stroke-width:0px")

        for g, group in enumerate(tier):
            if len(group) > 1:
                lines.append(f"    style T{t}G{g} stroke:#555555,fill:#efefef,stroke-width:4px,rx:10,ry:10")

            for a, agent in enumerate(group):
                lines.append(f"    style T{t}G{g}A{a} stroke:#E53348,stroke-width:3px,fill:#ffffff")

    return "\n".join(lines)

def get_agent_ranks_from_tiers(tier_group=None, n_agents=None, affected=None):
    assert tier_group is not None
    assert n_agents is not None

    # agent_ranks = np.zeros(n_agents, dtype=int)
    agent_ranks = np.ones(n_agents, dtype=int) * (n_agents + 1)  # Non-influencing agents assigned rank (n_agents+1)
    if affected is not None:
        agent_ranks[affected] = 0  # Rank of ego agent set to 0. (To be treated as np.nan when comparing ranks later)

    current_tier_rank = 1
    for tier in tier_group:
        agents_of_rank = 0
        for group in tier:
            for agent in group:
                agent_ranks[agent] = current_tier_rank
                agents_of_rank += 1
        current_tier_rank += agents_of_rank

    return agent_ranks


def get_agent_ranks_from_fear(fear=None, affected=None):
    # Ranking actors with positive FeAR

    assert fear is not None
    assert affected is not None

    fear_affected = fear[affected]
    n_agents = len(fear_affected)

    # agent_ranks = np.zeros(n_agents, dtype=int)
    agent_ranks = np.ones(n_agents, dtype=int) * (n_agents+1)  # Non-influencing agents assigned rank (n_agents+1)
    agent_ranks[affected] = 0 # Rank of ego agent set to 0. (To be treated as np.nan when comparing ranks later)

    # Finding assertive agents
    assertive_mask = fear_affected > 0
    assertive_agents = np.where(assertive_mask)[0]

    if len(assertive_agents) == 0:
        return agent_ranks

    positive_fear = fear_affected[assertive_agents]
    sorted_order = np.argsort(-positive_fear) # Sorting from most to least

    sorted_positive_fear = positive_fear[sorted_order]
    sorted_assertive_agents = assertive_agents[sorted_order]

    current_rank = 1
    current_fear = sorted_positive_fear[0]
    agents_of_rank = 1

    for i in range(len(positive_fear)):
        agent_ranks[sorted_assertive_agents[i]] = current_rank

        if i < len(positive_fear)-1:
            if sorted_positive_fear[i+1] == current_fear:
                agents_of_rank += 1
            else:
                current_fear = sorted_positive_fear[i+1]
                current_rank += agents_of_rank
                agents_of_rank = 1

    return agent_ranks


def print_group_tiers_for_affected(tier_group=None,tier_fear=None, affected=None):
    assert affected is not None
    print(f'Group FeAR rankings affecting agent {affected + 1}:')
    if len(tier_group) == 0:
        print('   - ')
        return

    for t in range(len(tier_group)):
        tier = tier_group[t]
        fear = tier_fear[t]
        tier_str = '{ ' + (
            ' , '.join(('[' + (', '.join(str(agent + 1) for agent in group)) + ']') for group in tier)) + ' }'
        print(f'   Tier {t + 1}: {fear=:.4f} : {tier_str} ')

def find_and_compare_ranks(
        n_agents=None,
        affected=None,
        tier_group=None,
        individual_fear=None,
        shapley_values=None,
        scenario_name=None,
        return_rank_summary=False,
):
    assert n_agents is not None
    assert affected is not None
    assert tier_group is not None
    assert individual_fear is not None
    assert shapley_values is not None

    tier_ranks = get_agent_ranks_from_tiers(tier_group=tier_group, n_agents=n_agents, affected=affected)
    fear_ranks = get_agent_ranks_from_fear(fear=individual_fear, affected=affected)
    shap_ranks = get_agent_ranks_from_fear(fear=shapley_values, affected=affected)

    print(f'{individual_fear[affected]=}')
    print(f'{shapley_values[affected]=}')

    save_ranks_table(affected, tier_ranks, shap_ranks, fear_ranks, scenario_name)

    print('-' * 80)
    print('Comparing ranks!')
    print('-' * 80)

    tau_tier_shap, p_tier_shap = compare_ranks_for_affected(affected=affected,rank_1=tier_ranks, rank_2=shap_ranks)
    tau_fear_shap, p_fear_shap = compare_ranks_for_affected(affected=affected,rank_1=fear_ranks, rank_2=shap_ranks)
    tau_fear_tier, p_fear_tier = compare_ranks_for_affected(affected=affected,rank_1=fear_ranks, rank_2=tier_ranks)

    print(f'Tier-Shap: tau={tau_tier_shap:.3f}, p={p_tier_shap:.3f}')
    print(f'FeAR-Shap: tau={tau_fear_shap:.3f}, p={p_fear_shap:.3f}')
    print(f'FeAR-Tier: tau={tau_fear_tier:.3f}, p={p_fear_tier:.3f}')

    print('-' * 80)
    print('-' * 80)

    if return_rank_summary:
        rank_summary = {'affected': affected,
                        'tier_ranks': tier_ranks,
                        'fear_ranks': fear_ranks,
                        'shap_ranks': shap_ranks,
                        'tau_tier_shap': tau_tier_shap,
                        'tau_fear_shap': tau_fear_shap,
                        'tau_fear_tier': tau_fear_tier,
                        'p_fear_tier': p_fear_tier,
                        'p_tier_shap': p_tier_shap,
                        'p_fear_shap': p_fear_shap,
                        }
        return rank_summary



def compare_ranks_for_affected(
        affected=None,
        rank_1=None,
        rank_2=None,
):
    from scipy.stats import kendalltau

    assert affected is not None
    assert rank_1 is not None
    assert rank_2 is not None

    # Exclude ranks for affected agents
    mask = np.ones(len(rank_1), dtype=bool)
    mask[affected] = False

    tau, p_value = kendalltau(rank_1[mask], rank_2[mask])

    return tau, p_value


def save_ranks_table(affected, tier_ranks, shap_ranks, fear_ranks, scenario_name, output_dir='Tables'):
    """
    Generate and save a LaTeX table showing rankings for an affected agent.

    Parameters:
    -----------
    affected : int
        The affected agent number
    tier_ranks : array-like
        1D array of tier rankings
    shap_ranks : array-like
        1D array of SHAP rankings
    fear_ranks : array-like
        1D array of FeAR rankings
    output_dir : str, optional
        Directory to save the .tex file (default: current directory)
    """

    n_agents = len(tier_ranks)

    # Finding the difference in number of Assertive agents
    n_assertive_ifear = np.sum((fear_ranks > 0) & (fear_ranks < n_agents+1))
    n_assertive_gfear = np.sum((tier_ranks > 0) & (tier_ranks < n_agents+1))
    diff_n_assertive = n_assertive_gfear - n_assertive_ifear

    # Convert to numpy arrays if needed
    tier_ranks = np.asarray(tier_ranks).astype(object)  # Use object type to allow mixed types
    shap_ranks = np.asarray(shap_ranks).astype(object)
    fear_ranks = np.asarray(fear_ranks).astype(object)

    print('-' * 80)
    print(f'{n_assertive_ifear=}')
    print(f'{n_assertive_gfear=}')
    print(f'{diff_n_assertive=}')
    print('-' * 80)

    print('-' * 80)
    print(f'fear_ranks= {" ".join(map(str, fear_ranks))}')
    print(f'tier_ranks= {" ".join(map(str, tier_ranks))}')
    print(f'shap_ranks= {" ".join(map(str, shap_ranks))}')
    print('-' * 80)

    # Replace (n_agents+1) s with '.')
    tier_ranks = np.where(tier_ranks == (n_agents+1), '\color{lightgray}{\\phdot}', tier_ranks)
    shap_ranks = np.where(shap_ranks == (n_agents+1), '\color{lightgray}{\\phdot}', shap_ranks)
    fear_ranks = np.where(fear_ranks == (n_agents+1), '\color{lightgray}{\\phdot}', fear_ranks)

    # Replace the affected agent's rank with '-'
    tier_ranks[affected] = '\color{lightgray}{\\blacksquare}'
    shap_ranks[affected] = '\color{lightgray}{\\blacksquare}'
    fear_ranks[affected] = '\color{lightgray}{\\blacksquare}'

    # Format arrays as LaTeX matrix entries
    tier_str = ' & '.join(map(str, tier_ranks))
    shap_str = ' & '.join(map(str, shap_ranks))
    fear_str = ' & '.join(map(str, fear_ranks))

    actors = np.arange(1, len(tier_ranks)+1)
    actors_str = ' & '.join(map(str, actors))

    # Create LaTeX content
    latex_content = f"""\\centering
             \\begin{{tabular}}{{r l r}}
             %\\toprule
             %\\ & \\textbf{{Actors}} & : $\\begin{{bmatrix}} {actors_str} ~\\end{{bmatrix}}$\\\\
             %\\midrule
             \\multicolumn{{2}}{{l}}{{\\textbf{{Affected: {affected+1} }} }} & $\\deltaAssertive={diff_n_assertive}$ \\,\\\\
             \\midrule
             iFeAR &           ranks & : $\\begin{{bmatrix}} {fear_str} \\end{{bmatrix}}$\\\\
             gFeAR & (Tier)    ranks & : $\\begin{{bmatrix}} {tier_str} \\end{{bmatrix}}$\\\\
             gFeAR & (Shapley) ranks & : $\\begin{{bmatrix}} {shap_str} \\end{{bmatrix}}$\\\\
\\bottomrule
\\end{{tabular}}"""

    # Save to file
    filename = os.path.join(output_dir,f"{scenario_name}_ranks_table_{affected}.tex")
    with open(filename, 'w') as f:
        f.write(latex_content)

    print(f"Saved: {filename}")












