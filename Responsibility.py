import math

import numpy as np;

np.random.seed(0)
import copy
from tqdm import tqdm
import GWorld
import Agent
from itertools import permutations, combinations
import math

import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import PlotGWorld
plotgw = PlotGWorld.PlotGWorld();  # Object for accessing plotters

from functools import lru_cache

VerboseFlag = False
EPS = 0.000001

# DISABLE_TQDM = True
DISABLE_TQDM = False


def compare_valids(func):
    """
    Decorator that compares the output of CountValidMovesOfAffected
    with WorldIn.get_feasibile_actions_for_affected for every call.
    """

    def wrapper(WorldIn, ActionID4Agents, AffectedID, show_plots=False,
                bypass=True,
                *args, **kwargs):

        if bypass:
            #  Just return the result of the newer faster method
            return WorldIn.get_feasibile_actions_for_affected(
                world_=WorldIn,
                affectedID=AffectedID,
                ActionID4Agents=ActionID4Agents
            )

        # Time the original function
        start_time_original = time.perf_counter()
        original_result = func(WorldIn, ActionID4Agents, AffectedID, *args, **kwargs)
        end_time_original = time.perf_counter()
        original_duration = end_time_original - start_time_original

        # Time the comparison function
        start_time_comparison = time.perf_counter()
        comparison_result = WorldIn.get_feasibile_actions_for_affected(
            world_=WorldIn,
            affectedID=AffectedID,
            ActionID4Agents=ActionID4Agents
        )
        end_time_comparison = time.perf_counter()
        comparison_duration = end_time_comparison - start_time_comparison

        # Print timing results
        print(f"Original function took  : {original_duration:.6f} seconds")
        print(f"Comparison function took: {comparison_duration:.6f} seconds")

        print('--------------------------------------------------------------')
        print(f"  CountValidMovesOfAffected: \n{original_result}")
        print(f"  get_feasibile_actions_for_affected: \n{comparison_result}")

        count1, validity1 = original_result
        count2, validity2 = comparison_result

        results_match = False
        if count1 == count2:
            try:
                validity_diff = validity1 - validity2
                if validity_diff.sum() == 0:
                    results_match = True
            except:
                results_match = False

        # Print comparison results
        if results_match:
            print(f"✓ Results match for AffectedID {AffectedID}")
        else:
            comparison_ = '>' if count1 > count2 else '<'
            print(f"✗ MISMATCH for AffectedID {AffectedID}! "
                  f"CountValidMovesOfAffected {comparison_} get_feasibile_actions_for_affected")

        if show_plots and not results_match:
            fig = plt.figure(figsize=(12, 10))
            fig.suptitle(f'Affected: {AffectedID + 1}')

            # Create a 2x2 grid
            gs = gridspec.GridSpec(2, 2, figure=fig,
                                   height_ratios=[2, 1],
                                   width_ratios=[1, 1],
                                   hspace=0.3, wspace=0.3)

            # Create axes
            axes = [
                fig.add_subplot(gs[0, :]),  # Top row, full width (spans both columns)
                fig.add_subplot(gs[1, 0]),  # Bottom left
                fig.add_subplot(gs[1, 1])  # Bottom right
            ]

            plotgw.ViewGWorld(WorldIn, ViewNextStep=True, ViewActionTrail=False, ax=axes[0],
                              Animate=False, game_mode=True)

            PlotGWorld.plot_valid_actions(validity_of_actions=validity1,
                                          ax=axes[1],
                                          title='CountValidMovesOfAffected')

            PlotGWorld.plot_valid_actions(validity_of_actions=validity2,
                                          ax=axes[2],
                                          title='get_feasibile_actions_for_affected')

            filepath = os.path.join('TestingValidMoves', f'{end_time_comparison:.2f}')
            fig.savefig(filepath + '.png')

        # Return the original result (function behavior unchanged)
        return original_result

    return wrapper


@compare_valids
def CountValidMovesOfAffected(WorldIn, ActionID4Agents, AffectedID):
    """Returns the CountValidMovesOfAffected_tuple function for the affected agent"""

    return CountValidMovesOfAffected_tuple(WorldIn, tuple(ActionID4Agents), AffectedID)


@lru_cache(maxsize=None)
def CountValidMovesOfAffected_tuple(WorldIn, ActionID4Agents, AffectedID):
    """Returns the number of valid moves for the affected agent for the actions chosen by others"""

    # Counts ValidMoves for Affected Agent for the Actions Chosen by Others
    ActionID4Agents = list(ActionID4Agents)

    FuncWorld_outer = copy.deepcopy(WorldIn)
    ActionID4Agents_outer = copy.deepcopy(ActionID4Agents)
    Affected = FuncWorld_outer.AgentList[AffectedID]

    ValidMovesCount = 0
    validity_of_moves_of_affected = np.zeros(len(Affected.Actions))

    for AffectedActionID in np.arange(len(Affected.Actions)):
        agentIDs4swaps = [AffectedID]
        actionIDs4swaps = [AffectedActionID]
        # SwapActionID for Affected Agent
        ActionID4Agents_inner = GWorld.SwapActionIDs4Agents(ActionID4Agents=ActionID4Agents_outer,
                                                            agentIDs4swaps=agentIDs4swaps,
                                                            actionIDs4swaps=actionIDs4swaps)

        FuncWorld = copy.deepcopy(FuncWorld_outer)

        AgentCrashes, RestrictedMoves = FuncWorld.UpdateGWorld(defaultAction='stay',
                                                               ActionID4Agents=ActionID4Agents_inner)

        if (AgentCrashes[AffectedID] is False) and (RestrictedMoves[AffectedID] is False):
            ValidMovesCount += 1
            validity_of_moves_of_affected[AffectedActionID] = 1
            # validity_of_moves_of_affected = 0 for crashes

        del FuncWorld
    del FuncWorld_outer

    return ValidMovesCount, validity_of_moves_of_affected


def FeAR(WorldIn, ActionID4Agents, MovesDeRigueur4Agents=[]):
    """Calculates the Feasible Action-Space Reduction Metric for all agents"""

    FuncWorld = copy.deepcopy(WorldIn)

    # Storing the Actions received for each agent
    ActionInputs = np.ones(len(FuncWorld.AgentList)).astype(int) * 0  # Default is Stay
    for AgentID, ActionID in ActionID4Agents:
        ActionInputs[AgentID] = ActionID

    # Storing the Move de Rigueurs received for each agent
    MovesDeRigueur = np.ones(len(FuncWorld.AgentList)).astype(int) * 0  # Default Move de Riguer is Stay
    for AgentID, ActionID in MovesDeRigueur4Agents:
        MovesDeRigueur[AgentID] = ActionID

    Resp = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList)))
    ValidMoves_moveDeRigueur = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList)))
    ValidMoves_action = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList)))

    list_of_actions_for_agents = []
    for agentID in FuncWorld.AgentList:
        list_of_actions_for_agents.append(len(agentID.Actions))
    max_n_actions = max(list_of_actions_for_agents)
    if VerboseFlag: print('max_n_actions : ', max_n_actions)
    Validity_of_Moves_moveDeRigueur = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList), max_n_actions))
    Validity_of_Moves_action = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList), max_n_actions))

    for ii in tqdm(range(len(FuncWorld.AgentList)), colour="red", ncols=100):  # Actors
        for jj in np.arange(len(FuncWorld.AgentList)):  # Affected
            if not (ii == jj):

                agentIDs4swaps = [ii]

                # Actor - Move de Rigueur
                actionIDs4swaps = [MovesDeRigueur[ii]]
                ActionID4Agents_ActorMoveDeRigueur = GWorld.SwapActionIDs4Agents(ActionID4Agents=ActionID4Agents,
                                                                                 agentIDs4swaps=agentIDs4swaps,
                                                                                 actionIDs4swaps=actionIDs4swaps)

                # Actor Moves
                actionIDs4swaps = [ActionInputs[ii]]
                ActionID4Agents_ActorMoves = GWorld.SwapActionIDs4Agents(ActionID4Agents=ActionID4Agents,
                                                                         agentIDs4swaps=agentIDs4swaps,
                                                                         actionIDs4swaps=actionIDs4swaps)

                if VerboseFlag:
                    print('Actor {:02d} Moves'.format(ii + 1))
                    print('ActionIDs_ActorStays :', ActionID4Agents_ActorMoveDeRigueur)
                    print('ActionIDs_ActorMoves :', ActionID4Agents_ActorMoves)

                ValidMoves_moveDeRigueur[ii][jj], Validity_of_Moves_moveDeRigueur[ii][jj] = \
                    CountValidMovesOfAffected(WorldIn=FuncWorld,
                                              ActionID4Agents=ActionID4Agents_ActorMoveDeRigueur,
                                              AffectedID=jj)
                ValidMoves_action[ii][jj], Validity_of_Moves_action[ii][jj] = \
                    CountValidMovesOfAffected(WorldIn=FuncWorld,
                                              ActionID4Agents=ActionID4Agents_ActorMoves,
                                              AffectedID=jj)

                Resp[ii][jj] = (ValidMoves_moveDeRigueur[ii][jj] - ValidMoves_action[ii][jj]) / \
                               (ValidMoves_moveDeRigueur[ii][jj] + EPS)
                # 0.1 is added to the denominator to resolve cases when ValidMoves_stay is 0

                Resp[ii][jj] = np.clip(Resp[ii][jj], -1, 1)
                # Clipping Resp to the range [-1,1]

    ValidMoves_moveDeRigueur = ValidMoves_moveDeRigueur.astype(int)
    ValidMoves_action = ValidMoves_action.astype(int)
    Validity_of_Moves_moveDeRigueur = Validity_of_Moves_moveDeRigueur.astype(int)
    Validity_of_Moves_action = Validity_of_Moves_action.astype(int)

    if VerboseFlag:
        print('Validity_of_Moves_moveDeRigueur : ', Validity_of_Moves_moveDeRigueur)
        print('Validity_of_Moves_action : ', Validity_of_Moves_action)

    return Resp, ValidMoves_moveDeRigueur, ValidMoves_action, Validity_of_Moves_moveDeRigueur, Validity_of_Moves_action


def FeAR_4_one_actor(WorldIn, ActionID4Agents, MovesDeRigueur4Agents=[], actor_ii=0):
    # Feasible Action-Space Reduction Metric

    FuncWorld = copy.deepcopy(WorldIn)

    # Storing the Actions received for each agent
    ActionInputs = np.ones(len(FuncWorld.AgentList)).astype(int) * 0  # Default is Stay
    for AgentID, ActionID in ActionID4Agents:
        ActionInputs[AgentID] = ActionID

    # Storing the Move de Rigueurs received for each agent
    MovesDeRigueur = np.ones(len(FuncWorld.AgentList)).astype(int) * 0  # Default Move de Riguer is Stay
    for AgentID, ActionID in MovesDeRigueur4Agents:
        MovesDeRigueur[AgentID] = ActionID

    Resp = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList)))
    ValidMoves_moveDeRigueur = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList)))
    ValidMoves_action = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList)))

    list_of_actions_for_agents = []
    for agentID in FuncWorld.AgentList:
        list_of_actions_for_agents.append(len(agentID.Actions))
    max_n_actions = max(list_of_actions_for_agents)
    if VerboseFlag: print('max_n_actions : ', max_n_actions)
    Validity_of_Moves_moveDeRigueur = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList), max_n_actions))
    Validity_of_Moves_action = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList), max_n_actions))

    ii = actor_ii
    for jj in tqdm(range(len(FuncWorld.AgentList)), colour="red", ncols=100):  # Affected
        if not (ii == jj):

            agentIDs4swaps = [ii]

            # Actor - Move de Rigueur
            actionIDs4swaps = [MovesDeRigueur[ii]]
            ActionID4Agents_ActorMoveDeRigueur = GWorld.SwapActionIDs4Agents(ActionID4Agents=ActionID4Agents,
                                                                             agentIDs4swaps=agentIDs4swaps,
                                                                             actionIDs4swaps=actionIDs4swaps)

            # Actor Moves
            actionIDs4swaps = [ActionInputs[ii]]
            ActionID4Agents_ActorMoves = GWorld.SwapActionIDs4Agents(ActionID4Agents=ActionID4Agents,
                                                                     agentIDs4swaps=agentIDs4swaps,
                                                                     actionIDs4swaps=actionIDs4swaps)

            if VerboseFlag:
                print('Actor {:02d} Moves'.format(ii + 1))
                print('ActionIDs_ActorStays :', ActionID4Agents_ActorMoveDeRigueur)
                print('ActionIDs_ActorMoves :', ActionID4Agents_ActorMoves)

            ValidMoves_moveDeRigueur[ii][jj], Validity_of_Moves_moveDeRigueur[ii][jj] = \
                CountValidMovesOfAffected(WorldIn=FuncWorld,
                                          ActionID4Agents=ActionID4Agents_ActorMoveDeRigueur,
                                          AffectedID=jj)
            ValidMoves_action[ii][jj], Validity_of_Moves_action[ii][jj] = \
                CountValidMovesOfAffected(WorldIn=FuncWorld,
                                          ActionID4Agents=ActionID4Agents_ActorMoves,
                                          AffectedID=jj)

            Resp[ii][jj] = (ValidMoves_moveDeRigueur[ii][jj] - ValidMoves_action[ii][jj]) / \
                           (ValidMoves_moveDeRigueur[ii][jj] + EPS)
            # 0.1 is added to the denominator to resolve cases when ValidMoves_stay is 0

            Resp[ii][jj] = np.clip(Resp[ii][jj], -1, 1)
            # Clipping Resp to the range [-1,1]

    ValidMoves_moveDeRigueur = ValidMoves_moveDeRigueur.astype(int)
    ValidMoves_action = ValidMoves_action.astype(int)
    Validity_of_Moves_moveDeRigueur = Validity_of_Moves_moveDeRigueur.astype(int)
    Validity_of_Moves_action = Validity_of_Moves_action.astype(int)

    if VerboseFlag:
        print('Validity_of_Moves_moveDeRigueur : ', Validity_of_Moves_moveDeRigueur)
        print('Validity_of_Moves_action : ', Validity_of_Moves_action)

    return Resp, ValidMoves_moveDeRigueur, ValidMoves_action, Validity_of_Moves_moveDeRigueur, Validity_of_Moves_action


def FeAR_4_affected_agents(WorldIn, ActionID4Agents, MovesDeRigueur4Agents=[], affected_jj=[0]):
    # Feasible Action-Space Reduction Metric

    FuncWorld = copy.deepcopy(WorldIn)

    # Storing the Actions received for each agent
    ActionInputs = np.ones(len(FuncWorld.AgentList)).astype(int) * 0  # Default is Stay
    for AgentID, ActionID in ActionID4Agents:
        ActionInputs[AgentID] = ActionID

    # Storing the Move de Rigueurs received for each agent
    MovesDeRigueur = np.ones(len(FuncWorld.AgentList)).astype(int) * 0  # Default Move de Riguer is Stay
    for AgentID, ActionID in MovesDeRigueur4Agents:
        MovesDeRigueur[AgentID] = ActionID

    Resp = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList)))
    ValidMoves_moveDeRigueur = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList)))
    ValidMoves_action = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList)))

    list_of_actions_for_agents = []
    for agentID in FuncWorld.AgentList:
        list_of_actions_for_agents.append(len(agentID.Actions))
    max_n_actions = max(list_of_actions_for_agents)
    if VerboseFlag: print('max_n_actions : ', max_n_actions)
    Validity_of_Moves_moveDeRigueur = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList), max_n_actions))
    Validity_of_Moves_action = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList), max_n_actions))

    for jj in affected_jj:
        for ii in tqdm(range(len(FuncWorld.AgentList)), colour="red", ncols=100):  # Actor
            if not (ii == jj):

                agentIDs4swaps = [ii]

                # Actor - Move de Rigueur
                actionIDs4swaps = [MovesDeRigueur[ii]]
                ActionID4Agents_ActorMoveDeRigueur = GWorld.SwapActionIDs4Agents(ActionID4Agents=ActionID4Agents,
                                                                                 agentIDs4swaps=agentIDs4swaps,
                                                                                 actionIDs4swaps=actionIDs4swaps)

                # Actor Moves
                actionIDs4swaps = [ActionInputs[ii]]
                ActionID4Agents_ActorMoves = GWorld.SwapActionIDs4Agents(ActionID4Agents=ActionID4Agents,
                                                                         agentIDs4swaps=agentIDs4swaps,
                                                                         actionIDs4swaps=actionIDs4swaps)

                if VerboseFlag:
                    print('Actor {:02d} Moves'.format(ii + 1))
                    print('ActionIDs_ActorStays :', ActionID4Agents_ActorMoveDeRigueur)
                    print('ActionIDs_ActorMoves :', ActionID4Agents_ActorMoves)

                ValidMoves_moveDeRigueur[ii][jj], Validity_of_Moves_moveDeRigueur[ii][jj] = \
                    CountValidMovesOfAffected(WorldIn=FuncWorld,
                                              ActionID4Agents=ActionID4Agents_ActorMoveDeRigueur,
                                              AffectedID=jj)
                ValidMoves_action[ii][jj], Validity_of_Moves_action[ii][jj] = \
                    CountValidMovesOfAffected(WorldIn=FuncWorld,
                                              ActionID4Agents=ActionID4Agents_ActorMoves,
                                              AffectedID=jj)

                Resp[ii][jj] = (ValidMoves_moveDeRigueur[ii][jj] - ValidMoves_action[ii][jj]) / \
                               (ValidMoves_moveDeRigueur[ii][jj] + EPS)
                # 0.1 is added to the denominator to resolve cases when ValidMoves_stay is 0

                Resp[ii][jj] = np.clip(Resp[ii][jj], -1, 1)
                # Clipping Resp to the range [-1,1]

    ValidMoves_moveDeRigueur = ValidMoves_moveDeRigueur.astype(int)
    ValidMoves_action = ValidMoves_action.astype(int)
    Validity_of_Moves_moveDeRigueur = Validity_of_Moves_moveDeRigueur.astype(int)
    Validity_of_Moves_action = Validity_of_Moves_action.astype(int)

    if VerboseFlag:
        print('Validity_of_Moves_moveDeRigueur : ', Validity_of_Moves_moveDeRigueur)
        print('Validity_of_Moves_action : ', Validity_of_Moves_action)

    return Resp, ValidMoves_moveDeRigueur, ValidMoves_action, Validity_of_Moves_moveDeRigueur, Validity_of_Moves_action


def GroupResponsibility(world_in, action_id_4agents, num_actors, mdr4agents=[]):
    # if affected_agent in group_ids:
    #     raise AgentInGroupError(affected_agent)

    func_world, mdr, actions = prepare_world(world_in, action_id_4agents, mdr4agents)

    list_of_actions_for_agents = []
    for agentID in func_world.AgentList:
        list_of_actions_for_agents.append(len(agentID.Actions))

    perms_all = permutations(list(range(len(func_world.AgentList))), num_actors)

    # Filter out permutations where the order doesn't matter
    perms = [perm for perm in perms_all if sorted(perm) == list(perm)]

    resp = np.zeros((len(func_world.AgentList), len(perms)))
    num_v_mdr = np.zeros((len(func_world.AgentList), len(perms)))
    num_v_a = np.zeros((len(func_world.AgentList), len(perms)))

    for affected in tqdm(range(len(func_world.AgentList)), colour="blue", ncols=100,
                         disable=DISABLE_TQDM, desc="Group Responsibility: Affected agents" ):
        for group in range(len(perms)):
            resp[affected][group], num_v_mdr[affected][group], num_v_a[affected][group] = (
                calculateSpecificGroupResponsibility(func_world, perms[group], affected, mdr, actions,
                                                     action_id_4agents))

    print(f'{GWorld.get_feasibile_actions_for_affected_tuple.cache_info()=}')

    return resp, num_v_mdr, num_v_a,

def calculateSpecificGroupResponsibility(func_worldi, group_ids, affected, mdr, actions, action_id_4agents,
                                         return_valid_actions=False):
    # print(f'{group_ids=}')
    func_world = copy.deepcopy(func_worldi)
    agentIDs4swaps = []
    mdr_swap = []
    action_swap = []
    if affected in group_ids:
        return np.nan, 0, 0
    for ii in tqdm(range(len(group_ids)), colour="red", ncols=100, disable=True):  # actors
        agentIDs4swaps.append(group_ids[ii])
        mdr_swap.append(mdr[group_ids[ii]])
        action_swap.append(actions[group_ids[ii]])

    actors_mdr = GWorld.SwapActionIDs4Agents(ActionID4Agents=action_id_4agents,
                                             agentIDs4swaps=agentIDs4swaps,
                                             actionIDs4swaps=mdr_swap)

    actors_actions = GWorld.SwapActionIDs4Agents(ActionID4Agents=action_id_4agents,
                                                 agentIDs4swaps=agentIDs4swaps,
                                                 actionIDs4swaps=action_swap)

    num_v_mdr, val_mdr = CountValidMovesOfAffected(WorldIn=func_world,
                                                   ActionID4Agents=actors_mdr,
                                                   AffectedID=affected)
    num_v_a, val_a = CountValidMovesOfAffected(WorldIn=func_world,
                                               ActionID4Agents=actors_actions,
                                               AffectedID=affected)

    resp = (num_v_mdr - num_v_a) / \
           (num_v_mdr + EPS)
    # 0.1 is added to the denominator to resolve cases when ValidMoves_stay is 0

    # Clipping Resp to the range [-1,1]
    resp = np.clip(resp, -1, 1)

    if return_valid_actions:
        return resp, num_v_mdr, num_v_a, val_mdr, val_a
    return resp, num_v_mdr, num_v_a


def FeAL(WorldIn, ActionID4Agents, MovesDeRigueur4Agents=[]):
    # Feasible Action-Space Left - for each agent
    # A measure of the agency of each agent -
    # - and thus an indicator of personal causal responsibility

    FuncWorld = copy.deepcopy(WorldIn)

    # Storing the Actions received for each agent
    ActionInputs = np.ones(len(FuncWorld.AgentList)).astype(int) * 0  # Default is Stay
    for AgentID, ActionID in ActionID4Agents:
        ActionInputs[AgentID] = ActionID

    # Storing the Move de Rigueurs received for each agent
    MovesDeRigueur = np.ones(len(FuncWorld.AgentList)).astype(int) * 0  # Default Move de Riguer is Stay
    for AgentID, ActionID in MovesDeRigueur4Agents:
        MovesDeRigueur[AgentID] = ActionID

    FeAL = np.zeros(len(FuncWorld.AgentList))
    ValidMoves_moveDeRigueur_FeAL = np.zeros(len(FuncWorld.AgentList))
    ValidMoves_action_FeAL = np.zeros(len(FuncWorld.AgentList))

    list_of_actions_for_agents = []
    for agentID in FuncWorld.AgentList:
        list_of_actions_for_agents.append(len(agentID.Actions))
    max_n_actions = max(list_of_actions_for_agents)
    if VerboseFlag: print('max_n_actions : ', max_n_actions)
    Validity_of_Moves_MdR_FeAL = np.zeros((len(FuncWorld.AgentList), max_n_actions))
    Validity_of_Moves_action_FeAL = np.zeros((len(FuncWorld.AgentList), max_n_actions))

    for ii in tqdm(range(len(FuncWorld.AgentList)), colour="red", ncols=100, disable=DISABLE_TQDM):  # Affected

        agentid_list = list(range(len(FuncWorld.AgentList)))
        agentid_list.pop(ii)
        agentIDs4swaps = agentid_list

        # All agents but ego agent - Move de Rigueur
        # action_mdr_list = MovesDeRigueur.copy()
        # action_mdr_list.pop(ii)
        actionIDs4swaps = np.delete(MovesDeRigueur, ii)
        if VerboseFlag:
            print("FuncWorld.AgentList,MovesDeRigueur", FuncWorld.AgentList, MovesDeRigueur)
            print("agentIDs4swaps,actionIDs4swaps : ", agentIDs4swaps, actionIDs4swaps)

        ActionID4Agents_OthersMoveDeRigueur = GWorld.SwapActionIDs4Agents(ActionID4Agents=ActionID4Agents,
                                                                          agentIDs4swaps=agentIDs4swaps,
                                                                          actionIDs4swaps=actionIDs4swaps)

        # All agents but ego agent -Actor Moves
        # action_move_list = ActionInputs
        # action_move_list.pop(ii)
        actionIDs4swaps = np.delete(ActionInputs, ii)
        if VerboseFlag:
            print("FuncWorld.AgentList,MovesDeRigueur", FuncWorld.AgentList, ActionInputs)
            print("agentIDs4swaps,actionIDs4swaps : ", agentIDs4swaps, actionIDs4swaps)

        ActionID4Agents_OthersMove = GWorld.SwapActionIDs4Agents(ActionID4Agents=ActionID4Agents,
                                                                 agentIDs4swaps=agentIDs4swaps,
                                                                 actionIDs4swaps=actionIDs4swaps)

        if VerboseFlag:
            print('Affected agent {:02d}!'.format(ii + 1))
            print('ActionIDs_OthersMdR :', ActionID4Agents_OthersMoveDeRigueur)
            print('ActionIDs_OthersMove :', ActionID4Agents_OthersMove)

        ValidMoves_moveDeRigueur_FeAL[ii], Validity_of_Moves_MdR_FeAL[ii] = \
            CountValidMovesOfAffected(WorldIn=FuncWorld,
                                      ActionID4Agents=ActionID4Agents_OthersMoveDeRigueur,
                                      AffectedID=ii)

        ValidMoves_action_FeAL[ii], Validity_of_Moves_action_FeAL[ii] = \
            CountValidMovesOfAffected(WorldIn=FuncWorld,
                                      ActionID4Agents=ActionID4Agents_OthersMove,
                                      AffectedID=ii)

        FeAL[ii] = (ValidMoves_action_FeAL[ii]) / \
                   (ValidMoves_moveDeRigueur_FeAL[ii] + EPS)
        # 0.1 is added to the denominator to resolve cases when ValidMoves_stay is 0
        FeAL[ii] = np.clip(FeAL[ii], -1, 1)
        # Clipping Resp to the range [-1,1]

    ValidMoves_moveDeRigueur_FeAL = ValidMoves_moveDeRigueur_FeAL.astype(int)
    ValidMoves_action_FeAL = ValidMoves_action_FeAL.astype(int)
    Validity_of_Moves_MdR_FeAL = Validity_of_Moves_MdR_FeAL.astype(int)
    Validity_of_Moves_action_FeAL = Validity_of_Moves_action_FeAL.astype(int)

    if VerboseFlag:
        print('Validity_of_Moves_moveDeRigueur : ', Validity_of_Moves_MdR_FeAL)
        print('Validity_of_Moves_action : ', Validity_of_Moves_action_FeAL)

    return FeAL, ValidMoves_moveDeRigueur_FeAL, ValidMoves_action_FeAL, \
        Validity_of_Moves_MdR_FeAL, Validity_of_Moves_action_FeAL


def prepare_world(world_in, action_id_4agents, mdr4agents):
    func_world = copy.deepcopy(world_in)

    # Storing the Actions received for each agent
    actions = np.zeros(len(func_world.AgentList))  # Default is Stay
    for agentId, actionId in action_id_4agents:
        actions[agentId] = actionId

    # Storing the Move de Rigueurs received for each agent
    mdr = np.zeros(len(func_world.AgentList))  # Default Move de Riguer is Stay
    for agentId, actionId in mdr4agents:
        mdr[agentId] = actionId

    return func_world, mdr, actions

def plot_group_effects_rox(scenario_name=None, affected=None, group=None,
                           show_title=False,
                           SaveImagestoFolder=None):
    assert scenario_name is not None
    assert affected is not None
    assert group is not None

    rng = np.random.default_rng(seed=0)

    # -----------------------------
    Scenario = GWorld.LoadJsonScenario(json_filename='Scenarios4FeARSims.json', scenario_name=scenario_name)

    N_Agents = Scenario['N_Agents']
    N_Cases = Scenario['N_Cases']
    N_iterations = Scenario['N_iterations']

    # Just a check - Minimum one iteration
    if N_iterations <= 0:
        N_iterations = 1

    ActionNames, ActionMoves = Agent.DefineActions()

    Region = np.array(Scenario['Map']['Region'])
    Walls = Scenario['Map']['Walls']
    OneWays = Scenario['Map']['OneWays']

    World = GWorld.GWorld(Region, Walls=Walls, OneWays=OneWays)  # Initialising GWorld from Scenario

    N_Agents = Scenario['N_Agents']
    AgentLocations = Scenario['AgentLocations'].copy()

    AgentLocations = []
    for location in Scenario['AgentLocations']:
        AgentLocations.append(tuple(location))

    # Adding N Agents at sorted random positions
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

    Action4Agents = World.SelectActionsForAll(defaultAction=defaultAction,
                                              InputActionID4Agents=SpecificAction4Agents)

    MdR4Agents_Default = 0  # Stay
    Specific_MdR4Agents = []  # None
    MdR4Agents = []
    # Setting the MdR for each Agent
    for ii in range(len(World.AgentList)):
        MdR4Agents.append([ii, MdR4Agents_Default])
    for agent, specific_mdr in Specific_MdR4Agents:
        MdR4Agents[agent] = [agent, specific_mdr]

    func_world, mdr, actions = prepare_world(World, Action4Agents, MdR4Agents)
    resp_, num_v_mdr_, num_v_a_, val_mdr, val_a = (
        calculateSpecificGroupResponsibility(World, group,
                                             affected,
                                             mdr, actions,
                                             Action4Agents,
                                             return_valid_actions=True))

    circ_locs = []
    cross_locs = []
    mdr_circ_locs = []

    for a in range(len(ActionNames)):
        if val_mdr[a] == 1:
            end_loc = np.array(ActionMoves[a]).sum(axis=0) + np.array(AgentLocations[affected])
            circ_locs.append(end_loc)
            # print(f'circ: {end_loc}')

        if val_a[a] - val_mdr[a] == -1:
            end_loc = np.array(ActionMoves[a]).sum(axis=0) + np.array(AgentLocations[affected])
            cross_locs.append(end_loc)
            # print(f'cross: {end_loc}')

        if val_a[a] - val_mdr[a] == 1:
            end_loc = np.array(ActionMoves[a]).sum(axis=0) + np.array(AgentLocations[affected])
            mdr_circ_locs.append(end_loc)
            # print(f'cross: {end_loc}')

    annot_rox = {
        'rects': {
            'tab:blue': {'x_s': [AgentLocations[affected][1]], 'y_s': [AgentLocations[affected][0]]},
            'tab:red': {'x_s': [AgentLocations[i][1] for i in group], 'y_s': [AgentLocations[i][0] for i in group]},
        },
        'circs': {
            'tab:blue': {'x_s': [loc[1] for loc in circ_locs], 'y_s': [loc[0] for loc in circ_locs], },
            'tab:red': {'x_s': [loc[1] for loc in mdr_circ_locs], 'y_s': [loc[0] for loc in mdr_circ_locs], },

        },
        'crosses': {
            'tab:red': {'x_s': [loc[1] for loc in cross_locs], 'y_s': [loc[0] for loc in cross_locs], },
        },
    }

    group_named = [i+1 for i in group]

    if show_title:
        title = f'{set(group_named)} on {affected+1}'
    else:
        title=None

    ax = plotgw.ViewGWorld(World, ViewNextStep=True, ViewActionTrail=False, annot_rox=annot_rox, annot_font_size=24,
                    saveFolder=SaveImagestoFolder,title=title,
                    imageName=f'{scenario_name}_effects_{group_named}_on_{affected+1}', overwrite_image=True,
                    highlight_actor=group, highlight_affected=[affected]);

    return ax


def ShapleyValue(world, action_id_4agents, mdr4agents=[]):
    """
    Calculate the Shapley value for each actor with respect to each affected agent.
    The Shapley value represents each actor's average marginal contribution to responsibility
    across all possible coalitions.

    Args:
        world: The world state
        action_id_4agents: Action IDs for agents
        mdr4agents: MDR for agents (optional)

    Returns:
        shapley_values: Array of shape (num_agents, num_agents) where shapley_values[affected][actor]
                       is the Shapley value of actor for the affected agent
    """
    func_world, mdr, actions = prepare_world(world, action_id_4agents, mdr4agents)

    num_agents = len(func_world.AgentList)
    shapley_values = np.zeros((num_agents, num_agents))

    # For each affected agent
    for affected in tqdm(range(num_agents), colour="blue", ncols=100,
                         disable=DISABLE_TQDM, desc="Shapley Value: Affected agents"):

        # Create the grand coalition excluding the affected agent
        all_actors = [i for i in range(num_agents) if i != affected]

        # For each actor in the coalition
        for actor in all_actors:
            marginal_contributions = []

            # Consider all possible coalitions that don't include this actor
            # (but exclude the affected agent from all coalitions)
            other_actors = [a for a in all_actors if a != actor]

            # Iterate through all possible subset sizes
            for coalition_size in range(len(other_actors) + 1):
                # Generate all coalitions of this size from other_actors
                for coalition in combinations(other_actors, coalition_size):
                    coalition = list(coalition)

                    # Calculate responsibility with the coalition (without actor)
                    if len(coalition) > 0:
                        resp_without, _, _ = calculateSpecificGroupResponsibility(
                            func_world, coalition, affected, mdr, actions, action_id_4agents
                        )
                    else:
                        resp_without = 0.0

                    # Calculate responsibility with the coalition plus the actor
                    coalition_with_actor = sorted(coalition + [actor])
                    resp_with, _, _ = calculateSpecificGroupResponsibility(
                        func_world, coalition_with_actor, affected, mdr, actions, action_id_4agents
                    )

                    # Marginal contribution
                    marginal_contribution = resp_with - resp_without

                    # Weight by the Shapley formula: |S|! * (n - |S| - 1)! / n!
                    # where S is the coalition, n is the total number of actors (excluding affected)
                    s = len(coalition)
                    n = len(all_actors)
                    weight = (math.factorial(s) * math.factorial(n - s - 1)) / math.factorial(n)

                    marginal_contributions.append(weight * marginal_contribution)

            # Sum all weighted marginal contributions
            shapley_values[affected][actor] = sum(marginal_contributions)

    print(f'{GWorld.get_feasibile_actions_for_affected_tuple.cache_info()=}')

    return shapley_values


def print_shapley_values(shapley_values, decimal_places=4):
    """
    Print the Shapley values in a neat, readable format.

    Args:
        shapley_values: Array of shape (num_agents, num_agents) from ShapleyValue()
        decimal_places: Number of decimal places to display (default: 4)
    """
    num_agents = shapley_values.shape[0]

    # Try to get agent names, otherwise use IDs
    try:
        agent_names = [f"Agent {i+1}" for i in range(num_agents)]
        # If your world has agent names/labels, uncomment and modify:
        # agent_names = [agent.name for agent in world_in.AgentList]
    except:
        agent_names = [f"Agent {i+1}" for i in range(num_agents)]

    print("\n" + "=" * 80)
    print(f"{'SHAPLEY VALUES - RESPONSIBILITY ATTRIBUTION':^80}")
    print("=" * 80)

    # Print for each affected agent
    for affected in range(num_agents):
        print(f"\n📊 Affected Agent: {agent_names[affected]}")
        print("-" * 120)

        # Get Shapley values for this affected agent (excluding the affected agent itself)
        actor_values = []
        for actor in range(num_agents):
            if actor != affected:
                actor_values.append((actor, shapley_values[affected][actor]))

        # Sort by Shapley value (descending)
        actor_values.sort(key=lambda x: x[1], reverse=True)

        # Find maximum absolute value for scaling
        max_abs_value = max([abs(val) for _, val in actor_values]) if actor_values else 1
        if max_abs_value == 0:
            max_abs_value = 1  # Avoid division by zero

        # Bar chart parameters
        max_bar_length = 30  # Maximum length of bar in characters

        # Print header
        print(f"{'Actor':<20} {'Shapley Value':>20} {'Courteous':>30}{'|':^1}{'Assertive':<30}")
        print("-" * 120)

        # Calculate total for summary
        total = sum([val for _, val in actor_values])

        # Print each actor's contribution
        for actor, value in actor_values:
            # Calculate bar length based on maximum absolute value
            normalized_value = value / max_abs_value
            bar_length = int(abs(normalized_value) * max_bar_length)

            # Create the bar chart
            if value >= 0:
                # Positive value - red bar to the right
                left_space = " " * max_bar_length
                bar = "\033[38;5;167m" + "█" * bar_length + "\033[0m"  # Red color
                visualization = f"{left_space}|{bar}"
            else:
                # Negative value - blue bar to the left
                bar = "\033[38;5;67m" + "█" * bar_length + "\033[0m"  # Blue color
                left_space = " " * (max_bar_length - bar_length)
                visualization = f"{left_space}{bar}|"

            print(f"{agent_names[actor]:<20} {value:>+20.{decimal_places}f} {visualization}")

        print(f"\n{'Total:':<20} {total:>+20.{decimal_places}f}")
        print(f"{'Max |value|:':<20} {max_abs_value:>20.{decimal_places}f}")

    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)

    # Overall statistics
    for actor in range(num_agents):
        total_contribution = sum(shapley_values[affected][actor]
                                 for affected in range(num_agents)
                                 if affected != actor)
        avg_contribution = total_contribution / (num_agents - 1) if num_agents > 1 else 0
        print(f"{agent_names[actor]:<20} - Total: {total_contribution:+.{decimal_places}f}, "
              f"Average: {avg_contribution:+.{decimal_places}f}")

    print("=" * 120 + "\n")


def print_shapley_values_table(shapley_values, decimal_places=4):
    """
    Print Shapley values as a table/matrix.

    Args:
        shapley_values: Array of shape (num_agents, num_agents) from ShapleyValue()
        decimal_places: Number of decimal places to display
    """
    num_agents = shapley_values.shape[0]

    try:
        agent_names = [f"Agent {i+1}" for i in range(num_agents)]
    except:
        agent_names = [f"Agent {i+1}" for i in range(num_agents)]

    print("\n" + "=" * 120)
    print("SHAPLEY VALUES MATRIX")
    print("Rows: Affected Agents | Columns: Actor Contributions")
    print("=" * 120)

    # Header row
    header = f"{'Affected - Actor':<20}"
    for name in agent_names:
        header += f"{name:<15}"
    print(header)
    print("-" * 120)

    # Data rows
    for affected in range(num_agents):
        row = f"{agent_names[affected]:<20}"
        for actor in range(num_agents):
            if actor == affected:
                row += f"{'---':<15}"  # No self-responsibility
            else:
                row += f"{shapley_values[affected][actor]:<+15.{decimal_places}f}"
        print(row)

    print("=" * 120 + "\n")


class AgentInGroupError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Agent: {self.value} is contained in the group of agents"
