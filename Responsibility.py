import numpy as np;

np.random.seed(0)
import copy
from tqdm import tqdm
import GWorld
import Agent

import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import PlotGWorld

plotgw = PlotGWorld.PlotGWorld();  # Object for accessing plotters

from functools import lru_cache

VerboseFlag = False
EPS = 0.000001


def compare_valids(func):
    """
    Decorator that compares the output of CountValidMovesOfAffected
    with WorldIn.get_feasibile_actions_for_affected for every call.
    """

    def wrapper(WorldIn, ActionID4Agents, AffectedID, show_plots=True,
                bypass=True,
                *args, **kwargs):

        if bypass:
            #  Just return the result of the newer faster method
            return WorldIn.get_feasibile_actions_for_affected(
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
    return CountValidMovesOfAffected_tuple(WorldIn, tuple(ActionID4Agents), AffectedID)


@lru_cache(maxsize=None)
def CountValidMovesOfAffected_tuple(WorldIn, ActionID4Agents, AffectedID):
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

    for ii in tqdm(range(len(FuncWorld.AgentList)), colour="red", ncols=100):  # Affected

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


def panic_4_fear(WorldIn, ActionID4Agents, MovesDeRigueur4Agents=[]):
    # Feasible Action-Space Reduction Metric with Probabilistic AND Information Criteria

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
    p_ValidMoves_moveDeRigueur = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList)))
    p_ValidMoves_action = np.zeros((len(FuncWorld.AgentList), len(FuncWorld.AgentList)))

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

                policy_affected = FuncWorld.AgentList[jj].ActionPolicy

                print(f'{policy_affected=}')
                print(f'{Validity_of_Moves_moveDeRigueur[ii][jj]=}')
                print(f'{Validity_of_Moves_action[ii][jj]=}')

                p_ValidMoves_moveDeRigueur[ii][jj] = np.dot(Validity_of_Moves_moveDeRigueur[ii][jj], policy_affected)
                p_ValidMoves_action[ii][jj] = np.dot(Validity_of_Moves_action[ii][jj], policy_affected)

                Resp[ii][jj] = (p_ValidMoves_moveDeRigueur[ii][jj] - p_ValidMoves_action[ii][jj]) / \
                               (p_ValidMoves_moveDeRigueur[ii][jj] + EPS)
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
