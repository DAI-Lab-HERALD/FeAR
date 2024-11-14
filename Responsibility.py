import numpy as np;

np.random.seed(0)
import copy
from tqdm import tqdm
import GWorld
import Agent

from functools import lru_cache

VerboseFlag = False
EPS = 0.000001


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
            print("FuncWorld.AgentList,MovesDeRigueur",FuncWorld.AgentList,MovesDeRigueur)
            print("agentIDs4swaps,actionIDs4swaps : ",agentIDs4swaps,actionIDs4swaps)

        ActionID4Agents_OthersMoveDeRigueur = GWorld.SwapActionIDs4Agents(ActionID4Agents=ActionID4Agents,
                                                                         agentIDs4swaps=agentIDs4swaps,
                                                                         actionIDs4swaps=actionIDs4swaps)

        # All agents but ego agent -Actor Moves
        # action_move_list = ActionInputs
        # action_move_list.pop(ii)
        actionIDs4swaps = np.delete(ActionInputs, ii)
        if VerboseFlag:
            print("FuncWorld.AgentList,MovesDeRigueur",FuncWorld.AgentList,ActionInputs)
            print("agentIDs4swaps,actionIDs4swaps : ",agentIDs4swaps,actionIDs4swaps)

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

