# Functions for computing metrics related to Complexity and Emergence
import numpy as np; np.random.seed(0)
import copy
from tqdm import tqdm
import GWorld
import Agent
from scipy.stats import entropy as entropy


def PolicyEntropy4Agents(World):
    PolicyEntropy = np.zeros(len(World.AgentList))

    for ii,agent in enumerate(World.AgentList):
        if not(agent.ActionPolicy is None):
            PolicyEntropy[ii] = entropy(agent.ActionPolicy)

    return PolicyEntropy

def GetMapofAgentEntropies(World,WorldState = None):
    PolicyEntropy = PolicyEntropy4Agents(World)

    # The option for passing in WorldState is provided to enable the choice for showing or hiding ActionTrails
    # When VeiwActionTrails is truned off, WorldState is computed based on the latest positions of the agents.
    if WorldState is None:
        WorldState = copy.copy(World.WorldState)

    EntropyMap = copy.copy(WorldState)

    for loc,agentID in np.ndenumerate(WorldState):
        if agentID>0 :
            EntropyMap[loc] = PolicyEntropy[(agentID-1).astype(int)] # AgentID is (agent index + 1)

    return EntropyMap



