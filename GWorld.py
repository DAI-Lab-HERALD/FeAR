import numpy as np;

np.random.seed(0)
import json
import pprint
import copy
from functools import lru_cache

import Agent

VerboseFlag = False
# VerboseFlag = True


class GWorld:

    def __init__(self, Map, Walls=[], OneWays=[]):

        self.AgentList = []
        self.AgentActiveStatus = []
        self.AgentCrash = []
        self.RestrictedMove = []
        self.AgentLocations = []
        self.MaxSteps = 4
        self.PreviousAgentLocations = []
        self.NewAgentLocations = []

        self.WorldMap = Map  # Grid with 0 for inactive cells and 1 for active cells

        self.WorldState = np.where(Map == 0, np.nan, Map)  # Inactive Cells - NaN
        self.WorldState = np.where(self.WorldState == 1, 0, self.WorldState)  # Active Cells - 0 at the start

        # Path restrictions = Walls and OneWays
        self.RestrictedPaths = []
        # Walls - Restricts movements in both directions
        # Check and Add Walls
        self.WorldWalls = []
        for wall in Walls:
            x1 = wall[0][0]
            y1 = wall[0][1]
            x2 = wall[1][0]
            y2 = wall[1][1]

            # Check whether the indices are within bounds
            if (x1 < self.WorldMap.shape[0]) and (x2 < self.WorldMap.shape[0]) and \
                    (y1 < self.WorldMap.shape[1]) and (y2 < self.WorldMap.shape[1]):
                # Check whether the indices are next to each other
                # i.e. they have atleast one common index
                # and the other index differs only by 1
                if (x1 == x2):
                    if ((abs(y1 - y2)) == 1):
                        self.WorldWalls.append(wall)
                        # Add both directions to restricted paths
                        self.RestrictedPaths.append(wall)
                        self.RestrictedPaths.append([wall[1], wall[0]])
                elif (y1 == y2):
                    if ((abs(x1 - x2)) == 1):
                        self.WorldWalls.append(wall)
                        # Add both directions to restricted paths
                        self.RestrictedPaths.append(wall)
                        self.RestrictedPaths.append([wall[1], wall[0]])

        # OneWays - Restricts movement in the opposite direction
        # Check and Add OneWays
        self.WorldOneWays = []
        for way in OneWays:
            x1 = way[0][0]
            y1 = way[0][1]
            x2 = way[1][0]
            y2 = way[1][1]

            # Check whether the indices are within bounds
            if (x1 < self.WorldMap.shape[0]) and (x2 < self.WorldMap.shape[0]) and \
                    (y1 < self.WorldMap.shape[1]) and (y2 < self.WorldMap.shape[1]):
                # Check whether the indices are next to each other
                # i.e. they have atleast one common index
                # and the other index differs only by 1
                if (x1 == x2):
                    if ((abs(y1 - y2)) == 1):
                        self.WorldOneWays.append(way)
                        # Add opposite direction to restricted paths
                        self.RestrictedPaths.append([way[1], way[0]])
                elif (y1 == y2):
                    if ((abs(x1 - x2)) == 1):
                        self.WorldOneWays.append(way)
                        # Add opposite direction to restricted paths
                        self.RestrictedPaths.append([way[1], way[0]])

        # #         #--- For Visualisation----
        # self.Gfig, self.Gax = plt.subplots()
        # #         line, = self.Gax.plot([])     # A tuple unpacking to unpack the only plot
        # plt.clf()
        # plt.axis('equal')

    #         #----

    #         self.AnimateGWorld()

    def __eq__(self, other):
        if not isinstance(other, GWorld):
            return NotImplemented
        return (self.AgentList == other.AgentList and
                self.AgentActiveStatus == other.AgentActiveStatus and
                self.AgentCrash == other.AgentCrash and
                self.RestrictedMove == other.RestrictedMove and
                self.AgentLocations == other.AgentLocations and
                self.MaxSteps == other.MaxSteps and
                self.PreviousAgentLocations == other.PreviousAgentLocations and
                np.array_equal(self.WorldMap, other.WorldMap) and
                np.array_equal(self.WorldState, other.WorldState, equal_nan=True) and
                self.RestrictedPaths == other.RestrictedPaths and
                self.WorldWalls == other.WorldWalls and
                self.WorldOneWays == other.WorldOneWays)

    def __hash__(self):
        return hash((tuple(self.AgentList),
                     tuple(self.AgentActiveStatus),
                     tuple(self.AgentCrash),
                     tuple(self.RestrictedMove),
                     tuple(self.AgentLocations),
                     self.MaxSteps,
                     tuple(self.PreviousAgentLocations),
                     self.WorldMap.tobytes(),
                     self.WorldState.tobytes(),
                     tuple(map(tuple, self.RestrictedPaths)),
                     tuple(map(tuple, self.WorldWalls)),
                     tuple(map(tuple, self.WorldOneWays))))

    # ----------------------------------------------------------------------------------------------- #
    # Function to Add Agent : AddAgent( Agent , Location )
    def AddAgent(self, Agent, Location, printStatus=True):
        if (0 <= Location[0] < self.WorldState.shape[0]) and ((0 <= Location[1] < self.WorldState.shape[1])):
            if self.WorldState[Location] == 0:
                if printStatus:
                    print('Adding Agent to ', Location)
                self.AgentList.append(Agent)
                self.AgentActiveStatus.append(True)
                self.AgentCrash.append(False)
                self.RestrictedMove.append(False)
                self.AgentLocations.append(Location)
                self.WorldState[Location] = 0.5  # A Dummy Value to keep track of occupied cells
                return True
            elif self.WorldState[Location] > 0:
                print('Adding Agent Aborted! Collision at spawn location : ', Location)
                return False
            else:
                print('Adding Agent Aborted! Invalid Location! : ', Location)
                return False
        else:
            print('Adding Agent Aborted! Location out of bounds ! : ', Location)
            return False

    # ----------------------------------------------------------------------------------------------- #
    # Function to remove an Agent !
    def RemoveAgent(self, agentID):
        if VerboseFlag: print(f'Removing agent {agentID + 1} !')

        location = self.AgentLocations[agentID]
        self.WorldState[location] = 0

        self.AgentList.pop(agentID)
        self.AgentActiveStatus.pop(agentID)
        self.AgentCrash.pop(agentID)
        self.RestrictedMove.pop(agentID)
        self.AgentLocations.pop(agentID)

        pass

    # ----------------------------------------------------------------------------------------------- #
    # Function get ActionSelection for Agents to be used in UpdateGWorld
    def getActionSelection4UpdateGWorld(self, ActionID4Agents=[], VerboseFlag=False, defaultAction='random'):

        # Storing the Actions received for each agent
        ActionInputs = np.ones(len(self.AgentList)).astype(int) * -1
        for AgentID, ActionID in ActionID4Agents:
            if AgentID in np.arange(len(self.AgentList)):
                ActionInputs[AgentID] = ActionID

        # Getting  Selected Actions from Agents
        for idx, agent in enumerate(self.AgentList):
            if self.AgentActiveStatus[idx] is True:

                if VerboseFlag: print("Getting the selected action for Active Agent ", idx + 1)
                # Action Selection
                # If Agent was given an ActionID
                if ActionInputs[idx] in np.arange(len(agent.Actions)):
                    if VerboseFlag: print('Selected ActionID :', ActionInputs[idx])
                    agent.ActionSelection(ActionID=ActionInputs[idx])
                # Default Actions
                else:
                    if defaultAction == 'random':
                        if VerboseFlag: print('Selecting Random Action')
                        agent.ActionSelection()
                    elif defaultAction == 'stay':
                        if VerboseFlag: print('Selected Action: Stay')
                        agent.ActionSelection(ActionID=0)
                    elif defaultAction in np.arange(len(agent.Actions)):
                        if VerboseFlag: print('Selected ActionID :', defaultAction)
                        agent.ActionSelection(ActionID=defaultAction)
                    else:
                        if VerboseFlag: print('Selecting Random Action')
                        agent.ActionSelection()
        pass

    # ----------------------------------------------------------------------------------------------- #
    # Function to revert steps with collisions and update NewAgentLocations !
    def revertStepsWithCollisions(self, step=None, NewAgentLocations_CurrentFloor=None):

        if step is None:
            print('Step not passed.')
            return False
        if NewAgentLocations_CurrentFloor is None:
            print('NewAgentLocations_CurrentFloor not passed.')
            return False

        # Revert the Moves of Crashed Agents
        for ii in np.arange(len(self.AgentList)):
            if self.AgentCrash[ii] == True:
                step_ii = (step + 1) * (len(self.AgentList[ii].SelectedAction)) / (self.MaxSteps)
                step_ii_floor = np.floor(step_ii).astype(int)
                # Make the agent stationary for all subsequent ActionSteps
                for kk in np.arange(step_ii_floor, len(self.NewAgentLocations[ii])):
                    self.NewAgentLocations[ii][kk] = self.AgentLocations[ii]
                # Make the floor location the last agent location.
                NewAgentLocations_CurrentFloor[ii] = self.AgentLocations[ii]
        return NewAgentLocations_CurrentFloor

    # ----------------------------------------------------------------------------------------------- #
    # Function to update AgentLocations and WorldLocations based on NewAgentLocations !
    def update_agent_locations_2_world_state(self, NewAgentLocations_CurrentFloor=None, VerboseFlag=False):

        if NewAgentLocations_CurrentFloor is None:
            print('NewAgentLocations_CurrentFloor not passed.')
            return False

        for idx, agent in enumerate(self.AgentList):

            # Update the Locations of Agents
            self.AgentLocations[idx] = NewAgentLocations_CurrentFloor[idx]
            if VerboseFlag: print('Updating AgentLocation from Last NewAgentLocation')
            if VerboseFlag: print('NewAgentLocations', self.NewAgentLocations)
            if VerboseFlag: print('AgentLocations', self.AgentLocations)

            if self.AgentActiveStatus[idx] is True:
                if VerboseFlag: print('Updating Location of Agent ', idx + 1, ' : ', self.AgentLocations[idx])
                # Updating Positions of Agents in WorldState
                self.WorldState[(self.AgentLocations[idx])] = idx + 1
        pass

    def collision_checks_and_resolution(self, step=None,
                                        NewAgentLocations_CurrentFloor=None,
                                        resolve=True,
                                        VerboseFlag=False, ):

        if step is None:
            print('Step not passed.')
            return False
        if NewAgentLocations_CurrentFloor is None:
            print('NewAgentLocations_CurrentFloor not passed.')
            return False

        # ..........................................#

        # Check for collisions in the NewAgentLocations
        # And revert moves that lead to collisions
        NumberOfAgents = len(self.AgentList)
        CollisionCount = NumberOfAgents  # No Agents => No Collisions
        LoopCount = 0  # Fail Safe in case of Unresolved Collisions

        while (CollisionCount > 0) and (LoopCount < 2 * NumberOfAgents):  # Arbitrary upper limit for LoopCount
            LoopCount += 1

            if VerboseFlag: print('Collisions : ', CollisionCount)

            CollisionCount = 0  # Reset the counter

            for ii in np.arange(NumberOfAgents - 1):
                step_ii = (step + 1) * (len(self.AgentList[ii].SelectedAction)) / (self.MaxSteps)
                step_ii_floor = np.floor(step_ii).astype(int)
                step_ii_ceil = np.ceil(step_ii).astype(int)
                NewAgentLocations_CurrentFloor[ii] = self.NewAgentLocations[ii][step_ii_floor]
                for jj in np.arange(ii + 1, NumberOfAgents):
                    step_jj = (step + 1) * (len(self.AgentList[jj].SelectedAction)) / (self.MaxSteps)
                    step_jj_floor = np.floor(step_jj).astype(int)
                    step_jj_ceil = np.ceil(step_jj).astype(int)
                    NewAgentLocations_CurrentFloor[jj] = self.NewAgentLocations[jj][step_jj_floor]

                    if VerboseFlag:
                        print('ii,step_floor', (ii, step), 'self.NewAgentLocations[ii][step] :'
                              , self.NewAgentLocations[ii][step_ii_floor])
                        print('ii,step_ceil', (ii, step), 'self.NewAgentLocations[ii][step] :'
                              , self.NewAgentLocations[ii][step_ii_ceil])
                        print('jj.step_floor', (jj, step), 'self.NewAgentLocations[jj][step] :'
                              , self.NewAgentLocations[jj][step_jj_floor])
                        print('jj.step_ceil', (jj, step), 'self.NewAgentLocations[jj][step] :'
                              , self.NewAgentLocations[jj][step_jj_ceil])

                    if (self.NewAgentLocations[ii][step_ii_floor] == self.NewAgentLocations[jj][step_jj_floor]) or \
                            (self.NewAgentLocations[ii][step_ii_ceil] == self.NewAgentLocations[jj][
                                step_jj_ceil]):
                        # COLLISION !
                        if VerboseFlag:
                            print('Collision !')
                            self.print_collision_report(ii, jj, step_ii_ceil, step_ii_floor, step_jj_ceil,
                                                        step_jj_floor)
                        CollisionCount = self.record_collision(CollisionCount, ii, jj)
                        # ---------------------------------#
                        # ---------------------------------#
                        # Agent HEALTH UPDATES go here !! #
                        # ---------------------------------#
                        # ---------------------------------#

                    elif (self.NewAgentLocations[ii][step_ii_floor] == self.NewAgentLocations[jj][
                        step_jj_ceil]) and \
                            (self.NewAgentLocations[ii][step_ii_ceil] == self.NewAgentLocations[jj][
                                step_jj_floor]):
                        # CROSSOVER COLLISION ! in NewLocations
                        if VerboseFlag:
                            print('Crossover Collision !')
                            self.print_collision_report(ii, jj, step_ii_ceil, step_ii_floor, step_jj_ceil,
                                                        step_jj_floor)
                        CollisionCount = self.record_collision(CollisionCount, ii, jj)
                        # ---------------------------------#
                        # ---------------------------------#
                        # Agent HEALTH UPDATES go here !! #
                        # ---------------------------------#
                        # ---------------------------------#

                    elif self.NewAgentLocations[ii][step_ii_floor] == self.NewAgentLocations[jj][step_jj_ceil]:
                        # Possible Collision - if the hangovers don't match

                        overhang_floor = step_ii_ceil - step_ii
                        overhang_ceil = step_jj - step_jj_floor
                        overhang_condition = ((overhang_floor + overhang_ceil) <= 1)

                        direction_ii = np.array(self.NewAgentLocations[ii][step_ii_ceil]) \
                                       - np.array(self.NewAgentLocations[ii][step_ii_floor])
                        direction_jj = np.array(self.NewAgentLocations[jj][step_jj_ceil]) \
                                       - np.array(self.NewAgentLocations[jj][step_jj_floor])
                        same_direction_condition = np.array_equal(direction_ii, direction_jj)

                        if VerboseFlag:
                            print('same_direction_condition :', same_direction_condition)
                            print('overhang_condition :', overhang_condition)

                        # If the above 2 conditions are met, then the agents can slide along without hitting each other

                        if not (overhang_condition and same_direction_condition):
                            # Collision happens in the case where the above conditions are not met.
                            if VerboseFlag:
                                print('Collision !')
                                self.print_collision_report(ii, jj, step_ii_ceil, step_ii_floor, step_jj_ceil,
                                                            step_jj_floor)

                            CollisionCount = self.record_collision(CollisionCount, ii, jj)
                            # ---------------------------------#
                            # ---------------------------------#
                            # Agent HEALTH UPDATES go here !! #
                            # ---------------------------------#
                            # ---------------------------------#

                    elif self.NewAgentLocations[ii][step_ii_ceil] == self.NewAgentLocations[jj][step_jj_floor]:
                        # Possible Collision - if the hangovers don't match
                        overhang_floor = step_jj_ceil - step_jj
                        overhang_ceil = step_ii - step_ii_floor
                        overhang_condition = ((overhang_floor + overhang_ceil) <= 1)

                        direction_ii = np.array(self.NewAgentLocations[ii][step_ii_ceil]) \
                                       - np.array(self.NewAgentLocations[ii][step_ii_floor])
                        direction_jj = np.array(self.NewAgentLocations[jj][step_jj_ceil]) \
                                       - np.array(self.NewAgentLocations[jj][step_jj_floor])
                        same_direction_condition = np.array_equal(direction_ii, direction_jj)

                        if VerboseFlag:
                            print('same_direction_condition :', same_direction_condition)
                            print('overhang_condition :', overhang_condition)

                        # If the above 2 conditions are met, then the agents can slide along without hitting each other

                        if not (overhang_condition and same_direction_condition):
                            # Collision happens in the case where the above conditions are not met.
                            if VerboseFlag:
                                print('Collision !')
                                self.print_collision_report(ii, jj, step_ii_ceil, step_ii_floor, step_jj_ceil,
                                                            step_jj_floor)
                            CollisionCount = self.record_collision(CollisionCount, ii, jj)
                            # ---------------------------------#
                            # ---------------------------------#
                            # Agent HEALTH UPDATES go here !! #
                            # ---------------------------------#
                            # ---------------------------------#

                    elif (((self.NewAgentLocations[ii][step_ii_floor] == self.AgentLocations[jj]) and
                           (self.AgentLocations[ii] == self.NewAgentLocations[jj][step_jj_floor])) or
                          ((self.NewAgentLocations[ii][step_ii_ceil] == self.AgentLocations[jj]) and
                           (self.AgentLocations[ii] == self.NewAgentLocations[jj][step_jj_ceil])) or
                          ((self.NewAgentLocations[ii][step_ii_floor] == self.AgentLocations[jj]) and
                           (self.AgentLocations[ii] == self.NewAgentLocations[jj][step_jj_ceil])) or
                          ((self.NewAgentLocations[ii][step_ii_ceil] == self.AgentLocations[jj]) and
                           (self.AgentLocations[ii] == self.NewAgentLocations[jj][step_jj_floor]))):
                        # CROSSOVER COLLISION ! - NewLocations with OldLocations
                        if VerboseFlag:
                            print('Crossover Collision !')
                            self.print_collision_report(ii, jj, step_ii_ceil, step_ii_floor, step_jj_ceil,
                                                        step_jj_floor)
                        CollisionCount = self.record_collision(CollisionCount, ii, jj)

                        # ---------------------------------#
                        # ---------------------------------#
                        # Agent HEALTH UPDATES go here !! #
                        # ---------------------------------#
                        # ---------------------------------#

            ########################################################################################################

            if resolve:
                NewAgentLocations_CurrentFloor = self.revertStepsWithCollisions(step=step,
                                                                                NewAgentLocations_CurrentFloor=
                                                                                NewAgentLocations_CurrentFloor)

            ########################################################################################################

            # Fail Safe in case of Unresolved Collisions
            if LoopCount >= 2 * NumberOfAgents:
                print('Collisions Not Resolved !!!')

        # ..........................................#
        return NewAgentLocations_CurrentFloor

    def record_collision(self, CollisionCount, ii, jj):
        CollisionCount += 1
        # Record Crash
        self.AgentCrash[ii] = True
        self.AgentCrash[jj] = True
        return CollisionCount

    def print_collision_report(self, ii, jj, step_ii_ceil, step_ii_floor, step_jj_ceil, step_jj_floor):
        print('Agent ', ii + 1, ' : Old :', self.AgentLocations[ii])
        print('Agent ', ii + 1, ' : New :', self.NewAgentLocations[ii][step_ii_floor])
        print('Agent ', ii + 1, ' : New :', self.NewAgentLocations[ii][step_ii_ceil])
        print('Agent ', jj + 1, ' : Old :', self.AgentLocations[jj])
        print('Agent ', jj + 1, ' : New :', self.NewAgentLocations[jj][step_jj_floor])
        print('Agent ', jj + 1, ' : New :', self.NewAgentLocations[jj][step_jj_ceil])

    # ----------------------------------------------------------------------------------------------- #
    # Function where the actions chosen by the agents are used to update the state of the world
    def UpdateGWorld(self, defaultAction='random', ActionID4Agents=[],
                     apples=None, apple_eaters=None):
        # Get Agent Actions
        # Update AgentLocations and Check which agents are active
        # Update WorldState

        if VerboseFlag: print("Updating GWorld")

        self.PreviousAgentLocations = self.AgentLocations.copy()

        self.WorldState = np.where(self.WorldMap == 0, np.nan, self.WorldMap)  # Inactive Cells - NaN
        self.WorldState = np.where(self.WorldState == 1, 0, self.WorldState)  # Active Cells - 0 at the start

        # If nothing happens, the agents will be in the old positions
        self.NewAgentLocations = []
        for AgentLocation in self.AgentLocations:
            self.NewAgentLocations.append([AgentLocation])

        ##################################################################

        self.getActionSelection4UpdateGWorld(ActionID4Agents, VerboseFlag=VerboseFlag, defaultAction=defaultAction)

        ##################################################################

        for idx, agent in enumerate(self.AgentList):
            if self.AgentActiveStatus[idx] is True:
                # Resetting Crash Record for all the Agents
                self.AgentCrash[idx] = False

                # Resetting Restricted Moves Record for all Agents
                self.RestrictedMove[idx] = False

        apples_caught = []
        # Getting (valid) NewLocations from action steps from Agents
        for step in np.arange(self.MaxSteps):

            NewAgentLocations_CurrentFloor = self.AgentLocations.copy()

            for idx, agent in enumerate(self.AgentList):
                if (self.AgentActiveStatus[idx] is True):
                    if step < len(agent.SelectedAction) and (self.AgentCrash[idx] is False):
                        ActionStep = agent.SelectedAction[step]

                    else:
                        ActionStep = (0, 0)  # If no steps left in SelectedAction

                    if VerboseFlag:
                        print("Selected Action Name : ", agent.SelectedActionName)
                        print("Selected Action Move : ", ActionStep)

                    old_location = self.NewAgentLocations[idx][step]
                    # self.NewAgentLocations[idx][0] is the PreviousLocation (before any action step)
                    # self.NewAgentLocations[idx][step+1] = where the agent will be after "step"

                    if VerboseFlag:
                        print('Old Location: ', old_location)
                        print('idx,step:', (idx, step), 'ActionStep:', ActionStep)
                    new_location = (old_location[0] + ActionStep[0], old_location[1] + ActionStep[1])

                    if VerboseFlag: print('New Location (from step): ', new_location)

                    # Making sure that the agents are not pushed off the grid
                    new_location0 = np.clip(new_location[0], 0, self.WorldMap.shape[0] - 1)
                    new_location1 = np.clip(new_location[1], 0, self.WorldMap.shape[1] - 1)
                    # Restricted move if the new location is clipped.
                    if not new_location == (new_location0, new_location1):
                        self.RestrictedMove[idx] = True
                        new_location = (new_location0, new_location1)

                    AgentPath = [old_location, new_location]

                    # Checking if the new position is a valid location
                    if self.WorldState[new_location] >= 0:
                        # Checking is the move is along a restricted path
                        if AgentPath not in self.RestrictedPaths:
                            self.NewAgentLocations[idx].append(new_location)
                            # Print NewAgentLocation
                            if VerboseFlag: print('idx,NewAgentLocations[idx] : '
                                                  , (idx, self.NewAgentLocations[idx]))

                        else:  # Setting OldLocation in case of Restricted Paths
                            self.NewAgentLocations[idx].append(old_location)
                            # Record Restricted Move
                            self.RestrictedMove[idx] = True
                            # Print NewAgentLocation
                            if VerboseFlag: print('idx,NewAgentLocations[idx] : '
                                                  , (idx, self.NewAgentLocations[idx]))

                    else:  # Setting OldLocation in case of Invalid Location
                        self.NewAgentLocations[idx].append(old_location)
                        # Record Restricted Move
                        self.RestrictedMove[idx] = True
                        # Print NewAgentLocation
                        if VerboseFlag: print('idx,NewAgentLocations[idx] : '
                                              , (idx, self.NewAgentLocations[idx]))

                        # Updating the NewAgentLocation when it is valid

            #############################################

            NewAgentLocations_CurrentFloor = self.collision_checks_and_resolution(step=step,
                                                                                  NewAgentLocations_CurrentFloor=
                                                                                  NewAgentLocations_CurrentFloor,
                                                                                  VerboseFlag=VerboseFlag)
            #############################################
            # Check for Apples

            eaten_apple_ids = []
            if apples is not None and apple_eaters is not None:
                print(f'{apples=}')
                for idx in apple_eaters:
                    for apple_id, apple in enumerate(apples):
                        print(f'{apple_id=}, {apple=}')
                        if NewAgentLocations_CurrentFloor[idx][0] == apple[0] and \
                                NewAgentLocations_CurrentFloor[idx][1] == apple[1]:
                            apples_caught.append([idx, apple_id])
                            eaten_apple_ids.append(apple_id)
                apple_ids = list(range(len(apples)))
                uneaten_apple_ids = [x for x in apple_ids if x not in eaten_apple_ids]
                apples = apples[uneaten_apple_ids]
                print(f'{apples =}')
                print(f'{apples_caught=}')

            #############################################

            self.update_agent_locations_2_world_state(NewAgentLocations_CurrentFloor=NewAgentLocations_CurrentFloor,
                                                      VerboseFlag=VerboseFlag)

            #############################################

        agent_crashes = self.AgentCrash.copy()
        restricted_moves = self.RestrictedMove.copy()

        if apples is not None and apple_eaters is not None:
            return agent_crashes, restricted_moves, apples, apples_caught

        return agent_crashes, restricted_moves

    def SelectActionsForAll(self, defaultAction='random', InputActionID4Agents=[]):

        # Storing the Actions received for each agent
        ActionInputs = np.ones(len(self.AgentList)).astype(int) * -1
        for AgentID, ActionID in InputActionID4Agents:
            if AgentID in np.arange(len(self.AgentList)):
                ActionInputs[AgentID] = ActionID

        ActionID4Agents = []

        # Getting  Selected Actions from Agents
        for idx, agent in enumerate(self.AgentList):
            if self.AgentActiveStatus[idx] is True:

                if VerboseFlag: print("Getting the selected action for Active Agent ", idx + 1)
                # Action Selection
                # If Agent was given an ActionID
                if ActionInputs[idx] in np.arange(len(agent.Actions)):
                    if VerboseFlag: print('Selected ActionID :', ActionInputs[idx])
                    SelectedActionID = agent.ActionSelection(ActionID=ActionInputs[idx])

                # Default Actions
                else:
                    if defaultAction == 'random':
                        if VerboseFlag: print('Selecting Random Action')
                        SelectedActionID = agent.ActionSelection()
                    elif defaultAction == 'stay':
                        if VerboseFlag: print('Selected Action: Stay')
                        SelectedActionID = agent.ActionSelection(ActionID=0)
                    elif defaultAction in np.arange(len(agent.Actions)):
                        if VerboseFlag: print('Selected ActionID :', defaultAction)
                        SelectedActionID = agent.ActionSelection(ActionID=defaultAction)
                    else:
                        if VerboseFlag: print('Selecting Random Action')
                        SelectedActionID = agent.ActionSelection()

                ActionID4Agents.append((idx, SelectedActionID))

        return ActionID4Agents

    # ----------------------------------------------------------------------------------------------- #
    def GetAgentContext(self):
        pass

    # ----------------------------------------------------------------------------------------------- #
    def SetAgentSensoryStates(self, Agent):
        pass

    # ----------------------------------------------------------------------------------------------- #
    @staticmethod
    def get_feasibile_actions_for_affected(world_,
                                           affectedID=None,
                                           defaultAction='random',
                                           ActionID4Agents=[]):
        return get_feasibile_actions_for_affected_tuple(world_=world_,
                                                        affectedID=affectedID,
                                                        defaultAction=defaultAction,
                                                        ActionID4Agents=tuple(ActionID4Agents))

    @staticmethod
    def collision_checks_affected(other_world,
                                  affected=None,
                                  afffected_action_list=None,
                                  new_agent_locations_for_affected=None,
                                  old_agent_locations_for_affected=None,
                                  feasible_action_ids=None,
                                  step=None, ):

        assert step is not None
        assert affected is not None
        assert afffected_action_list is not None
        assert new_agent_locations_for_affected is not None
        assert old_agent_locations_for_affected is not None
        assert feasible_action_ids is not None

        # ..........................................#

        # Check for collisions in the NewAgentLocations
        NumberOfAgents = len(other_world.AgentList)

        actions_to_remove = []

        for action_id in feasible_action_ids:
            step_jj = (step + 1) * (len(afffected_action_list[action_id])) / (other_world.MaxSteps)
            step_jj_floor = np.floor(step_jj).astype(int)
            step_jj_ceil = np.ceil(step_jj).astype(int)

            location_jj_floor = new_agent_locations_for_affected[action_id][step_jj_floor]
            location_jj_ceil = new_agent_locations_for_affected[action_id][step_jj_ceil]
            old_location_jj = old_agent_locations_for_affected[action_id]

            for ii in np.arange(NumberOfAgents):
                step_ii = (step + 1) * (len(other_world.AgentList[ii].SelectedAction)) / (other_world.MaxSteps)
                step_ii_floor = np.floor(step_ii).astype(int)
                step_ii_ceil = np.ceil(step_ii).astype(int)

                location_ii_floor = other_world.NewAgentLocations[ii][step_ii_floor]
                location_ii_ceil = other_world.NewAgentLocations[ii][step_ii_ceil]

                old_location_ii = other_world.AgentLocations[ii]

                if (location_ii_floor == location_jj_floor) or \
                        (location_ii_ceil == location_jj_ceil):
                    actions_to_remove.append(action_id)
                    break

                elif (location_ii_floor == location_jj_ceil) and \
                        (location_ii_ceil == location_jj_floor):
                    # CROSSOVER COLLISION ! in NewLocations
                    actions_to_remove.append(action_id)
                    break

                elif location_ii_floor == location_jj_ceil:
                    # Possible Collision - if the hangovers don't match

                    overhang_floor = step_ii_ceil - step_ii
                    overhang_ceil = step_jj - step_jj_floor
                    overhang_condition = ((overhang_floor + overhang_ceil) <= 1)

                    direction_ii = np.array(location_ii_ceil) - np.array(location_ii_floor)
                    direction_jj = np.array(location_jj_ceil) - np.array(location_jj_floor)
                    same_direction_condition = np.array_equal(direction_ii, direction_jj)

                    # If the above 2 conditions are met, then the agents can slide along without hitting each other

                    if not (overhang_condition and same_direction_condition):
                        # Collision happens in the case where the above conditions are not met.
                        actions_to_remove.append(action_id)
                        break

                elif location_ii_ceil == location_jj_floor:
                    # Possible Collision - if the hangovers don't match
                    overhang_floor = step_jj_ceil - step_jj
                    overhang_ceil = step_ii - step_ii_floor
                    overhang_condition = ((overhang_floor + overhang_ceil) <= 1)

                    direction_ii = np.array(location_ii_ceil) \
                                   - np.array(location_ii_floor)
                    direction_jj = np.array(location_jj_ceil) \
                                   - np.array(location_jj_floor)
                    same_direction_condition = np.array_equal(direction_ii, direction_jj)

                    # If the above 2 conditions are met, then the agents can slide along without hitting each other

                    if not (overhang_condition and same_direction_condition):
                        # Collision happens in the case where the above conditions are not met.
                        actions_to_remove.append(action_id)
                        break

                elif (((location_ii_floor == old_location_jj) and
                       (old_location_ii == location_jj_floor)) or
                      ((location_ii_ceil == old_location_jj) and
                       (old_location_ii == location_jj_ceil)) or
                      ((location_ii_floor == old_location_jj) and
                       (old_location_ii == location_jj_ceil)) or
                      ((location_ii_ceil == old_location_jj) and
                       (old_location_ii == location_jj_floor))):
                    # CROSSOVER COLLISION ! - NewLocations with OldLocations
                    actions_to_remove.append(action_id)
                    break


        for action_id in actions_to_remove:
            feasible_action_ids.remove(action_id)
        actions_to_remove = []  # Clear actions_to_remove

        # ..........................................#
        return feasible_action_ids

    def debug_cache_behavior(self):
        """Call this to understand cache behavior"""

        # Get current cache state
        cache_info = get_feasibile_actions_for_affected_tuple.cache_info()
        print(f"\n=== Cache Debug ===")
        print(f"Hits: {cache_info.hits}, Misses: {cache_info.misses}")
        print(
            f"Hit rate: {cache_info.hits / (cache_info.hits + cache_info.misses) if cache_info.misses > 0 else 0:.2%}")

        # Test if current world would match a copy
        world_copy = copy.deepcopy(self)
        print(f"\nCurrent world hash: {hash(self)}")
        print(f"Copy hash:          {hash(world_copy)}")
        print(f"Hashes equal:       {hash(self) == hash(world_copy)}")
        print(f"Objects equal:      {self == world_copy}")

        # Check which attributes differ if not equal
        if self != world_copy:
            print("\nDifferences found:")
            for attr in ['AgentList', 'AgentActiveStatus', 'AgentCrash',
                         'RestrictedMove', 'AgentLocations', 'MaxSteps',
                         'PreviousAgentLocations', 'RestrictedPaths',
                         'WorldWalls', 'WorldOneWays']:
                if getattr(self, attr) != getattr(world_copy, attr):
                    print(f"  - {attr} differs")
                    print(f'{getattr(self, attr)}=')
                    print(f'{getattr(world_copy, attr)=}')

            if not np.array_equal(self.WorldMap, world_copy.WorldMap):
                print(f"  - WorldMap differs")
            if not np.array_equal(self.WorldState, world_copy.WorldState):
                print(f"  - WorldState differs")


@lru_cache(maxsize=None)
def get_feasibile_actions_for_affected_tuple(world_,
                                             affectedID=None,
                                             defaultAction='random',
                                             ActionID4Agents=()):
    ActionID4Agents = list(ActionID4Agents)

    assert affectedID is not None

    affected = world_.AgentList[affectedID]
    afffected_action_list = affected.Actions
    feasible_action_ids = list(range(len(afffected_action_list)))
    affected_location = world_.AgentLocations[affectedID]

    other_world = copy.deepcopy(world_)  # Make a copy of the World
    ##################################################################
    other_world.getActionSelection4UpdateGWorld(ActionID4Agents,
                                                defaultAction=defaultAction)
    ##################################################################
    other_world.RemoveAgent(agentID=affectedID)  # Remove the affected agent from it after updating actions

    other_world.PreviousAgentLocations = other_world.AgentLocations.copy()

    # Inactive Cells - NaN
    other_world.WorldState = np.where(other_world.WorldMap == 0, np.nan, other_world.WorldMap)
    # Active Cells - 0 at the start
    other_world.WorldState = np.where(other_world.WorldState == 1, 0, other_world.WorldState)

    # If nothing happens, the agents will be in the old positions
    other_world.NewAgentLocations = []
    for AgentLocation in other_world.AgentLocations:
        other_world.NewAgentLocations.append([AgentLocation])

    for idx, agent in enumerate(other_world.AgentList):
        if other_world.AgentActiveStatus[idx] is True:
            # Resetting Crash Record for all the Agents
            other_world.AgentCrash[idx] = False
            # Resetting Restricted Moves Record for all Agents
            other_world.RestrictedMove[idx] = False

    old_agent_locations_for_affected = []
    new_agent_locations_for_affected = []
    for actionID in range(len(affected.Actions)):
        old_agent_locations_for_affected.append([affected_location])
        # if nothing happens, the affected stays at its original location
        new_agent_locations_for_affected.append([affected_location])

    actions_to_remove = []

    # Getting (valid) NewLocations from action steps from Agents
    for step in np.arange(other_world.MaxSteps):
        # Updating new locations and checking validity of moves for affected

        for actionID in feasible_action_ids:
            affected_selectedAction = afffected_action_list[actionID]
            if step < len(affected_selectedAction):
                ActionStep = affected_selectedAction[step]

            else:
                ActionStep = (0, 0)  # If no steps left in SelectedAction

            old_location = new_agent_locations_for_affected[actionID][step]
            new_location = (old_location[0] + ActionStep[0], old_location[1] + ActionStep[1])

            # Making sure that the agents are not pushed off the grid
            new_location0 = np.clip(new_location[0], 0, other_world.WorldMap.shape[0] - 1)
            new_location1 = np.clip(new_location[1], 0, other_world.WorldMap.shape[1] - 1)
            # Restricted move if the new location is clipped.
            if not new_location == (new_location0, new_location1):
                actions_to_remove.append(actionID)
                continue

            AgentPath = [old_location, new_location]

            # Checking if the new position is a valid location
            if other_world.WorldState[new_location] >= 0:
                # Checking is the move is along a restricted path
                if AgentPath not in other_world.RestrictedPaths:
                    new_agent_locations_for_affected[actionID].append(new_location)

                else:  # Setting OldLocation in case of Restricted Paths
                    actions_to_remove.append(actionID)
                    continue

            else:  # Setting OldLocation in case of Invalid Location
                actions_to_remove.append(actionID)
                continue

        # Remove actions with collisions
        for action_id in actions_to_remove:
            feasible_action_ids.remove(action_id)
        actions_to_remove = []  # Clear actions_to_remove
        # print(f'Restricted:{feasible_action_ids=}')

        ########################################################################################################################

        # Updating new locations and checking validity of moves for other_world
        NewAgentLocations_CurrentFloor = other_world.AgentLocations.copy()

        for idx, agent in enumerate(other_world.AgentList):
            if (other_world.AgentActiveStatus[idx] is True):
                if step < len(agent.SelectedAction) and (other_world.AgentCrash[idx] is False):
                    ActionStep = agent.SelectedAction[step]

                else:
                    ActionStep = (0, 0)  # If no steps left in SelectedAction

                old_location = other_world.NewAgentLocations[idx][step]
                new_location = (old_location[0] + ActionStep[0], old_location[1] + ActionStep[1])

                # Making sure that the agents are not pushed off the grid
                new_location0 = np.clip(new_location[0], 0, other_world.WorldMap.shape[0] - 1)
                new_location1 = np.clip(new_location[1], 0, other_world.WorldMap.shape[1] - 1)
                # Restricted move if the new location is clipped.
                if not new_location == (new_location0, new_location1):
                    other_world.RestrictedMove[idx] = True
                    new_location = (new_location0, new_location1)

                AgentPath = [old_location, new_location]

                # Checking if the new position is a valid location
                if other_world.WorldState[new_location] >= 0:
                    # Checking is the move is along a restricted path
                    if AgentPath not in other_world.RestrictedPaths:
                        other_world.NewAgentLocations[idx].append(new_location)

                    else:  # Setting OldLocation in case of Restricted Paths
                        other_world.NewAgentLocations[idx].append(old_location)
                        # Record Restricted Move
                        other_world.RestrictedMove[idx] = True
                        # Print NewAgentLocation

                else:  # Setting OldLocation in case of Invalid Location
                    other_world.NewAgentLocations[idx].append(old_location)
                    # Record Restricted Move
                    other_world.RestrictedMove[idx] = True
                    # Print NewAgentLocation

                    # Updating the NewAgentLocation when it is valid

        #############################################
        # Check feasibility of actions and update list of feasible actions
        feasible_action_ids = world_.collision_checks_affected(other_world=other_world,
                                                             step=step,
                                                             affected=affected,
                                                             afffected_action_list=afffected_action_list,
                                                             new_agent_locations_for_affected=
                                                             new_agent_locations_for_affected,
                                                             old_agent_locations_for_affected=
                                                             old_agent_locations_for_affected,
                                                             feasible_action_ids=feasible_action_ids,
                                                             )
        # print(f'Collision:{feasible_action_ids=}')
        #############################################
        # Check and resolve collisions for other_world

        NewAgentLocations_CurrentFloor = \
            other_world.collision_checks_and_resolution(step=step,
                                                        NewAgentLocations_CurrentFloor=NewAgentLocations_CurrentFloor,
                                                        VerboseFlag=VerboseFlag)
        #############################################

        other_world.update_agent_locations_2_world_state(
            NewAgentLocations_CurrentFloor= NewAgentLocations_CurrentFloor)

        # Update old locations for affected agent at the end of a step
        for actionID in feasible_action_ids:
            old_agent_locations_for_affected[actionID].append(new_agent_locations_for_affected[actionID][step])

        #############################################

    valid_moves_count = len(feasible_action_ids)
    validity_of_moves_of_affected = np.zeros(len(afffected_action_list))
    for actionID in feasible_action_ids:
        validity_of_moves_of_affected[actionID] = 1

    return valid_moves_count, validity_of_moves_of_affected

def LoadJsonScenario(json_filename='Scenarios.json', scenario_name='Base'):
    # Reading Dictionary from JSON file

    with open(json_filename) as json_file:
        data = json.load(json_file)

    Scenario = data[scenario_name]

    if 'MdRs' in Scenario.keys():
        for mdr_key in Scenario['MdRs']:
            mdr = Scenario['MdRs'][mdr_key]

            for slice_key in ['slicex', 'slicey']:
                slice_arg = mdr[slice_key]
                if slice_arg[2] == 0:  # If slice_step is 0
                    slice_ = slice(slice_arg[0], slice_arg[1], None)
                else:
                    slice_ = slice(slice_arg[0], slice_arg[1], slice_arg[2])
                Scenario['MdRs'][mdr_key][slice_key] = slice_

    if 'Policies' in Scenario.keys():
        for policy_key in Scenario['Policies']:
            policy = Scenario['Policies'][policy_key]

            for slice_key in ['slicex', 'slicey']:
                slice_arg = policy[slice_key]
                if slice_arg[2] == 0:  # If slice_step is 0
                    slice_ = slice(slice_arg[0], slice_arg[1], None)
                else:
                    slice_ = slice(slice_arg[0], slice_arg[1], slice_arg[2])
                Scenario['Policies'][policy_key][slice_key] = slice_

    return Scenario


def AddJsonScenario(json_filename='Scenarios.json', new_scenario=None, new_scenario_name=None, Overwrite=False):
    # Appending a new Scenario to the JSON file

    # Reading Dictionary from JSON file
    with open(json_filename) as json_file:
        data = json.load(json_file)

    # Adding new scenario
    if new_scenario is None:
        print('Error ! - No New Scenario Added')
        return False
    elif new_scenario_name is None:
        print("Error - No Scenario Name - Aborted")
        return False

    # Convert slice objects back to lists for JSON serialization
    if 'MdRs' in new_scenario:
        for mdr_key in new_scenario['MdRs']:
            mdr = new_scenario['MdRs'][mdr_key]

            for slice_key in ['slicex', 'slicey']:
                if isinstance(mdr[slice_key], slice):
                    slice_obj = mdr[slice_key]
                    slice_step = slice_obj.step if slice_obj.step is not None else 0
                    new_scenario['MdRs'][mdr_key][slice_key] = [slice_obj.start, slice_obj.stop, slice_step]

    if 'Policies' in new_scenario:
        for policy_key in new_scenario['Policies']:
            policy = new_scenario['Policies'][policy_key]

            for slice_key in ['slicex', 'slicey']:
                if isinstance(policy[slice_key], slice):
                    slice_obj = policy[slice_key]
                    slice_step = slice_obj.step if slice_obj.step is not None else 0
                    new_scenario['Policies'][policy_key][slice_key] = [slice_obj.start, slice_obj.stop, slice_step]

    if new_scenario_name in data:
        if Overwrite is True:
            print('Overwriting Scenario: ', new_scenario_name)
            data[new_scenario_name] = new_scenario
        else:
            print('Error - Scenario Exists - Overwrite is False !')
            return False
    else:
        data[new_scenario_name] = new_scenario

    pretty_print_json = pprint.pformat(data, width=150).replace("'", '"')

    with open(json_filename, 'w') as f:
        f.write(pretty_print_json)

    return True


def SwapActionIDs4Agents(ActionID4Agents=None, agentIDs4swaps=[], actionIDs4swaps=[]):
    if ActionID4Agents is None:
        print('No ActionID4Agents Provided! Returning empty list!')
        return []
    elif not (len(agentIDs4swaps) == len(actionIDs4swaps)):
        print('The lengths of agentIDs4swaps and actionIDs4swaps should match ! No Swaps Done!')
        return ActionID4Agents
    else:
        new_ActionIDs4Agents = []
        for agentID, actionID in ActionID4Agents:
            if agentID in agentIDs4swaps:
                idx = agentIDs4swaps.index(agentID)
                new_actionID = actionIDs4swaps[idx]
                new_ActionIDs4Agents.append((int(agentID), int(new_actionID)))
            else:
                new_ActionIDs4Agents.append((int(agentID), int(actionID)))

    return new_ActionIDs4Agents


def agent_crash_report(agent_crashes=None, restricted_moves=None):
    assert agent_crashes is not None
    assert restricted_moves is not None

    n_agents = len(agent_crashes)

    crashed_agents = [ ii for ii in range(n_agents) if (agent_crashes[ii] or restricted_moves[ii]) ]
    print('Crashed agents: ', [ ii+1 for ii in crashed_agents ])


