import numpy as np; np.random.seed(0)

VerboseFlag = False


class Agent:
    def __init__(self):

        ActionNames, ActionMoves = DefineActions()

        self.ActionNames = ActionNames
        self.Actions = ActionMoves

        self.SelectedAction = (0, 0)
        self.SelectedActionName = None

        self.ActionPolicy = None
        # self.SensorStates = None

    def __eq__(self, other):
        if not isinstance(other, Agent):
            return NotImplemented
        return (
                self.Actions == other.Actions and
                self.SelectedAction == other.SelectedAction and
                np.array_equal(self.ActionPolicy,other.ActionPolicy)
                )

    def __hash__(self):
        return hash((
            tuple(tuple(action) for action in self.Actions),  # Convert each inner list to tuple
            tuple(self.SelectedAction),
            self.ActionPolicy.tobytes()
        ))

    def UpdateActionPolicy(self, NewPolicy):
        if len(NewPolicy) == len(self.Actions):
            self.ActionPolicy = NewPolicy
        else:
            print('Policy Not Updated ! Wrong Dimensions. \n'
                  ' Policy must have the same shape as Actions')

    def ActionSelection(self, ActionID=-1):
        if ActionID >= 0 and ActionID < len(self.Actions):  # Select the Given Action
            SelectedActionID = ActionID
        else:  # Select Random Action (according to Policy)
            SelectedActionID = np.random.choice(np.arange(len(self.Actions)), p=self.ActionPolicy)
        if VerboseFlag: print('ActionID,SelectedActionID', ActionID, SelectedActionID)
        self.SelectedAction = self.Actions[SelectedActionID]
        self.SelectedActionName = self.ActionNames[SelectedActionID]

        return SelectedActionID

# -----------------------------------------------------#


def DefineActions():
    # Defining Actions - for all agents
    # Each agent would be having actions which are a subset of all the actions

    # ---------------------------------------------------------------
    # Defining Individual Actions as Dictionaries
    # ... so that the names and moves are easily verifiable

    Action0 = {
        'Name': 'Stay',
        'Move': [(0, 0)]
    }
    Action1 = {
        'Name': 'Up1',
        'Move': [(-1, 0)]
    }
    Action2 = {
        'Name': 'Down1',
        'Move': [(1, 0)]
    }
    Action3 = {
        'Name': 'Left1',
        'Move': [(0, -1)]
    }
    Action4 = {
        'Name': 'Right1',
        'Move': [(0, 1)]
    }

    # ---------------------------#

    Action5 = {
        'Name': 'Up2',
        'Move': [(-1, 0), (-1, 0)]
    }
    Action6 = {
        'Name': 'Down2',
        'Move': [(1, 0), (1, 0)]
    }
    Action7 = {
        'Name': 'Left2',
        'Move': [(0, -1), (0, -1)]
    }
    Action8 = {
        'Name': 'Right2',
        'Move': [(0, 1), (0, 1)]
    }

    # ---------------------------#

    Action9 = {
        'Name': 'Up3',
        'Move': [(-1, 0), (-1, 0), (-1, 0)]
    }
    Action10 = {
        'Name': 'Down3',
        'Move': [(1, 0), (1, 0), (1, 0)]
    }
    Action11 = {
        'Name': 'Left3',
        'Move': [(0, -1), (0, -1), (0, -1)]
    }
    Action12 = {
        'Name': 'Right3',
        'Move': [(0, 1), (0, 1), (0, 1)]
    }

    # ---------------------------#

    Action13 = {
        'Name': 'Up4',
        'Move': [(-1, 0), (-1, 0), (-1, 0), (-1, 0)]
    }
    Action14 = {
        'Name': 'Down4',
        'Move': [(1, 0), (1, 0), (1, 0), (1, 0)]
    }
    Action15 = {
        'Name': 'Left4',
        'Move': [(0, -1), (0, -1), (0, -1), (0, -1)]
    }
    Action16 = {
        'Name': 'Right4',
        'Move': [(0, 1), (0, 1), (0, 1), (0, 1)]
    }

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------

    # Collecting all the actions into a dictionary of Actions
    # Actions = {
    #     0 : Action0,
    #     1 : Action1,
    #     2 : Action2,
    #     3 : Action3,
    #     4 : Action4

    # }

    Actions = {
        0: Action0,
        1: Action1,
        2: Action2,
        3: Action3,
        4: Action4,
        5: Action5,
        6: Action6,
        7: Action7,
        8: Action8,
        9: Action9,
        10: Action10,
        11: Action11,
        12: Action12,
        13: Action13,
        14: Action14,
        15: Action15,
        16: Action16

    }

    # Separating out the Action Names and Moves
    ActionNames = []
    ActionMoves = []

    for _, Action in Actions.items():
        # print('Name :', Action['Name'], ', Move :', Action['Move'])
        ActionNames.append(Action['Name'])
        ActionMoves.append(Action['Move'])

    # print('ActionNames : ', ActionNames)
    # print('ActionMoves : ', ActionMoves)

    # for Move in ActionMoves:
    #     for step in np.arange(2):
    #         if step < len(Move):
    #             print('Step', step, ':', Move, Move[step])

    return ActionNames, ActionMoves


def GeneratePolicy(StepWeights = None, DirectionWeights=None, policy_weights=None):
    """
    Function to get normalised policy weights for selecting actions

    :param StepWeights: list
    :param DirectionWeights: list
    :param policy_weights: list of lists
    :return: Policy
    """
    # If policy weights are given, use them directly
    if policy_weights:
        p = []
        for p_w in policy_weights:
            p = p + p_w

    # Use step weights and direction weights otherwise
    else:
        if StepWeights is None:
            StepWeights = [5, 4, 3, 2, 1]
        if DirectionWeights is None:
            DirectionWeights = [1, 1, 1, 1]

        p = [StepWeights[0] * 4]  # Only one action has 0 steps - 'Stay'
        # The factor of four is because other actions have 4 directions

        for sw in StepWeights[1:]:
            # Multiply stepweight with directionweight
            p = p + [sw * x for x in DirectionWeights]

    p = np.array(p)  # Converting to numpy array
    if p.sum() != 0:
        p = p / p.sum()  # Normalising so that sum of probabilities is 1

    if VerboseFlag: print('P : ', p, 'Sum: ', p.sum())

    return p

