from __future__ import print_function, division, absolute_import
from core.clause import *


class SymbolicEnvironment(object):
    def __init__(self, background, initial_state):
        '''
        :param language_frame
        :param background: list of atoms, the background knowledge
        :param positive: list of atoms, positive instances
        :param negative: list of atoms, negative instances
        '''
        self.background = background
        self.state = initial_state
        self.initial_state = initial_state

class LanguageFrame(object):
    def __init__(self, actions, extensional, constants):
        '''
        :param target: string, target predicate
        :param extensional: list of Predicates, extensional predicates and their arity
        :param constants: list of strings, constants
        '''
        self.target = actions
        self.extensional = extensional
        self.constants = constants

UP = Predicate("up",2)
DOWN = Predicate("down",2)
LEFT = Predicate("left",2)
RIGHT = Predicate("right",2)
LESS = Predicate("less",2)
CLIFF = Predicate("cliff",2)
GOAL = Predicate("goal",2)
WIDTH = 5
actions = [UP, DOWN, LEFT, RIGHT]
language = LanguageFrame(actions, extensional=[LESS, CLIFF, GOAL],
                         constants=[str(i) for i in range(WIDTH)])

class CliffWalking(SymbolicEnvironment):
    def __init__(self):
        background = []
        background.append(Atom(GOAL, [str(WIDTH-1), "0"]))
        background.extend([Atom(str(x), "0") for x in range(1, WIDTH)])
        background.extend([Atom(str(i), str(j)) for i in range(0, WIDTH)
                           for j in range(0, WIDTH) if i<j])
        super(CliffWalking, self).__init__(background, ("0","0"))
        self.acc_reward = 0

    def reset(self):
        self.acc_reward = 0
        self.state = self.initial_state

    def step(self, action):
        x = int(self.state[0])
        y = int(self.state[1])
        reward, finished = self.get_reward(action)
        self.acc_reward += reward
        if action.predicate == UP:
            self.state = (str(x), str(y+1))
        elif action.predicate == DOWN:
            self.state = (str(x), str(y-1))
        elif action.predicate == LEFT:
            self.state = (str(x-1), str(y))
        elif action.predicate == RIGHT:
            self.state = (str(x+1), str(y))
        else:
            raise ValueError()
        return reward, finished

    def get_reward(self, action):
        """
        :param action: action atom
        :return: reward value, and whether an episode is finished
        """
        for atom in self.background:
            if atom.predicate == GOAL and tuple(atom.terms) == self.state:
                return 10.0, True
            elif atom.predicate == CLIFF and tuple(atom.terms) == self.state:
                return -100.0, True
            else:
                return 0, False


