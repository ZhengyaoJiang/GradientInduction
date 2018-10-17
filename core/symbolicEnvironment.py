from __future__ import print_function, division, absolute_import
from core.clause import *
from core.ilp import LanguageFrame


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


UP = Predicate("up",2)
DOWN = Predicate("down",2)
LEFT = Predicate("left",2)
RIGHT = Predicate("right",2)
LESS = Predicate("less",2)
ZERO = Predicate("zero",1)
CLIFF = Predicate("cliff",2)
GOAL = Predicate("goal",2)
WIDTH = 5

class CliffWalking(SymbolicEnvironment):
    def __init__(self):
        actions = [UP, DOWN, LEFT, RIGHT]
        self.language = LanguageFrame(actions, extensional=[LESS, CLIFF, GOAL, ZERO],
                                      constants=[str(i) for i in range(WIDTH)])
        background = []
        background.append(Atom(GOAL, [str(WIDTH-1), "0"]))
        background.extend([Atom(CLIFF, [str(x), "0"]) for x in range(1, WIDTH-1)])
        background.extend([Atom(LESS, [str(i), str(j)]) for i in range(0, WIDTH)
                           for j in range(0, WIDTH) if i<j])
        background.append(Atom(ZERO, ["0"]))
        super(CliffWalking, self).__init__(background, ("0","0"))
        self.acc_reward = 0
        self.actions = actions

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2symbol(self, action_index):
        return self.actions[action_index]

    def reset(self):
        self.acc_reward = 0
        self.state = self.initial_state

    def step(self, action):
        x = int(self.state[0])
        y = int(self.state[1])
        reward, finished = self.get_reward(action)
        self.acc_reward += reward
        if action == UP and y<WIDTH-1:
            self.state = (str(x), str(y+1))
        elif action == DOWN and y>0:
            self.state = (str(x), str(y-1))
        elif action == LEFT and x>0:
            self.state = (str(x-1), str(y))
        elif action == RIGHT and x<WIDTH-1:
            self.state = (str(x+1), str(y))
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
                return -10.0, True
        return -0.1, False


