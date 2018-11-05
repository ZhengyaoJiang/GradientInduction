from __future__ import print_function, division, absolute_import
from core.clause import *
from core.ilp import LanguageFrame
import copy


class SymbolicEnvironment(object):
    def __init__(self, background, initial_state, actions):
        '''
        :param language_frame
        :param background: list of atoms, the background knowledge
        :param positive: list of atoms, positive instances
        :param negative: list of atoms, negative instances
        '''
        self.background = background
        self.state = initial_state
        self.initial_state = copy.deepcopy(initial_state)
        self.actions = actions
        self.acc_reward = 0
        self.step = 0

    def reset(self):
        self.acc_reward = 0
        self.step = 0
        self.state = copy.deepcopy(self.initial_state)


UP = Predicate("up",2)
DOWN = Predicate("down",2)
LEFT = Predicate("left",2)
RIGHT = Predicate("right",2)
LESS = Predicate("less",2)
ZERO = Predicate("zero",1)
CLIFF = Predicate("cliff",2)
SUCC = Predicate("succ",2)
GOAL = Predicate("goal",2)
WIDTH = 5

class CliffWalking(SymbolicEnvironment):
    def __init__(self):
        actions = [UP, DOWN, LEFT, RIGHT]
        self.language = LanguageFrame(actions, extensional=[LESS, ZERO, SUCC],
                                      constants=[str(i) for i in range(WIDTH)])
        background = []
        self.unseen_background = []
        self.unseen_background.append(Atom(GOAL, [str(WIDTH-1), "0"]))
        self.unseen_background.extend([Atom(CLIFF, [str(x), "0"]) for x in range(1, WIDTH-1)])
        #background.append(Atom(GOAL, [str(WIDTH-1), "0"]))
        #background.extend([Atom(CLIFF, [str(x), "0"]) for x in range(1, WIDTH-1)])
        background.extend([Atom(LESS, [str(i), str(j)]) for i in range(0, WIDTH)
                           for j in range(0, WIDTH) if i<j])
        background.extend([Atom(SUCC, [str(i), str(i+1)]) for i in range(WIDTH-1)])
        background.append(Atom(ZERO, ["0"]))
        #background.extend([Atom(CLIFF, ["1", str(y)]) for y in range(2, WIDTH)])
        #background.extend([Atom(CLIFF, ["3", str(y)]) for y in range(1, WIDTH-1)])
        super(CliffWalking, self).__init__(background, ("0","0"), actions)
        self.max_step = 50

    @property
    def all_states(self):
        return [(str(i), str(j)) for i in range(WIDTH) for j in range(WIDTH)]

    def action_vec2symbol(self, action_vec):
        """
        :param action_vec: one-hot action vector (numpy array)
        :return:
        """
        return self.actions[np.argmax(action_vec)[0]]

    def action_index2symbol(self, action_index):
        return self.actions[action_index]

    @property
    def action_n(self):
        return len(self.actions)

    def next_step(self, action):
        x = int(self.state[0])
        y = int(self.state[1])
        self.step+=1
        reward, finished = self.get_reward()
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

    def get_reward(self):
        """
        :param action: action atom
        :return: reward value, and whether an episode is finished
        """
        for atom in self.background+self.unseen_background:
            if atom.predicate == GOAL and tuple(atom.terms) == self.state:
                return 10.0, True
            elif atom.predicate == CLIFF and tuple(atom.terms) == self.state:
                return -10.0, True
            elif self.step>=self.max_step:
                return -10.0, True
        return -0.1, False

ON = Predicate("on", 2)
CLEAR = Predicate("clear", 1)
MOVE = Predicate("move", 2)
BLOCK_N = 4
INI_STATE = [["a", "b", "c", "d"]]
INI_STATE2 = [["a"], ["b"], ["c"], ["d"]]

class BlockWorld(SymbolicEnvironment):
    """
    state is represented as a list of lists
    """
    def __init__(self, initial_state=INI_STATE):
        actions = [MOVE]
        self.language = LanguageFrame(actions, extensional=[ON, CLEAR],
                                      constants=sum(initial_state, [])+["floor"])
        background = []
        super(BlockWorld, self).__init__(background, initial_state, actions)
        self.max_step = 50

    def clean_empty_stack(self):
        self.state = [stack for stack in self.state if stack]

    def next_step(self, action):
        """
        :param action: action is a ground atom
        :return:
        """

        self.step+=1
        reward, finished = self.get_reward()
        self.acc_reward += reward

        self.clean_empty_stack()
        block1, block2 = action.terms
        for stack1 in self.state:
            if stack1[-1] == block1:
                for stack2 in self.state:
                    if stack2[-1] == block2:
                        del stack1[-1]
                        stack2.append(block1)
                        return reward, finished
        if block2 == "floor":
            for stack1 in self.state:
                if stack1[-1] == block1 and len(stack1)>1:
                    del stack1[-1]
                    self.state.append([block1])
                    return reward, finished
        return reward, finished

    @property
    def action_n(self):
        return (BLOCK_N+1)**2

    def state2atoms(self, state):
        atoms = set()
        for stack in state:
            if len(stack)>0:
                atoms.add(Atom(ON, [stack[0], "floor"]))
                atoms.add(Atom(CLEAR, [stack[-1]]))
            for i in range(len(stack)-1):
                atoms.add(Atom(ON, [stack[i+1], stack[i]]))
        atoms.add(Atom(CLEAR, ["floor"]))
        return atoms

    def get_reward(self):
        pass

class Unstack(BlockWorld):
    def get_reward(self):
        if self.step >= self.max_step:
            return -1.0, True
        for stack in self.state:
            if len(stack) > 1:
                return -0.05, False
        return 1.0, True

class Stack(BlockWorld):
    def get_reward(self):
        if self.step >= self.max_step:
            return -0.0, True
        for stack in self.state:
            if len(stack) == BLOCK_N:
                return 1.0, True
        return -0.02, False
