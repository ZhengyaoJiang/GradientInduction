from __future__ import print_function, division, absolute_import
from core.clause import *
from core.ilp import LanguageFrame
import copy
from random import shuffle, choice


class SymbolicEnvironment(object):
    def __init__(self, background, initial_state, actions):
        '''
        :param language_frame
        :param background: list of atoms, the background knowledge
        :param positive: list of atoms, positive instances
        :param negative: list of atoms, negative instances
        '''
        self.background = background
        self._state = initial_state
        self.initial_state = copy.deepcopy(initial_state)
        self.actions = actions
        self.acc_reward = 0
        self.step = 0

    def reset(self):
        self.acc_reward = 0
        self.step = 0
        self._state = copy.deepcopy(self.initial_state)


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
        self.state_dim = 2
        self.all_actions = actions

    @property
    def all_states(self):
        return [(str(i), str(j)) for i in range(WIDTH) for j in range(WIDTH)]

    def state2vector(self, state):
        return np.array([float(state[0]), float(state[1])])

    @property
    def state(self):
        return copy.deepcopy(self._state)

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
        x = int(self._state[0])
        y = int(self._state[1])
        self.step+=1
        reward, finished = self.get_reward()
        self.acc_reward += reward
        if action == UP and y<WIDTH-1:
            self._state = (str(x), str(y+1))
        elif action == DOWN and y>0:
            self._state = (str(x), str(y-1))
        elif action == LEFT and x>0:
            self._state = (str(x-1), str(y))
        elif action == RIGHT and x<WIDTH-1:
            self._state = (str(x+1), str(y))
        return reward, finished

    def get_reward(self):
        """
        :param action: action atom
        :return: reward value, and whether an episode is finished
        """
        for atom in self.background+self.unseen_background:
            if atom.predicate == GOAL and tuple(atom.terms) == self._state:
                return 10.0, True
            elif atom.predicate == CLIFF and tuple(atom.terms) == self._state:
                return -10.0, True
            elif self.step>=self.max_step:
                return -10.0, True
        return -0.1, False

ON = Predicate("on", 2)
CLEAR = Predicate("clear", 1)
MOVE = Predicate("move", 2)
BLOCK_N = 6
INI_STATE = [["a", "b", "c", "d", "e", "f"]]
INI_STATE2 = [["a"], ["b"], ["c"], ["d"]]

import string
class BlockWorld(SymbolicEnvironment):
    """
    state is represented as a list of lists
    """
    def __init__(self, initial_state=INI_STATE, additional_predicates=(), background=()):
        actions = [MOVE]
        self.language = LanguageFrame(actions, extensional=[ON, CLEAR]+list(additional_predicates),
                                      constants=sum(initial_state, [])+["floor"])
        super(BlockWorld, self).__init__(list(background), initial_state, actions)
        self.max_step = 50
        self._block_encoding = {"a":1, "b": 2, "c":3, "d":4, "e": 5, "f":6}
        self.state_dim = BLOCK_N**3
        self._all_blocks = list(string.ascii_lowercase)[:BLOCK_N]+["floor"]

    def clean_empty_stack(self):
        self._state = [stack for stack in self._state if stack]

    @property
    def all_actions(self):
        return [Atom(MOVE, [a, b]) for a in self._all_blocks for b in self._all_blocks]

    @property
    def state(self):
        return tuple([tuple(stack) for stack in self._state])

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
        if finished and reward<1:
            self._state = [[]]
            return reward, finished
        for stack1 in self._state:
            if stack1[-1] == block1:
                for stack2 in self._state:
                    if stack2[-1] == block2:
                        del stack1[-1]
                        stack2.append(block1)
                        return reward, finished
        if block2 == "floor":
            for stack1 in self._state:
                if stack1[-1] == block1 and len(stack1)>1:
                    del stack1[-1]
                    self._state.append([block1])
                    return reward, finished
        return reward, finished

    @property
    def action_n(self):
        return (BLOCK_N+1)**2

    def state2vector(self, state):
        matrix = np.zeros([BLOCK_N, BLOCK_N, BLOCK_N])
        for i, stack in enumerate(state):
            for j, block in enumerate(stack):
                matrix[i][j][self._block_encoding[block]-1] = 1.0
        return matrix.flatten()

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
            return -0.0, True
        for stack in self._state:
            if len(stack) > 1:
                return -0.02, False
        return 1.0, True

class Stack(BlockWorld):
    def get_reward(self):
        if self.step >= self.max_step:
            return -0.0, True
        for stack in self._state:
            if len(stack) == BLOCK_N:
                return 1.0, True
        return -0.02, False

GOAL_ON = Predicate("goal_on", 2)
class On(BlockWorld):
    def __init__(self, initial_state=INI_STATE, goal_state=Atom(GOAL_ON, ["a", "b"])):
        super(On, self).__init__(initial_state, additional_predicates=[GOAL_ON], background=[goal_state])
        self.goal_state = goal_state

    def get_reward(self):
        if self.step >= self.max_step:
            return 0.0, True
        if Atom(ON, self.goal_state.terms) in self.state2atoms(self._state):
            return 1.0, True
        return -0.02, False

def random_initial_state():
    result = [[] for _ in range(BLOCK_N)]
    all_entities = ["a", "b", "c", "d", "e", "f", "g"][:BLOCK_N]
    shuffle(all_entities)
    for entity in all_entities:
        stack_id = np.random.randint(0, BLOCK_N)
        result[stack_id].append(entity)
    return result


PLACE = Predicate("place", 2)
MINE = Predicate("mine", 2)
EMPTY = Predicate("empty", 2)
OPPONENT = Predicate("opponent", 2)
class TicTacTeo(SymbolicEnvironment):
    def __init__(self, width=3, know_valid_pos=True):
        actions = [PLACE]
        self.language = LanguageFrame(actions, extensional=[ZERO, MINE, EMPTY, OPPONENT, SUCC],
                                      constants=[str(i) for i in range(width)])
        background = []
        #background.extend([Atom(LESS, [str(i), str(j)]) for i in range(0, WIDTH)
        #                   for j in range(0, WIDTH) if i < j])
        background.extend([Atom(SUCC, [str(i), str(i + 1)]) for i in range(width - 1)])
        background.append(Atom(ZERO, ["0"]))
        self.max_step = 50
        initial_state = np.zeros([3,3])
        super(TicTacTeo, self).__init__(background, initial_state, actions)
        self.width = width
        self.all_positions = [(i, j) for i in range(width) for j in range(width)]
        self.know_valid_pos = know_valid_pos
        self.action_n = len(self.all_positions)
        self.state_dim = width**2

    def next_step(self, action):
        def tuple2int(t):
            return (int(t[0]), int(t[1]))
        self.step += 1
        reward, finished = self.get_reward()
        if finished:
            return reward, finished
        valids = self.get_valid()
        if tuple2int(action.terms) in valids:
            self._state[tuple2int(action.terms)] = 1
        self.random_move(self.know_valid_pos)
        return reward, finished

    def get_valid(self):
        return [(x,y) for x,y in self.all_positions if self._state[x,y]==0]

    @property
    def all_actions(self):
        return [Atom(PLACE, [str(position[0]), str(position[1])]) for position in self.all_positions]

    def state2vector(self, state):
        return state.flatten()

    def state2atoms(self, state):
        atoms = set()
        def tuple2strings(t):
            return str(t[0]), str(t[1])
        for position in self.all_positions:
            if state[position] == 0:
                atoms.add(Atom(EMPTY, tuple2strings(position)))
            elif state[position] == -1:
                atoms.add(Atom(OPPONENT, tuple2strings(position)))
            elif state[position] == 1:
                atoms.add(Atom(MINE, tuple2strings(position)))
        return atoms

    @property
    def state(self):
        return copy.deepcopy(self._state)

    def random_move(self, know_valid):
        valid_position = self.get_valid()
        if not valid_position:
            return
        if know_valid:
            position = choice(valid_position)
            self._state[position] = -1
        else:
            position = choice(self.all_positions)
            if position in valid_position:
                self._state[position] = -1

    def get_reward(self):
        if np.any(np.sum(self._state, axis=0)==3) or np.any(np.sum(self._state, axis=1)==3):
            return 1, True
        for i in range(-self.width, self.width):
            if np.trace(self._state, i)==3 or np.trace(np.flip(self._state, 0),i)==3:
                return 1, True
        if np.any(np.sum(self._state, axis=0)==-3) or np.any(np.sum(self._state, axis=1)==-3):
            return -1, True
        for i in range(-self.width, self.width):
            if np.trace(self._state, i)==-3 or np.trace(np.flip(self._state, 0),i)==-3:
                return -1, True
        if not self.get_valid():
            return 0, True
        return 0, False

