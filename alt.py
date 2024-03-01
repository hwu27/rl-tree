import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class AltruismEnvironment(py_environment.PyEnvironment):

    def __init__(self, size, max_actions):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=0, maximum=4, name='observation')
        self._tree = tree_obj(type='monster', position=np.array([2, 2]), color=[0, 0, 255])
        self._state = np.array([size-1, 0]) # start position
        self._last_terminated_position = [] # last position of an episode
        self._last_terminated_number_actions = []
        self._size = size # grid size n x n
        self._goal= [size - 1, size - 1]
        self._max_actions = max_actions
        self._actions_count = 0

    def last_terminated_position(self):
        return self._last_terminated_position

    def last_terminated_number_actions(self):
        return self._last_terminated_number_actions

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._actions_count = 0
        self._state = np.array([self._size-1, 0])  # reset to start position
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        if self._actions_count == self._max_actions-1:
            self._last_terminated_number_actions.append(self._actions_count)
            # print("Ran out of moves.")
            return self.reset() 
        # define movement actions
        if action == 0:   # up
            self._state[0] = max(0, self._state[0] - 1)
        elif action == 1: # down
            self._state[0] = min(self._size - 1, self._state[0] + 1)
        elif action == 2: # left
            self._state[1] = max(0, self._state[1] - 1)
        elif action == 3: # right
            self._state[1] = min(self._size - 1, self._state[1] + 1)

        self._actions_count += 1

        next_to = np.abs(self._state - self._tree.position) 
        # checker for if the agent is next to the tree
        next_to_tree = np.all(next_to <= 1) and np.any(next_to == 1)

        next_to_occupied = False
        # check if the cell was already occupied    
        if (len(self._last_terminated_position) != 0):
            next_to_occupied = any(np.array_equal(pos, self._state) for pos in self._last_terminated_position)

        if (next_to_tree and next_to_occupied):
            #print("Occupied!")
            self._last_terminated_number_actions.append(self._actions_count)# save last terminate action count
            return ts.termination(np.array(self._state, dtype=np.int32), reward=-15)
        # if it you are next to the tree and has not been occupied
        elif(next_to_tree):
            #print("You have taken a spot!") 
            self._last_terminated_position.append(self._state) # save last terminated positon
            self._last_terminated_number_actions.append(self._actions_count) # save last terminate action count
            return ts.termination(np.array(self._state, dtype=np.int32), reward=20)

        return ts.transition(np.array(self._state, dtype=np.int32), reward=-1)
        
    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            grid = np.zeros((self._size, self._size, 3), dtype=np.uint8)  # RGB canvas

            # define colors
            agent_color = [255, 0, 0]  # red
            goal_color = [0, 255, 0]   # green
            occupied_color = [0, 0, 0]
            background_color = [255, 255, 255]  # white

            # fill the background with white
            grid = np.full((self._size, self._size, 3), background_color, dtype=np.uint8)

            # goal's position
            grid[self._goal[0], self._goal[1], :] = goal_color

            # tree position
            tree_pos = self._tree.position
            grid[tree_pos[0], tree_pos[1]] = self._tree.color

            # last terminated position
            for pos in self._last_terminated_position:
                grid[pos[0], pos[1], :] = occupied_color

            # agent's position
            grid[self._state[0], self._state[1], :] = agent_color

            return grid
        else:
            raise NotImplementedError("Render mode {} is not implemented.".format(mode))

class tree_obj:
    def __init__(self, type, position, color):
        self.type = type # type of tree, aka neutral or monster
        self.position = position # position on board
        self.color = color # color of the specific type of tree
