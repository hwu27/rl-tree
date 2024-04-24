from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import extra.helper as helper
import copy
import collections

class ParkingEnvironment(py_environment.PyEnvironment):

    def __init__(self, size, max_actions):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,),  # might have to consider the occupied positions as a possible observation
            dtype=np.int32,
            minimum=[0, 0], 
            maximum=[size - 1, size - 1],  
            name='observation')
        self._size = size # grid size n x n
        # y and then x
        self._state = np.array([size-1, 0], dtype=np.int32) # start position
        self._tree_position = np.array([size/2, size/2], dtype=np.int32) # tree position
        self._max_actions = max_actions
        self._actions_count = 1 
        self._current_time_step = None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._current_time_step = ts.restart(np.array(self._state, dtype=np.int32))
        self._actions_count = 1
        self._state = np.array([self._size-1, 0], dtype=np.int32)  # reset to start position
        return self._current_time_step  
    
    # reward (0) takes a step, but also punishes for going to the edge
    def _make_step(self, action):
        # define movement actions

        if action == 0:   # up
            # check if the agent is going to the edge and punish accordingly using ts.transition
            self._state[0] = max(0, self._state[0] - 1)
        elif action == 1: # down
            self._state[0] = min(self._size - 1, self._state[0] + 1)
        elif action == 2: # left
            self._state[1] = max(0, self._state[1] - 1)
        elif action == 3: # right
            self._state[1] = min(self._size - 1, self._state[1] + 1)

    def _step(self, action):

        # -------------------------------------- (0) step --------------------------------------
        
        # make a step based off the action
        self._make_step(action)
        
    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            grid = np.zeros((self._size, self._size, 3), dtype=np.uint8)  # RGB canvas

            # define colors
            agent_color = [255, 0, 0]  # red
            occupied_color = [0, 0, 0]
            tree_color = [0, 255, 0]  # green
            background_color = [255, 255, 255]  # white

            # fill the background with white
            grid = np.full((self._size, self._size, 3), background_color, dtype=np.uint8)

            # tree position
            tree_pos = self._tree_position
            grid[tree_pos[0], tree_pos[1]] = tree_color

            # agent's position
            grid[self._state[0], self._state[1], :] = agent_color

            return grid
        else:
            raise NotImplementedError("Render mode {} is not implemented.".format(mode))
