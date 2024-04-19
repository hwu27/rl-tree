from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import helper

class ParkingEnvironment(py_environment.PyEnvironment):

    def __init__(self, size, max_actions):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,),  # agent x, agent y
            dtype=np.int32,
            minimum=[0, 0], 
            maximum=[size - 1, size - 1],  
            name='observation')
        self._size = size # grid size n x n
        # y and then x
        self._state = np.array([size-1, 0], dtype=np.int32) # start position
        self._tree_position = np.array([size/2, size/2], dtype=np.int32) # tree position
        self._max_actions = max_actions
        self._actions_count = 0
        self._occupied_positions = np.empty((0, 2), dtype=np.int32)        
        self._last_terminated_number_actions = np.array([], dtype=np.int32)
        self._prev_state = self._state 
        self._current_time_step = None


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def last_terminated_number_actions(self):
        return self._last_terminated_number_actions
    
    def _reset(self):
        self._current_time_step = ts.restart(np.array(self._state, dtype=np.int32))
        self._actions_count = 0
        self._state = np.array([self._size-1, 0], dtype=np.int32)  # reset to start position
        return self._current_time_step  

    def _step(self, action):
        
        if self._actions_count == self._max_actions-1:
            # Terminate the episode
            self._last_terminated_number_actions = np.append(self._last_terminated_number_actions, self._actions_count)
            return self._reset()
        
        
        # define movement actions and punishments for moving to the edge
        if action == 0:   # up
            if (self._state[0] == 0): # if the agent is at the top edge, punish and reset
                self._current_time_step = ts.transition(np.array(self._state, dtype=np.int32), reward=-1)
                self._actions_count += 1 
                return self._current_time_step
            self._state[0] = max(0, self._state[0] - 1)
        elif action == 1: # down
            if (self._state[0] == self._size - 1): # if the agent is at the bottom edge, punish and reset
                self._current_time_step = ts.transition(np.array(self._state, dtype=np.int32), reward=-1)
                self._actions_count += 1 
                return self._current_time_step
            self._state[0] = min(self._size - 1, self._state[0] + 1)
        elif action == 2: # left
            if (self._state[1] == 0): # if the agent is at the left edge, punish and reset
                self._current_time_step = ts.transition(np.array(self._state, dtype=np.int32), reward=-1)
                self._actions_count += 1 
                return self._current_time_step
            self._state[1] = max(0, self._state[1] - 1)
        elif action == 3: # right
            if (self._state[1] == self._size - 1): # if the agent is at the right edge, punish and reset
                self._current_time_step = ts.transition(np.array(self._state, dtype=np.int32), reward=-1)
                self._actions_count += 1 
                return self._current_time_step
            self._state[1] = min(self._size - 1, self._state[1] + 1)

        closest_tile = helper.closestUnoccupiedTile(self._tree_position, self._state, self._occupied_positions, self._size)
        if (helper.manhattan_distance(self._state, closest_tile) > helper.manhattan_distance(self._prev_state, closest_tile)):
            self._current_time_step = ts.transition(np.array(self._state, dtype=np.int32), reward=5)
            self._actions_count += 1 
            return self._current_time_step
        
        self._prev_state = self._state

        # check if the agent is in an occupied position
        for pos in self._occupied_positions:
            if np.array_equal(self._state, pos):
                # reset to start position
                self._state = np.array([self._size-1, 0], dtype=np.int32)
                self._current_time_step = ts.transition(np.array(self._state, dtype=np.int32), reward=-5)
                self._actions_count += 1 
                return self._current_time_step

        # check if the agent is in a tile adjacent to the tree
        state_list = self._state.tolist()
        occupied_positions_list = self._occupied_positions.tolist()

        if helper.is_adjacent(self._state, self._tree_position) and state_list not in occupied_positions_list:
            self._occupied_positions = np.append(self._occupied_positions, [self._state], axis=0)
            self._state = np.array([self._size - 1, 0], dtype=np.int32)
            self._current_time_step = ts.transition(np.array(self._state, dtype=np.int32), reward=10)
            self._actions_count += 1
            return self._current_time_step
        
        # goal reached
        if len(self._occupied_positions) == 3:
            self._current_time_step = ts.termination(np.array(self._state, dtype=np.int32), reward=20)
            return self._current_time_step

        # increment action count at each step
        self._current_time_step = ts.transition(np.array(self._state, dtype=np.int32), reward=-1)
        self._actions_count += 1  
        return self._current_time_step
        
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
