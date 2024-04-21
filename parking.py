from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import extra.helper as helper

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
        self._actions_count = 1 
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
        self._actions_count = 1
        self._state = np.array([self._size-1, 0], dtype=np.int32)  # reset to start position
        return self._current_time_step  

    def _reward_occured(self, end):
        # check if the agent is in an occupied position
        # reset to start position
        if (end):
            print("Max actions reached, but occupied position reached")
            self._last_terminated_number_actions = np.append(self._last_terminated_number_actions, self._actions_count)
            termination_step = ts.termination(np.array(self._state, dtype=np.int32), reward=-5)
            self._reset()
            return termination_step
        print("Occupied position reached")
        self._state = np.array([self._size-1, 0], dtype=np.int32)
        self._current_time_step = ts.transition(np.array(self._state, dtype=np.int32), reward=-5)
        self._actions_count += 1 
                
    def _reward_adjacent(self, end=False):
        # check if the agent is in a tile adjacent to the tree
        if (end):
            print("Max actions reached, but adjacent")
            self._last_terminated_number_actions = np.append(self._last_terminated_number_actions, self._actions_count)
            termination_step = ts.termination(np.array(self._state, dtype=np.int32), reward=10)
            self._reset()
            return termination_step
        print("Adjacent to tree")
        self._occupied_positions = np.append(self._occupied_positions, [self._state], axis=0)
        self._state = np.array([self._size - 1, 0], dtype=np.int32)
        self._current_time_step = ts.transition(np.array(self._state, dtype=np.int32), reward=10)
        self._actions_count += 1

    def _reward_closest(self, end=False):
        # check if the agent is going towards the closest unoccupied tile adjacent to the tree
        if (end):
            print("Max actions reached, but getting closer to adjacent tree tile")
            self._last_terminated_number_actions = np.append(self._last_terminated_number_actions, self._actions_count)
            termination_step = ts.termination(np.array(self._state, dtype=np.int32), reward=5)
            self._reset()
            return termination_step
        else:
            self._current_time_step = ts.transition(np.array(self._state, dtype=np.int32), reward=5)
            self._actions_count += 1 
            print("Going towards closest tree adjacent tile")

    def _reward_goal_reached(self, end=False):
        # goal reached
        if (end):
            print("Max actions reached, but YOU DID IT")
            self._last_terminated_number_actions = np.append(self._last_terminated_number_actions, self._actions_count)
            termination_step = ts.termination(np.array(self._state, dtype=np.int32), reward=20)
            self._reset()
            return termination_step
        else:
            self._current_time_step = ts.termination(np.array(self._state, dtype=np.int32), reward=20)
            print("YOU DID IT")

    def _make_step(self, action):
        
        # define movement actions and punishments for moving to the edge
        if action == 0:   # up
            self._state[0] = max(0, self._state[0] - 1)
        elif action == 1: # down
            self._state[0] = min(self._size - 1, self._state[0] + 1)
        elif action == 2: # left
            self._state[1] = max(0, self._state[1] - 1)
        elif action == 3: # right
            self._state[1] = min(self._size - 1, self._state[1] + 1)
        self._prev_state = self._state

    def _step(self, action):

        # make a step based off the action
        self._make_step(action)
        print("Agent's position:", self._state)
        print("Tree's position:", self._tree_position)
        # check if the agent is in an occupied position
        for pos in self._occupied_positions:
            if np.array_equal(self._state, pos):
                self._reward_occured()
                return self._current_time_step
            
        # check if the agent is in a tile adjacent to the tree
        state_list = self._state.tolist()
        occupied_positions_list = self._occupied_positions.tolist()
        
        if helper.is_adjacent(self._state, self._tree_position) and state_list not in occupied_positions_list:
            if (self._actions_count == self._max_actions):  
                return self._reward_adjacent(True)
            else:
                self._reward_adjacent()
                return self._current_time_step
        
        # check if the agent is going towards the closest tile to the tree
        closest_tile = helper.closestUnoccupiedTile(self._tree_position, self._state, self._occupied_positions, self._size)
        if (helper.manhattan_distance(self._state, closest_tile) < helper.manhattan_distance(self._prev_state, closest_tile)):
            if (self._actions_count == self._max_actions):
                return self._reward_closest(True)
            else:
                self._reward_closest()
                return self._current_time_step
        
        # check if the agent has reached the goal
        if len(self._occupied_positions) == 4:
            if (self._actions_count == self._max_actions):
                return self._reward_goal_reached(True)
            else:
                self._reward_goal_reached()
                return self._current_time_step

        # actions count and max

        # increment action count at each step and check for end if no other checks
        if (self._actions_count == self._max_actions):  
            print("Max actions reached")
            self._last_terminated_number_actions = np.append(self._last_terminated_number_actions, self._actions_count)
            termination_step = ts.termination(np.array(self._state, dtype=np.int32), reward=0)
            self._reset()
            return termination_step
        else:
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
