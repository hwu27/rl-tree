from __future__ import absolute_import, division, print_function

import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import rl_tree
import extra.animate as animate

import tensorflow as tf
from tf_agents.environments import tf_py_environment

max_actions = 25
eval_py_env = rl_tree.ParkingEnvironment(size=6, max_actions=max_actions)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

eval_env.reset()
init_frame = eval_env.render()
frames = [init_frame]

episodes = 10
# Evaluating the trained model
frames = []
saved_policy = tf.saved_model.load('policy')

for _ in range(episodes):
    time_step = eval_env.reset()
    count = 0
    for _ in range(max_actions+1):
        if time_step.is_last():
            print("Episode terminated")
            break
        action = saved_policy.action(time_step).action
        print(action)
        time_step = eval_env.step(action)
        frame = eval_env.render()
        frames.append(frame)
        count += 1
    # print reward

actions_arr = eval_py_env.last_terminated_number_actions()
eval_env.close()

print(actions_arr)
animate.animate_frames(frames, actions_per_episode=actions_arr)