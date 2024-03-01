import numpy as np
import alt
import animate

import tensorflow as tf
from tf_agents.environments import tf_py_environment


# Basic hyperparameters from DQN tutorial from TensorFlow
num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

max_actions = 10
env = alt.AltruismEnvironment(size=6, max_actions=max_actions)
tf_env = tf_py_environment.TFPyEnvironment(env)
tf_env.reset()

init_frame = tf_env.render()

frames = [init_frame]


episodes = 10
for episode in range(episodes):
    print(f"Episode: {episode}")
    reward_score = 0
    tf_env.reset()  # reset the environment at the start of each episode
    time_step = tf_env.current_time_step()  #  initial time step

    actions = np.array([np.random.randint(0, 4) for _ in range(max_actions)])
    for action in actions:
        if time_step.is_last():
            break  # end the action loop if the episode has terminated
        time_step = tf_env.step(action)  # action
        reward_score += time_step.reward.numpy()[0]  # reward
        frame = tf_env.render()  # render current state
        frames.append(frame)  # append the frame to the frames list
    print(f"Rewards: {reward_score}")

actions_arr = env.last_terminated_number_actions()

#print(actions_arr)
#animate.animate_frames(frames, actions_per_episode=actions_arr)
 

