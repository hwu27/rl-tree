import numpy as np
import parking
import animate

import tensorflow as tf
from tf_agents.environments import tf_py_environment

max_actions = 30
env = parking.ParkingEnvironment(size=6, max_actions=max_actions)
tf_env = tf_py_environment.TFPyEnvironment(env)
tf_env.reset()

init_frame = tf_env.render()

frames = [init_frame]

episodes = 10
for episode in range(episodes):
    #print(f"Episode: {episode}")
    reward_score = 0
    tf_env.reset()  # reset the environment at the start of each episode
    time_step = tf_env.current_time_step()  #  initial time step
    count = 0
        
    actions = np.array([np.random.randint(0, 4) for _ in range(max_actions)])
    for action in actions:
        count+=1
        print("Actual number of actions", count)
        if time_step.is_last():
            break  # end the action loop if the episode has terminated
        time_step = tf_env.step(action)  # action
        reward_score += time_step.reward.numpy()[0]  # reward
        frame = tf_env.render()  # render current state
        frames.append(frame)  # append the frame to the frames list
    #print(f"Rewards: {reward_score}")

actions_arr = env.last_terminated_number_actions()

print(actions_arr)
animate.animate_frames(frames, actions_per_episode=actions_arr)
 

