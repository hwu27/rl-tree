# This DQN pipeline is based off the Tensorflow DQN tutorial 
# https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

from __future__ import absolute_import, division, print_function

import numpy as np
import rl_tree
import reverb

import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

# Basic hyperparameters from DQN tutorial from TensorFlow
num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 2000  # @param {type:"integer"}
collect_steps_per_iteration =   100 # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3 # @param {type:"number"}
log_interval = 100  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 100  # @param {type:"integer"}

# create the environment and wrap it in a tf wrapper
max_actions = 25
env = rl_tree.ParkingEnvironment(size=6, max_actions=max_actions)

train_py_env = rl_tree.ParkingEnvironment(size=6, max_actions=max_actions)
eval_py_env = rl_tree.ParkingEnvironment(size=6, max_actions=max_actions)  # create an evaluation environment that mirrors the original env for training

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (128, 64) # this is going to be the number of neurons in each dense layer 

action_tensor_spec = tensor_spec.from_spec(env.action_spec()) # specifications of action space (left, right, up, down)

num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1 # max number of actions aka 4
# print(num_actions)

# helper function for creating dense layers

def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="truncated_normal"))

# create a Q-network consisting of dense layers followed by dense layer with "num_actions" units (a q_value for each action unit is generated)

dense_layers = [dense_layer(num_units) for num_units in fc_layer_params] # list of dense layers

# output layer where
# each neuron outputs a corresponding Q-value based on actions

q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))

# sequential model with dense layers and output layers
q_net = sequential.Sequential(dense_layers + [q_values_layer])

# adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# training counter
train_step_counter = tf.Variable(0)

# create agent
agent = dqn_agent.DqnAgent(
    # information about environment
    train_env.time_step_spec(), 
    # possible actioms
    train_env.action_spec(),
    # q network
    q_network = q_net,
    optimizer = optimizer,
    # wise-squared loss function
    td_errors_loss_fn = common.element_wise_squared_loss, 
    # step counter
    train_step_counter = train_step_counter
)

# initialize agent
agent.initialize()

# policies for evaluation and one for data collection
eval_policy = agent.policy
collect_policy = agent.collect_policy

# function for evaluation of our RL model
# " The return is the sum of rewards obtained while running a policy in an environment for an episode. 
# Several episodes are run, creating an average return."

def compute_avg_return(environment, policy, num_episodes, max_steps_per_episode=max_actions):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        step_count = 0

        while not time_step.is_last() and step_count < max_steps_per_episode:
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            step_count += 1

        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


# here is an example with the random policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

# print(compute_avg_return(eval_env, random_policy, num_eval_episodes))

# we will use reverb to keep track of data in the environment that is to be fed into the agent during training
table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=2)

# this is a Trajectory tuple that contains the observations, actions, and rewards from the data collection
#print(agent.collect_data_spec)



# data collection with PyDriver
# starts off with random policy

py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
      random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps).run(train_py_env.reset())

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# Start training

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)
# Reset the train step.
agent.train_step_counter.assign(0)
# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)

returns = [avg_return]


# Reset the environment.
time_step = train_py_env.reset()
global_step = tf.Variable(0) # global step counter over multiple training session
policy_dir = 'policy'
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
      collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

checkpoint_dir = 'checkpoint'
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)

#train_checkpointer.initialize_or_restore()

for _ in range(num_iterations):
    # Collect a few steps and save to the replay buffer.
    time_step, _ = collect_driver.run(time_step)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)

    train_loss = agent.train(experience).loss

    # Checking Q-Values
    observation = tf.expand_dims(time_step.observation, axis=0)
    #print("Q-values:", agent._q_network(observation))

    step = agent.train_step_counter.numpy() # this step is just for the current training session
    global_step.assign_add(1)

    if step % log_interval == 0:
        train_checkpointer.save(global_step=step)
        tf_policy_saver.save(policy_dir)
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)


# REMEMBER TO USE THIS: WRAPT_DISABLE_EXTENSIONS=1 python3 dqn_model.py 