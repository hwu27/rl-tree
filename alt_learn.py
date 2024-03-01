# This DQN pipeline is based off the Tensorflow DQN tutorial 
# https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

from __future__ import absolute_import, division, print_function

import numpy as np
import alt
import animate
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
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

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

# create the environment and wrap it in a tf wrapper
max_actions = 10
env = alt.AltruismEnvironment(size=6, max_actions=max_actions)
tf_env = tf_py_environment.TFPyEnvironment(env)
tf_env.reset()

eval_env = tf_env # create an evaluation environment that mirrors the original tf_env for training

fc_layer_params = (100, 50) # this is going to be the number of neurons in each dense layer 

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
    tf_env.time_step_spec(), 
    # possible actioms
    tf_env.action_spec(),
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
collect_policty = agent.collect_policy

# function for evaluation of our RL model
# " The return is the sum of rewards obtained while running a policy in an environment for an episode. 
# Several episodes are run, creating an average return."

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


# here is an example with the random policy
random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(),
                                                tf_env.action_spec())

print(compute_avg_return(eval_env, random_policy, num_eval_episodes))

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
agent.collect_data_spec
