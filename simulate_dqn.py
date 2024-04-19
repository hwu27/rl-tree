from __future__ import absolute_import, division, print_function

import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import parking
import animate

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential

max_actions = 10
eval_py_env = parking.ParkingEnvironment(size=6, max_actions=max_actions)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

def load_model(eval_env):

    learning_rate = 1e-3 # @param {type:"number"}

    # create the environment and wrap it in a tf wrapper

    fc_layer_params = (128, 64) # this is going to be the number of neurons in each dense layer 

    action_tensor_spec = tensor_spec.from_spec(eval_py_env.action_spec()) # specifications of action space (left, right, up, down)

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
        eval_env.time_step_spec(), 
        # possible actioms
        eval_env.action_spec(),
        # q network
        q_network = q_net,
        optimizer = optimizer,
        # wise-squared loss function
        td_errors_loss_fn = common.element_wise_squared_loss, 
        # step counter
        train_step_counter = train_step_counter
    )

    checkpoint_dir = 'checkpoints'
    eval_checkpointer = common.Checkpointer(ckpt_dir=checkpoint_dir, max_to_keep=1, agent=agent)
    eval_checkpointer.initialize_or_restore()
    return agent

agent = load_model(eval_env)

eval_env.reset()
init_frame = eval_env.render()
frames = [init_frame]

episodes = 10
# Evaluating the trained model
frames = []
actions_count = 0
for _ in range(episodes):
    time_step = eval_env.reset()
    while actions_count != max_actions:
        action_step = agent.policy.action(time_step)
        #print(action_step)
        time_step = eval_env.step(action_step.action)
        frames.append(eval_env.render())
        actions_count+=1
    actions_count = 0

actions_arr = eval_py_env.last_terminated_number_actions()
# Animating the frames  
animate.animate_frames(frames, actions_arr)