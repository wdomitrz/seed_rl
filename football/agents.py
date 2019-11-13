# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SEED agent using Keras."""

import collections
from seed_rl.common import utils
from seed_rl.football import observation
import tensorflow as tf

import gym

AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


class _Stack(tf.Module):
  """Stack of pooling and convolutional blocks with residual connections."""

  def __init__(self, num_ch, num_blocks):
    
    super(_Stack, self).__init__(name='stack')
    self._conv = tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same',
                                        kernel_initializer='lecun_normal')
    self._max_pool = tf.keras.layers.MaxPool2D(
        pool_size=3, padding='same', strides=2)

    self._res_convs0 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_0' % i,
            kernel_initializer='lecun_normal')
        for i in range(num_blocks)
    ]
    self._res_convs1 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_1' % i,
            kernel_initializer='lecun_normal')
        for i in range(num_blocks)
    ]

  def __call__(self, conv_out):
    # Downscale.
    conv_out = self._conv(conv_out)
    conv_out = self._max_pool(conv_out)

    # Residual block(s).
    for (res_conv0, res_conv1) in zip(self._res_convs0, self._res_convs1):
      block_input = conv_out
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv0(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv1(conv_out)
      conv_out += block_input

    return conv_out


def make_logits(layer_fn, action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return layer_fn(action_space.n, 'policy_logits')
    else:
        print(layer_fn(0, 'policy_logits'))
        return [layer_fn(n, 'policy_logits') for n in action_space.nvec]


def apply_net(action_space, policy_logits, core_output):
    print('apply_net:')
    if isinstance(action_space, gym.spaces.Discrete):
        return policy_logits(core_output)
    else:
        n_actions = action_space.nvec.shape[0]
        arr = [policy_logits[i](core_output) for i in range(n_actions)]
        arr = tf.stack(arr)
        print('a_n_before:', arr)
        arr = tf.transpose(arr, perm=[1, 0, 2])
        print('a_n_after', arr)
        return arr


def sample_action(action_space, policy_logits):
    print('sample_action')
    if isinstance(action_space, gym.spaces.Discrete):
        new_action = tf.random.categorical(policy_logits, 1, dtype=tf.int32)
        new_action = tf.squeeze(new_action, 1, name='action')
        return new_action
    else:
        n_actions = action_space.nvec.shape[0]
        policy_logits = tf.transpose(policy_logits, perm=[1, 0, 2])
        print("s_a_pl:", policy_logits)
        print(tf.random.categorical(policy_logits[0], 1, dtype=tf.int32))
        new_action = tf.stack([tf.squeeze(tf.random.categorical(policy_logits[i], 1, dtype=tf.int32), 1) for i in range(n_actions)])
        print("s_a_pl_a:")
        print('s_a_before', new_action)
        new_action = tf.transpose(new_action, perm=[1, 0])
        print('s_a_after', new_action)
        return new_action


class GFootball(tf.Module):
  """Agent with ResNet, but without LSTM and additional inputs.

  Four blocks instead of three in ImpalaAtariDeep.
  """

  def __init__(self, action_space):
    super(GFootball, self).__init__(name='gfootball')

    # Parameters and layers for unroll.
    self._action_space = action_space

    # Parameters and layers for _torso.
    self._stacks = [
        _Stack(num_ch, num_blocks)
        for num_ch, num_blocks in [(16, 2), (32, 2), (32, 2), (32, 2)]
    ]
    self._conv_to_linear = tf.keras.layers.Dense(
        256, kernel_initializer='lecun_normal')

    # Layers for _head.


    self._policy_logits = make_logits(lambda num_units, name: tf.keras.layers.Dense(
        num_units,
        name=name,
        kernel_initializer='lecun_normal'), self._action_space)
    self._baseline = tf.keras.layers.Dense(
        1, name='baseline', kernel_initializer='lecun_normal')

  def initial_state(self, batch_size):
    return ()

  def _torso(self, unused_prev_action, env_output):
    _, _, frame = env_output

    frame = observation.unpackbits(frame)
    frame /= 255

    conv_out = frame
    for stack in self._stacks:
      conv_out = stack(conv_out)

    conv_out = tf.nn.relu(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)

    conv_out = self._conv_to_linear(conv_out)
    return tf.nn.relu(conv_out)

  def _head(self, core_output):
    policy_logits = apply_net(self._action_space, self._policy_logits, core_output)
    baseline = tf.squeeze(self._baseline(core_output), axis=-1)

    # Sample an action from the policy.
    new_action = sample_action(self._action_space, policy_logits)

    return AgentOutput(new_action, policy_logits, baseline)

  def __call__(self, input_, core_state, unroll=False):
    if not unroll:
      # Add time dimension.
      input_ = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), input_)
    prev_actions, env_outputs = input_
    outputs, core_state = self._unroll(prev_actions, env_outputs, core_state)
    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

    return outputs, core_state

  def _unroll(self, prev_actions, env_outputs, core_state):
    torso_outputs = utils.batch_apply(self._torso, (prev_actions, env_outputs))
    return utils.batch_apply(self._head, (torso_outputs,)), core_state
