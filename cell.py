from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import *
import logging as logging


class HadamardGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
            logging.warning("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope(scope or "gru_cell"):
            with vs.variable_scope("gates"):
                input_dim = inputs.get_shape()[1]
                # Reset gate
                w_r = vs.get_variable(
                    "W_r", [input_dim, self._num_units], dtype=tf.float32)
                u_r = vs.get_variable(
                    "U_r", [self._num_units, self._num_units], dtype=tf.float32)
                b_r = vs.get_variable(
                    "b_r", [self._num_units],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(1., dtype=tf.float32))
                # Update gate
                w_z = vs.get_variable(
                    "W_z", [input_dim, self._num_units], dtype=tf.float32)
                u_z = vs.get_variable(
                    "U_z", [self._num_units, self._num_units], dtype=tf.float32)
                b_z = vs.get_variable(
                    "b_z", [self._num_units],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(1., dtype=tf.float32))

                reset_gate = sigmoid(tf.matmul(inputs, w_r) * tf.matmul(state, u_r) + b_r)
                update_gate = sigmoid(tf.matmul(inputs, w_z) * tf.matmul(state, u_z) + b_z)

            with vs.variable_scope("candidate"):
                # Candidate State
                w_c = vs.get_variable(
                    "W_c", [input_dim, self._num_units], dtype=tf.float32)
                u_c = vs.get_variable(
                    "U_c", [self._num_units, self._num_units], dtype=tf.float32)
                b_c = vs.get_variable(
                    "b_c", [self._num_units],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(1., dtype=tf.float32))
                c = self._activation(
                    tf.matmul(inputs, w_c) +
                    tf.matmul(reset_gate * state, u_c) +
                    b_c)
            new_h = update_gate * state + (1 - update_gate) * c
        return new_h, new_h
