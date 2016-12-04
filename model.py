import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicRNNCell

SEQ_LEN = 2
BATCH_SIZE = 10
TOTAL_SEQ = 100000
HIDDEN_SIZE = 10
NUM_BATCH = TOTAL_SEQ // (SEQ_LEN * BATCH_SIZE)
LEARNING_RATE_START = 3e-4
NUM_EPOCH = 100
random_digits = [random.randint(0, 3) for i in range(TOTAL_SEQ)]
random_inputs = np.zeros([NUM_BATCH, BATCH_SIZE, SEQ_LEN, 10])
random_outputs = np.zeros([NUM_BATCH, BATCH_SIZE, SEQ_LEN], dtype=np.int32)

for batch_idx in range(NUM_BATCH):
    for example_idx in range(BATCH_SIZE):
        for seq_idx in range(SEQ_LEN):
            label = random_digits.pop()
            random_inputs[batch_idx, example_idx, seq_idx, label] = 1
            random_outputs[batch_idx, example_idx, seq_idx] = label

rnn_cell = BasicRNNCell(10, activation=tf.sigmoid)


def make_batch(raw_batch_inputs, raw_batch_outputs):
    xs = raw_batch_inputs[:, :SEQ_LEN - 1]
    ys = raw_batch_outputs[:, 1:]

    return xs, ys


input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, SEQ_LEN - 1, HIDDEN_SIZE], name="input")

target_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, SEQ_LEN - 1], name="target")
learning_rate_placeholder = tf.placeholder(dtype=tf.float32, shape=None)

outputs = []
state = rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
states = []

with tf.variable_scope("RNN"):
    for time_step in range(SEQ_LEN - 1):
        if time_step > 0:
            tf.get_variable_scope().reuse_variables()

        (cell_output, state) = rnn_cell(input_placeholder[:, time_step, :], state)
        states.append(state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(1, outputs), [-1, HIDDEN_SIZE])
    cell_states = tf.reshape(tf.concat(1, states), [-1, HIDDEN_SIZE])

    loss = tf.nn.seq2seq.sequence_loss(
        [output],
        [tf.reshape(target_placeholder, [-1])],
        [tf.ones([BATCH_SIZE * (SEQ_LEN - 1)])])

optimizer = tf.train.AdamOptimizer(learning_rate_placeholder)
train_step = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
learning_rate = LEARNING_RATE_START

for epoch_idx in range(NUM_EPOCH):
    fetch_loss = 0
    # if epoch_idx % 10 == 0:
    #     learning_rate /= 2
    for batch_idx in range(NUM_BATCH):
        batch_input, batch_output = make_batch(random_inputs[batch_idx], random_outputs[batch_idx])
        fetch_loss, _ = sess.run(
            [loss, train_step],
            feed_dict={
                input_placeholder: batch_input,
                target_placeholder: batch_output,
                learning_rate_placeholder: learning_rate}
        )

    print("Epoch:", epoch_idx, "Loss:", fetch_loss, "LR:", learning_rate)
