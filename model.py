import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicRNNCell

SEQ_LEN = 2
BATCH_SIZE = 10
TOTAL_SEQ = 100000
HIDDEN_SIZE = 10
NUM_BATCH = TOTAL_SEQ // (SEQ_LEN * BATCH_SIZE)
LEARNING_RATE_START = 1e-1
LEARNING_RATE_MIN = 1e-6
LEARNING_RATE_CUT_EPOCH = 10
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
        # if time_step > 0:
        #     tf.get_variable_scope().reuse_variables()

        (cell_output, state) = rnn_cell(input_placeholder[:, time_step, :], state)
        states.append(state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(1, outputs), [-1, HIDDEN_SIZE])
    cell_states = tf.reshape(tf.concat(1, states), [-1, HIDDEN_SIZE])

    labels_batched = tf.reshape(target_placeholder, [-1])
    target_weights = tf.ones([BATCH_SIZE * (SEQ_LEN - 1)])

    softmax_outputs = tf.nn.softmax(output)
    loss = tf.nn.seq2seq.sequence_loss(
        [output],
        [labels_batched],
        [target_weights])

optimizer = tf.train.AdamOptimizer(learning_rate_placeholder)
grads_and_vars = optimizer.compute_gradients(loss)
train_step = optimizer.apply_gradients(grads_and_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
learning_rate = LEARNING_RATE_START

best_loss = np.inf
epochs_without_improvement = 0

for epoch_idx in range(NUM_EPOCH):
    if epochs_without_improvement >= LEARNING_RATE_CUT_EPOCH:
        learning_rate /= 10
        print("Cutting learning rate to", learning_rate)
    if learning_rate <= LEARNING_RATE_MIN:
        print("Ending training since model is not learning")
        break
    for batch_idx in range(NUM_BATCH):
        batch_input, batch_output = make_batch(random_inputs[batch_idx], random_outputs[batch_idx])
        fetch_output, fetch_labels, fetch_label_weights, fetch_softmax, fetch_loss, fetch_grad_vars, _ = sess.run(
            [output, labels_batched, target_weights, softmax_outputs, loss, grads_and_vars, train_step],
            feed_dict={
                input_placeholder: batch_input,
                target_placeholder: batch_output,
                learning_rate_placeholder: learning_rate}
        )
    if best_loss * 0.9999 < fetch_loss:
        print("Current loss ", fetch_loss, "was not significantly better than best loss of", best_loss)
        epochs_without_improvement += 1
        print("Now at", epochs_without_improvement, "epoch(s) without improvement out of", LEARNING_RATE_CUT_EPOCH)
    else:
        best_loss = fetch_loss
        epochs_without_improvement = 0
        print("Got new best loss of: ", best_loss)

    if not fetch_loss:
        raise Exception("You set either NUM_EPOCH or NUM_BATCH to 0")
    else:
        print("Epoch:", epoch_idx, "Loss:", fetch_loss, "LR:", learning_rate)

print("Training completed - attained loss", best_loss)
