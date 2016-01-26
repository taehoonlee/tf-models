import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

class SeqClassificationModel(object):

  def __init__(self, config, dtype=tf.float32):

    self.batch_size = config.batch_size
    seq_len = config.seq_len
    num_classes = config.num_classes
    rnn_units = config.rnn_units
    fc_units = config.fc_units
    input_size = config.input_size
    stride_size = config.stride_size
    weight_decay = config.weight_decay

    self.inputs = tf.placeholder(dtype, shape=[self.batch_size, seq_len])
    self.targets = tf.placeholder(dtype, shape=[self.batch_size, num_classes])

    cell = rnn_cell.LSTMCell(num_units=rnn_units, input_size=input_size)
    self.istate = cell.zero_state(self.batch_size, dtype)
    state = self.istate

    r = 1 / np.sqrt(input_size + rnn_units + 1) / 10.0
    with tf.variable_scope("RNN", initializer=tf.random_uniform_initializer(-r, r)):
      for step, i in enumerate(range(input_size, seq_len+1, stride_size)):
        if step > 0: tf.get_variable_scope().reuse_variables()
        hidden, state = cell(self.inputs[:,i-input_size:i], state)

    with tf.variable_scope("RNN/LSTMCell", reuse=True):
      RNNW = tf.get_variable("W_0")

    reg = 0.0

    if fc_units > 0:
      with tf.variable_scope("hidden"):
        r = 1 / np.sqrt(rnn_units + fc_units + 1) / 10.0
        HiddenW = tf.get_variable("W", [rnn_units, fc_units], initializer=tf.random_uniform_initializer(-3*r, r))
        Hiddenb = tf.get_variable("b", [fc_units], initializer=tf.constant_initializer(0.0))
      hidden_units = fc_units
      hiddens = tf.nn.relu(tf.matmul(hidden, HiddenW) + Hiddenb)
      reg += tf.nn.l2_loss(HiddenW)
    else:
      hidden_units = rnn_units

    with tf.variable_scope("softmax"):
      r = 1 / np.sqrt(hidden_units + num_classes + 1) / 10.0
      OutW = tf.get_variable("W", [hidden_units, num_classes], initializer=tf.random_uniform_initializer(-3*r, r))
      Outb = tf.get_variable("b", [num_classes], initializer=tf.constant_initializer(0.0))

    if fc_units > 0:
      self.outputs = tf.matmul(hiddens, OutW) + Outb
    else:
      self.outputs = tf.matmul(hidden, OutW) + Outb
    reg += tf.nn.l2_loss(OutW)

    self.loss = weight_decay * reg + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.outputs, self.targets)) / self.batch_size

    self.alpha = tf.placeholder(dtype, shape=[])
    self.momentum = tf.placeholder(dtype, shape=[])

    self.train = tf.train.MomentumOptimizer(self.alpha, self.momentum).minimize(self.loss)

    self.init = tf.initialize_all_variables()

    with tf.device("/cpu:0"):
      self.saver = tf.train.Saver()

  def load(self, sess, filename):
    self.saver.restore(sess, filename)

  def save(self, sess, filename):
    self.saver.save(sess, filename)

  def forward_all(self, sess, X, y, acc=False):
    preds = []
    for b in range(0, X.shape[0], self.batch_size):
      state = sess.run(self.istate)
      pred = sess.run(self.outputs,
                     {self.istate: state,
                      self.inputs: X[b:b+self.batch_size],
                      self.targets: y[b:b+self.batch_size]})
      preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    if acc:
      return sum(np.argmax(preds, axis=1) == np.argmax(y, axis=1)) / float(X.shape[0])
    else:
      return preds
