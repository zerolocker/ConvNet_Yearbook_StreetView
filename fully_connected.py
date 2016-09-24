# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
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
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from input_pipeline import *
import mnist

from IPython import embed # used for interactive debugging


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')


def do_eval(sess, eval_correct, feed_dict, eval_data_size):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The op that returns the number of correct predictions.
  """
  start_time = time.time()
  true_count = 0  # Counts the number of correct predictions.
  
  # BUG! (and I don't bother to fix): due to integer division,  the final batch that contains fewer examples 
  # than the batch size will be just discarded and not used in evaluation. To fix this we need tf.train.batch(allow_smaller_final_batch=True), but it only available in TensorFlow 0.10
  steps_per_epoch = eval_data_size // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  duration = time.time() - start_time
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f Time: %.3f sec' %
        (num_examples, true_count, precision, duration))


def run_training():


  # Tell TensorFlow that the model will be built into the default Graph.
  #with tf.Graph().as_default():
  if True:
    # Generate placeholders for the images and labels.
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,SHAPE))
    labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))

    # Build a Graph that computes predictions from the inference model.
    logits = mnist.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    # Start the training loop.
    duration = 0.0
    for step in xrange(FLAGS.max_steps):
      
      start_time = time.time()
      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      train_feed_dict = {
                images_placeholder: train_image_batch.eval(session=sess),
                labels_placeholder: train_label_batch.eval(session=sess)
      }
                                 
      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss], feed_dict=train_feed_dict)
      duration += time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        duration = 0.0
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=train_feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=step)
        
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess, eval_correct, feed_dict = {
                    images_placeholder: train_image_batch.eval(session=sess),
                    labels_placeholder: train_label_batch.eval(session=sess)
                }, eval_data_size = TRAIN_SIZE)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess, eval_correct, feed_dict = {
                    images_placeholder: val_image_batch.eval(session=sess),
                    labels_placeholder: val_label_batch.eval(session=sess)
                }, eval_data_size = VAL_SIZE)
        print('\n')


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()