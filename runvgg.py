"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vggrepo.vgg19_trainable as vgg19
import vggrepo.utils as utils
from input_pipeline import *
from IPython import embed;
import time, os


def do_eval(sess, eval_correct, eval_data_size, 
            images_placeholder, labels_placeholder, 
            image_batch, label_batch):
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
    # Never, ever run image_batch.eval() + label_batch.eval() separately
    np_image_batch, np_label_batch = sess.run([image_batch, label_batch])
    true_count += sess.run(eval_correct, feed_dict={
        images_placeholder: np_image_batch,
        labels_placeholder: np_label_batch,
        train_mode: False
    })
  precision = float(true_count) / num_examples
  duration = time.time() - start_time
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f Time: %.3f sec' %
        (num_examples, true_count, precision, duration))

        
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 30, 'Batch size.  ' # for VGG-19, batch_size can only be 30
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_float('eps', 1e-8, 'for ADAM optimizer: a small constant for numerical stability.')
printdebug('learning_rate: %.0e  eps: %.0e' % (FLAGS.learning_rate, FLAGS.eps))

# create input pipelines for training set and validation set
train_image_batch, train_label_batch, TRAIN_SIZE = create_input_pipeline(LABELS_FILE_TRAIN, FLAGS.batch_size, num_epochs=None, produceVGGInput=True)
val_image_batch, val_label_batch, VAL_SIZE = create_input_pipeline(LABELS_FILE_VAL, FLAGS.batch_size, num_epochs=None, produceVGGInput=True)

printdebug("TRAIN_SIZE: %d VAL_SIZE: %d BATCH_SIZE: %d" % (TRAIN_SIZE, VAL_SIZE, FLAGS.batch_size))

# Generate placeholders for the images and labels.
images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 224, 224, 3)) # TODO fix input pipeline
labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('./vggrepo/vgg19.npy')
vgg.build(images_placeholder, train_mode)

sess = tf.Session()

# define loss (input: logits, labels):
logits = vgg.fc8
labels = tf.to_int64(labels_placeholder)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits, labels, name='xentropy') # the logists are stored in vgg.fc8
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')


# define training (input: loss, learning_rate, eps):
# Add a scalar summary for the snapshot loss.
tf.scalar_summary(loss.op.name, loss)
# Create a variable to track the global step.
global_step = tf.Variable(0, name='global_step', trainable=False)
# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.eps)
# Use the optimizer to apply the gradients that minimize the loss
# (and also increment the global step counter) as a single training step.
train_op = optimizer.minimize(loss, global_step=global_step)

# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.merge_all_summaries()
# Instantiate a SummaryWriter to output summaries and the Graph.
summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

# Add the Op to compare the logits to the labels during evaluation.
correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
# Op that computes the number of true entries.
eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())
tf.train.start_queue_runners(sess=sess)

# Start the training loop.
printdebug("Training starts!")
duration = 0.0
for step in xrange(FLAGS.max_steps):
    start_time = time.time()
    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    # Never, ever run image_batch.eval() + label_batch.eval() separately
    np_image_batch, np_label_batch = sess.run([train_image_batch, train_label_batch])
    train_feed_dict = {
            images_placeholder: np_image_batch,
            labels_placeholder: np_label_batch,
            train_mode: True
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
    if (step + 1) % 400 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=step)

        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess, eval_correct, TRAIN_SIZE, 
                images_placeholder, labels_placeholder, 
                train_image_batch, train_label_batch )
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess, eval_correct, VAL_SIZE, 
                images_placeholder, labels_placeholder, 
                val_image_batch, val_label_batch )
        print('\n')


# vgg.save_npy() save the model
vgg.save_npy(sess, './vggrepo/myVGGmodel.npy')


