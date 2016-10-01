from input_pipeline import *
import time

BATCH_SIZE=100

def do_eval(sess, eval_correct, eval_data_size, 
            images_placeholder, labels_placeholder, 
            image_batch, label_batch):
  start_time = time.time()
  true_count = 0  # Counts the number of correct predictions.
  # BUG! (and I don't bother to fix): due to integer division,  the final batch that contains fewer examples 
  # than the batch size will be just discarded and not used in evaluation. To fix this we need tf.train.batch(allow_smaller_final_batch=True), but it only available in TensorFlow 0.10
  steps_per_epoch = eval_data_size // BATCH_SIZE
  num_examples = steps_per_epoch * BATCH_SIZE
  for step in xrange(steps_per_epoch):
    # Never, ever run image_batch.eval() + label_batch.eval() separately
    np_image_batch, np_label_batch = sess.run([image_batch, label_batch])
    true_count += sess.run(eval_correct, feed_dict={
        images_placeholder: np_image_batch,
        labels_placeholder: np_label_batch,
    })
  precision = float(true_count) / num_examples
  duration = time.time() - start_time
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f Time: %.3f sec' %
        (num_examples, true_count, precision, duration))

train_image_batch, train_label_batch, TRAIN_SIZE = create_input_pipeline(LABELS_FILE_TRAIN, BATCH_SIZE, num_epochs=None, produceVGGInput=False)
val_image_batch, val_label_batch, VAL_SIZE = create_input_pipeline(LABELS_FILE_VAL, BATCH_SIZE, num_epochs=None, produceVGGInput=False)
printdebug("TRAIN_SIZE: %d VAL_SIZE: %d BATCH_SIZE: %d " % (TRAIN_SIZE, VAL_SIZE, BATCH_SIZE))

# Generate placeholders for the images and labels.
images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE,NUM_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE))

# simple model
w = tf.get_variable("w1", [NUM_PIXELS, LABEL_CNT])
logits = tf.matmul(images_placeholder, w)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_placeholder)

# Add the Op to compare the logits to the labels during evaluation.
correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
# Op that computes the number of true entries.
eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

# for monitoring
loss_mean = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

step = 0
duration = 0.0
while True:
    step += 1
    
    start_time = time.time()
    np_image_batch, np_label_batch = sess.run([train_image_batch, train_label_batch])
    _, loss_val = sess.run([train_op, loss_mean], feed_dict={
        images_placeholder: np_image_batch,
        labels_placeholder: np_label_batch
    })
    duration += time.time() - start_time
    
    if step % 100 == 0:
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_val, duration))
        duration = 0.0
    if step % 1000 == 0:
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
        
