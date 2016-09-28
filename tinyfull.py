from input_pipeline import *
import time


train_image_batch, train_label_batch, TRAIN_SIZE = create_input_pipeline(LABELS_FILE_TRAIN, BATCH_SIZE, num_epochs=None, produceVGGInput=False)

# simple model
w = tf.get_variable("w1", [SHAPE, LABEL_CNT])
y_pred = tf.matmul(train_image_batch, w)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, train_label_batch)

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
    _, loss_val = sess.run([train_op, loss_mean])
    duration += time.time() - start_time
    
    if step % 100 == 0:
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_val, duration))
        duration = 0.0
