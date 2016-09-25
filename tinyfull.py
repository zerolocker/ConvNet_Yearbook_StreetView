from input_pipeline import *



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

while True:
  # pass it in through the feed_dict
  _, loss_val = sess.run([train_op, loss_mean])
  print loss_val
  