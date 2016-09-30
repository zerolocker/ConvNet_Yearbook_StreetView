import os
import tensorflow as tf
import numpy as np

import vggrepo.vgg19_trainable as vgg19

from IPython import embed

class regression:
    def buildForVGG(self, vgg, init_bias=1905):
        x = vgg.relu7
        shape = x.get_shape().as_list()
        self.W = tf.Variable(tf.truncated_normal([shape[1],1],
                            stddev=1.0 / shape[1]))
        self.b = tf.Variable([init_bias], dtype=tf.float32)
        self.pred = tf.nn.bias_add(tf.matmul(x, self.W), self.b)

if __name__ == "__main__":
	print("Running unit tests for regression layer")
	reg = regression()
	vgg = vgg19.Vgg19('./vggrepo/vgg19.npy')

	images_placeholder = tf.placeholder(tf.float32, shape=(10, 224, 224, 3)) 
	rnd_batch = np.random.randn(10, 224, 224, 3)
	train_mode = tf.placeholder(tf.bool)

	vgg.build(images_placeholder, train_mode)
	reg.buildForVGG(vgg)

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	ww,bb = sess.run([reg.W, reg.b])
	pred = sess.run(reg.pred, feed_dict={images_placeholder: rnd_batch, train_mode: True})
	print(ww)
	print(bb)
	print(pred)

