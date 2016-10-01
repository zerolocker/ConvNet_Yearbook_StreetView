
from util import *
import tensorflow as tf
import vggrepo.vgg19_trainable_ours as vgg19
import numpy as np


class Predictor:
  DATASET_TYPE = 'yearbook'
  def __init__(self):
    # indicator of how many pictures have been tested
    self.cnt = 0
    self.g = tf.Graph()
    print("Start initialize a predictor")
    with self.g.as_default():
      
      self.image_path = tf.placeholder(tf.string)
      self.image_file = tf.read_file(self.image_path)
      self.image_tf = tf.image.decode_png(self.image_file)
      self.image_float = tf.image.convert_image_dtype(self.image_tf, tf.float32)
      self.image_vggfmt = tf.image.resize_images(self.image_float, 224, 224)
      self.image_batch = tf.reshape(self.image_vggfmt, [1,224,224,3])

      self.vgg = vgg19.Vgg19('./vggrepo/myVGG.lr.1e-04.eps.1e-07.npy', trainable=False)
      self.vgg.build(self.image_batch)
      self.logits = self.vgg.fc8


      self.init = tf.initialize_all_variables()
      self.sess = tf.Session()
      self.sess.run(self.init)

  def predict(self, image_path):
    # Using the default graph
    with self.g.as_default():
      feed_dict = {self.image_path: image_path}
      
      self.logits_run = self.sess.run(self.logits, feed_dict=feed_dict)
      self.year = np.argmax(self.logits_run) + 1905
      # test(print) the prediction each time
      print "Pred: %d for file %s" % (self.year, image_path)
      
      if (self.cnt%100 == 0):
        print('Number of tested images: '+str(self.cnt))
      self.cnt += 1
      return self.year

