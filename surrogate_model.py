import tensorflow as tf
from nets import resnet_v2
import numpy as np
import cv2

slim = tf.contrib.slim

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel

kernel = gkern(15, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)

class ResNet152:
  def __init__(self, source_image_size, use_smoothed_grad=False):
    self.image_size = 299
    self.source_image_size = source_image_size
    self.num_classes = 1001
    self.predictions_is_correct = False

    batch_shape = [None, self.image_size, self.image_size, 3]
    self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
    self.target_label = tf.placeholder(tf.int32, shape=[None])
    target_onehot = tf.one_hot(self.target_label, self.num_classes)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      logits, end_points = resnet_v2.resnet_v2_152(
        self.x_input, num_classes=self.num_classes, is_training=False)

    self.predicted_labels = tf.argmax(end_points['predictions'], 1)
    #logits -= tf.reduce_min(logits)
    #real = tf.reduce_max(logits * target_onehot, 1)
    #other = tf.reduce_max(logits * (1 - target_onehot), 1)
    #self.loss = other - real
    self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_onehot, logits=logits)
    self.grad = 2*tf.gradients(self.loss, self.x_input)[0]
    if use_smoothed_grad:
      self.grad = tf.nn.depthwise_conv2d(self.grad, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')

    saver = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
    self.sess = tf.get_default_session()
    saver.restore(self.sess, 'resnet_v2_152.ckpt')

  def get_loss(self, images, labels):
    if len(images.shape) == 3:
      images = images[np.newaxis]
    if self.source_image_size != self.image_size:
      images = np.array([cv2.resize(images[i], (self.image_size, self.image_size))
        for i in range(images.shape[0])])
    images = images * 2.0 - 1.0

    return self.sess.run(self.loss,
      feed_dict={self.x_input: images, self.target_label: labels})

  def get_grad(self, images, labels):
    if len(images.shape) == 3:
      images = images[np.newaxis]
    if self.source_image_size != self.image_size:
      images = np.array([cv2.resize(images[i], (self.image_size, self.image_size))
        for i in range(images.shape[0])])
    images = images * 2.0 - 1.0

    grad = self.sess.run(self.grad,
      feed_dict={self.x_input: images, self.target_label: labels})
    if self.source_image_size != self.image_size:
      grad = np.array([cv2.resize(grad[i], (self.source_image_size, self.source_image_size))
        for i in range(grad.shape[0])])
    return grad

  def get_pred(self, images):
    if len(images.shape) == 3:
      images = images[np.newaxis]
    if self.source_image_size != self.image_size:
      images = np.array([cv2.resize(images[i], (self.image_size, self.image_size))
        for i in range(images.shape[0])])
    images = images * 2.0 - 1.0

    return self.sess.run(self.predicted_labels,
      feed_dict={self.x_input: images})