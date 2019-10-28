from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import math
import numpy as np
from scipy.misc import imread, imsave, imresize
import cv2

import tensorflow as tf
from nets import inception_v3, resnet_v2

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_enum(
    'norm', 'l2', ['l2', 'linfty'], 'The norm used in the attack.')

tf.flags.DEFINE_enum(
    'method', 'biased', ['uniform', 'biased', 'fixed_biased'], 'Methods used in the attack.')

tf.flags.DEFINE_boolean(
    'dataprior', False, 'Whether to use data prior in the attack.')

tf.flags.DEFINE_boolean(
    'show_loss', False, 'Whether to print loss in some given step sizes.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory to save results.')

tf.flags.DEFINE_integer(
    'samples_per_draw', 50, 'Number of samples to estimate the gradient.')

tf.flags.DEFINE_float(
    'sigma', 1e-4, 'Sampling variance.')

tf.flags.DEFINE_integer(
    'plateau_length', 5, 'Tuning learning rate.')

tf.flags.DEFINE_integer(
    'number_images', 1000, 'Number of images for evaluation.')

tf.flags.DEFINE_integer(
    'max_queries', 10000, 'Maximum number of queries.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir):
  for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png')))[:FLAGS.number_images]:
    image = imread(filepath, mode='RGB').astype(np.float) / 255.0

    yield os.path.basename(filepath), image

class Model:
  def __init__(self):
    batch_shape = [None, 299, 299, 3]
    self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
    self.target_label = tf.placeholder(tf.int32, shape=[None])
    target_onehot = tf.one_hot(self.target_label, 1001)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
      logits, end_points = inception_v3.inception_v3(
        self.x_input, num_classes=1001, is_training=False)

    self.predicted_labels = tf.argmax(end_points['Predictions'], 1)
    #logits -= tf.reduce_min(logits)
    #real = tf.reduce_max(logits * target_onehot, 1)
    #other = tf.reduce_max(logits * (1 - target_onehot), 1)
    #self.loss = other - real
    self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_onehot, logits=logits)
    self.grad = 2*tf.gradients(self.loss, self.x_input)[0]

    saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
    self.sess = tf.get_default_session()
    saver.restore(self.sess, FLAGS.checkpoint_path)

  def get_loss(self, images, labels):
    if len(images.shape) == 3:
      images = images[np.newaxis]
    images = images * 2.0 - 1.0

    return self.sess.run(self.loss,
      feed_dict={self.x_input: images, self.target_label: labels})

  def get_grad(self, images, labels):
    if len(images.shape) == 3:
      images = images[np.newaxis]
    images = images * 2.0 - 1.0

    return self.sess.run(self.grad,
      feed_dict={self.x_input: images, self.target_label: labels})

  def get_pred(self, images):
    if len(images.shape) == 3:
      images = images[np.newaxis]
    images = images * 2.0 - 1.0

    return self.sess.run(self.predicted_labels,
      feed_dict={self.x_input: images})

class Model_surrogate:
  def __init__(self):
    batch_shape = [None, 299, 299, 3]
    self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
    self.target_label = tf.placeholder(tf.int32, shape=[None])
    target_onehot = tf.one_hot(self.target_label, 1001)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      logits, end_points = resnet_v2.resnet_v2_152(
        self.x_input, num_classes=1001, is_training=False)

    self.predicted_labels = tf.argmax(end_points['predictions'], 1)
    #logits -= tf.reduce_min(logits)
    #real = tf.reduce_max(logits * target_onehot, 1)
    #other = tf.reduce_max(logits * (1 - target_onehot), 1)
    #self.loss = other - real
    self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_onehot, logits=logits)
    self.grad = 2*tf.gradients(self.loss, self.x_input)[0]

    saver = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
    self.sess = tf.get_default_session()
    saver.restore(self.sess, 'resnet_v2_152.ckpt')

  def get_loss(self, images, labels):
    if len(images.shape) == 3:
      images = images[np.newaxis]
    images = images * 2.0 - 1.0

    return self.sess.run(self.loss,
      feed_dict={self.x_input: images, self.target_label: labels})

  def get_grad(self, images, labels):
    if len(images.shape) == 3:
      images = images[np.newaxis]
    images = images * 2.0 - 1.0

    return self.sess.run(self.grad,
      feed_dict={self.x_input: images, self.target_label: labels})

  def get_pred(self, images):
    if len(images.shape) == 3:
      images = images[np.newaxis]
    images = images * 2.0 - 1.0

    return self.sess.run(self.predicted_labels,
      feed_dict={self.x_input: images})


def main(_):
  image_height = 299
  image_width = 299

  if FLAGS.norm == 'l2':
    epsilon = 1e-3
    eps = np.sqrt(epsilon * image_height * image_width * 3)
    learning_rate = 2.0 / 299 / 1.7320508
  else:
    epsilon = 0.05
    eps = epsilon
    learning_rate = 0.005


  tf.logging.set_verbosity(tf.logging.INFO)
  output_logging = open(os.path.join(FLAGS.output_dir, 'logging'), 'w')

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.Graph().as_default():
    # Prepare graph
    with tf.Session(config=config).as_default():
      model = Model()
      if FLAGS.method != 'uniform':
        model_s = Model_surrogate()
    success = 0
    queries = []
    correct = 0

    names_images = load_images(FLAGS.input_dir)
    for filename, image in names_images:
      sigma = FLAGS.sigma
      np.random.seed(0)
      tf.set_random_seed(0)

      adv_image = image.copy()
      label = model.get_pred(image)
      l = model.get_loss(image, label)
      print(filename, 'original prediction:', label, 'loss:', l)

      lr = learning_rate
      last_loss = []
      total_q = 0
      ite = 0

      while total_q <= FLAGS.max_queries:
        total_q += 1

        true = np.squeeze(model.get_grad(adv_image, label))
        print("Grad norm", np.sqrt(np.sum(true*true)))

        if ite % 2 == 0 and sigma != FLAGS.sigma:
          print("checking if sigma could be set to be 1e-4")
          rand = np.random.normal(size=adv_image.shape)
          rand = rand / np.maximum(1e-12, np.sqrt(np.mean(np.square(rand))))
          rand_loss = model.get_loss(adv_image + FLAGS.sigma * rand, label)
          total_q += 1
          rand = np.random.normal(size=adv_image.shape)
          rand = rand / np.maximum(1e-12, np.sqrt(np.mean(np.square(rand))))
          rand_loss2 = model.get_loss(adv_image + FLAGS.sigma * rand, label)
          total_q += 1
          if (rand_loss - l)[0] != 0 and (rand_loss2 - l)[0] != 0:
            print("set sigma back to 1e-4")
            sigma = FLAGS.sigma

        if FLAGS.method != 'uniform':
          prior = np.squeeze(model_s.get_grad(adv_image, label))
          alpha = np.sum(true*prior) / np.maximum(1e-12, np.sqrt(np.sum(true*true) * np.sum(prior*prior)))
          print("alpha =", alpha)
          prior = prior / np.maximum(1e-12, np.sqrt(np.mean(np.square(prior))))

        if FLAGS.method == 'biased':
          start_iter = 3
          if ite % 10 == 0 or ite == start_iter:
            # Estimate norm of true gradient
            s = 10
            pert = np.random.normal(size=(s,) + adv_image.shape)
            for i in range(s):
              pert[i] = pert[i] / np.maximum(1e-12, np.sqrt(np.mean(np.square(pert[i]))))
            eval_points = adv_image + sigma * pert
            losses = model.get_loss(eval_points, np.repeat(label, s))
            total_q += s
            norm_square = np.average(((losses - l) / sigma) ** 2)

          while True:
            prior_loss = model.get_loss(adv_image + sigma * prior, label)
            total_q += 1
            diff_prior = (prior_loss - l)[0]
            if diff_prior == 0:
              sigma *= 2
              print("multiply sigma by 2")
            else:
              break

          est_alpha = diff_prior / sigma / np.maximum(np.sqrt(np.sum(np.square(prior)) * norm_square), 1e-12)
          print("Estimated alpha =", est_alpha)
          alpha = est_alpha
          if alpha < 0:
            prior = -prior
            alpha = -alpha

        q = FLAGS.samples_per_draw
        n = image_height * image_width * 3
        d = 50*50*3
        gamma = 3.5
        A_square = d / n * gamma

        return_prior = False
        if FLAGS.method == 'biased':
          if FLAGS.dataprior:
            best_lambda = A_square * (A_square - alpha ** 2 * (d + 2 * q - 2)) / (
                    A_square ** 2 + alpha ** 4 * d ** 2 - 2 * A_square * alpha ** 2 * (q + d * q - 1))
          else:
            best_lambda = (1 - alpha ** 2) * (1 - alpha ** 2 * (n + 2 * q - 2)) / (
                    alpha ** 4 * n * (n + 2 * q - 2) - 2 * alpha ** 2 * n * q + 1)
          print('best_lambda = ', best_lambda)
          if best_lambda < 1 and best_lambda > 0:
            lmda = best_lambda
          else:
            if alpha ** 2 * (n + 2 * q - 2) < 1:
              lmda = 0
            else:
              lmda = 1
          if np.abs(alpha) >= 1:
            lmda = 1
          print('lambda = ', lmda)
          if lmda == 1:
            return_prior = True
        elif FLAGS.method == 'fixed_biased':
          lmda = 0.5

        if not return_prior:
          if FLAGS.dataprior:
            pert = np.random.normal(size=(q, 50, 50, 3))
            pert = np.array([cv2.resize(pert[i], adv_image.shape[:2],
                                        interpolation=cv2.INTER_NEAREST) for i in range(q)])
          else:
            pert = np.random.normal(size=(q,) + adv_image.shape)
          for i in range(q):
            if FLAGS.method == 'biased' or FLAGS.method == 'fixed_biased':
              angle_prior = np.sum(pert[i] * prior) / np.maximum(1e-12, np.sqrt(
                np.sum(pert[i] * pert[i]) * np.sum(prior * prior)))
              pert[i] = pert[i] - angle_prior * prior
              pert[i] = pert[i] / np.maximum(1e-12, np.sqrt(np.mean(np.square(pert[i]))))
              pert[i] = np.sqrt(1 - lmda) * pert[i] + np.sqrt(lmda) * prior
            else:
              pert[i] = pert[i] / np.maximum(1e-12, np.sqrt(np.mean(np.square(pert[i]))))

          while True:
            eval_points = adv_image + sigma * pert
            losses = model.get_loss(eval_points, np.repeat(label, q))
            total_q += q

            grad = (losses - l).reshape(-1,1,1,1) * pert
            grad = np.mean(grad, axis=0)
            norm_grad = np.sqrt(np.mean(np.square(grad)))
            if norm_grad == 0:
              sigma *= 5
              print("estimated grad == 0, multiply sigma by 5")
            else:
              break
          grad = grad / np.maximum(1e-12, np.sqrt(np.mean(np.square(grad))))

          def print_loss(model, direction):
            length = [1e-4, 1e-3]
            les = []
            for ss in length:
              les.append((model.get_loss(adv_image + ss * direction, label) - l)[0])
            print("losses", les)

          if FLAGS.show_loss:
            if FLAGS.method == 'biased' or FLAGS.method == 'fixed_biased':
              lprior = model.get_loss(adv_image + lr * prior, label) - l
              print_loss(model, prior)
              lgrad = model.get_loss(adv_image + lr * grad, label) - l
              print_loss(model, grad)
              print(lprior, lgrad)
        else:
          grad = prior

        print("angle =", np.sum(true*grad) / np.maximum(1e-12, np.sqrt(np.sum(true*true) * np.sum(grad*grad))))

        if FLAGS.norm == 'l2':
          adv_image = adv_image + lr * grad / np.maximum(1e-12, np.sqrt(np.mean(np.square(grad))))
          norm = max(1e-12, np.linalg.norm(adv_image - image))
          factor = min(1, eps / norm)
          adv_image = image + (adv_image - image) * factor
        else:
          adv_image = adv_image + lr * np.sign(grad)
          adv_image = np.clip(adv_image, image - eps, image + eps)
        adv_image = np.clip(adv_image, 0, 1)

        adv_label = model.get_pred(adv_image)
        l = model.get_loss(adv_image, label)

        print('queries:', total_q, 'loss:', l, 'learning rate:', lr, 'sigma:', sigma, 'prediction:', adv_label,
          'distortion:', np.max(np.abs(adv_image - image)), np.linalg.norm(adv_image - image))

        ite += 1

        if adv_label != label:
          print('Stop at queries:', total_q)
          success += 1
          queries.append(total_q)
          imsave(os.path.join(FLAGS.output_dir, filename), adv_image)
          output_logging.write(filename + ' succeed; queries: ' + str(total_q) + '\n')
          break
      else:
        imsave(os.path.join(FLAGS.output_dir, filename), adv_image)
        output_logging.write(filename + ' fail.\n')

    total = FLAGS.number_images
    print('Success rate:', success / total, 'Queries', queries)
    output_logging.write('Success rate: ' + str(success / total)
                         + ', Queries on success: ' + str(np.mean(queries)))

    output_logging.close()


if __name__ == '__main__':
  tf.app.run()
