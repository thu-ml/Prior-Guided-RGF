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

tf.flags.DEFINE_enum(
  'model', 'inception-v3', ['inception-v3', 'vgg-16', 'resnet-50',
                            'jpeg', 'random', 'denoiser'],
  'Model to be attacked.')

tf.flags.DEFINE_enum(
  'norm', 'l2', ['l2', 'linfty'], 'The norm used in the attack.')

tf.flags.DEFINE_enum(
  'method', 'biased', ['uniform', 'biased', 'average',
                       'fixed_biased', 'fixed_average'],
  'Methods used in the attack.')

tf.flags.DEFINE_float(
  'fixed_const', 0.5, 'Value of lambda used in fixed_biased,'
                      ' or value of mu used in fixed_average')

tf.flags.DEFINE_boolean(
  'dataprior', False, 'Whether to use data prior in the attack.')

tf.flags.DEFINE_boolean(
  'show_true', False, 'Whether to print statistics about the true gradient.')

tf.flags.DEFINE_boolean(
  'show_loss', False, 'Whether to print loss in some given step sizes.')

tf.flags.DEFINE_string(
  'input_dir', 'images', 'Input directory with images.')

tf.flags.DEFINE_string(
  'output_dir', '', 'Output directory to save results.')

tf.flags.DEFINE_integer(
  'samples_per_draw', 50, 'Number of samples to estimate the gradient.')

tf.flags.DEFINE_integer(
  'number_images', 1000, 'Number of images for evaluation.')

tf.flags.DEFINE_integer(
  'max_queries', 10000, 'Maximum number of queries.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, image_size):
  for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png')))[:FLAGS.number_images]:
    image = imread(filepath, mode='RGB')
    if image_size != 299:
      image = imresize(image, [image_size, image_size])
    image = image.astype(np.float) / 255.0

    yield os.path.basename(filepath), image


def main(_):
  print("FLAGS values:", tf.app.flags.FLAGS.flag_values_dict())
  tf.logging.set_verbosity(tf.logging.INFO)

  config = tf.ConfigProto()
  if FLAGS.model == 'denoiser':
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
  else:
    config.gpu_options.allow_growth = True

  if FLAGS.model == 'denoiser':
    from models import Denoiser
    model = Denoiser()

  with tf.Graph().as_default():
    # Prepare graph
    with tf.Session(config=config).as_default():
      if FLAGS.model == 'inception-v3':
        from models import InceptionV3
        model = InceptionV3()
      elif FLAGS.model == 'vgg-16':
        from models import VGG16
        model = VGG16()
      elif FLAGS.model == 'resnet-50':
        from models import ResNet50
        model = ResNet50()
      elif FLAGS.model == 'jpeg':
        from models import JPEG
        model = JPEG()
      elif FLAGS.model == 'random':
        from models import Random
        model = Random()

      if FLAGS.method != 'uniform':
        from surrogate_model import ResNet152
        model_s = ResNet152(source_image_size=model.image_size, use_smoothed_grad=model.use_smoothed_grad)

  image_size = model.image_size
  
  # ---Setting hyperparameters---
  if FLAGS.norm == 'l2':
    epsilon = 1e-3
    eps = np.sqrt(epsilon * image_size * image_size * 3)
    learning_rate = 2.0 / np.sqrt(image_size * image_size * 3)
  else:
    epsilon = 0.05
    eps = epsilon
    learning_rate = 0.005
  if model.use_larger_step_size:
    ini_sigma = 1e-3
  else:
    ini_sigma = 1e-4
  # -----------------------------

  if not model.predictions_is_correct:
    l = open(os.path.join(FLAGS.input_dir, 'labels')).readlines()
    gts = {}
    for i in l:
      i = i.strip().split(' ')
      gts[i[0]] = int(i[1]) + (model.num_classes - 1000)

  success = 0
  queries = []
  correct = 0

  names_images = load_images(FLAGS.input_dir, model.image_size)
  for filename, image in names_images:
    output_logging = open(os.path.join(FLAGS.output_dir, 'logging'), 'a')
    sigma = ini_sigma
    np.random.seed(0)
    tf.set_random_seed(0)

    adv_image = image.copy()
    label = model.get_pred(image)
    l = model.get_loss(image, label)
    print(filename, 'original prediction:', label, 'loss:', l)
    if not model.predictions_is_correct:
      correct += (label[0] == gts[filename])
      if label[0] != gts[filename]:
        output_logging.write(filename + ' original misclassified.\n')
        output_logging.close()
        continue

    lr = learning_rate
    last_loss = []
    total_q = 0
    ite = 0

    while total_q <= FLAGS.max_queries:
      total_q += 1

      if FLAGS.show_true and hasattr(model, 'get_grad'):
        true = np.squeeze(model.get_grad(adv_image, label))
        print("Grad norm", np.sqrt(np.sum(true*true)))

      if ite % 2 == 0 and sigma != ini_sigma:
        print("sigma has been increased before; checking if sigma could be set back to ini_sigma")
        rand = np.random.normal(size=adv_image.shape)
        rand = rand / np.maximum(1e-12, np.sqrt(np.mean(np.square(rand))))
        rand_loss = model.get_loss(adv_image + ini_sigma * rand, label)
        total_q += 1
        rand = np.random.normal(size=adv_image.shape)
        rand = rand / np.maximum(1e-12, np.sqrt(np.mean(np.square(rand))))
        rand_loss2 = model.get_loss(adv_image + ini_sigma * rand, label)
        total_q += 1
        if (rand_loss - l)[0] != 0 and (rand_loss2 - l)[0] != 0:
          print("set sigma back to ini_sigma")
          sigma = ini_sigma

      if FLAGS.method != 'uniform':
        if model.num_classes < model_s.num_classes:
          s_label = label + 1
        elif model.num_classes > model_s.num_classes:
          s_label = label - 1
        else:
          s_label = label
        prior = np.squeeze(model_s.get_grad(adv_image, s_label))
        if FLAGS.show_true and hasattr(model, 'get_grad'):
          alpha = np.sum(true*prior) / np.maximum(1e-12, np.sqrt(np.sum(true*true) * np.sum(prior*prior)))
          print("alpha =", alpha)
        prior = prior / np.maximum(1e-12, np.sqrt(np.mean(np.square(prior))))

      if FLAGS.method in ['biased', 'average']:
        start_iter = 3
        if ite % 10 == 0 or ite == start_iter:
          # Estimate norm of true gradient periodically when ite == 0/10/20...;
          # since gradient norm may change fast in the early iterations, we also
          # estimate the gradient norm when ite == 3.
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
            # Avoid the numerical issue in finite difference
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
      n = image_size * image_size * 3
      d = 50*50*3
      gamma = 3.5
      A_square = d / n * gamma

      return_prior = False
      if FLAGS.method == 'average':
        if FLAGS.dataprior:
          alpha_nes = np.sqrt(A_square * q / (d + q + 1))
        else:
          alpha_nes = np.sqrt(q / (n + q + 1))
        if alpha >= 1.414 * alpha_nes:
          return_prior = True
      elif FLAGS.method == 'biased':
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
        lmda = FLAGS.fixed_const

      if not return_prior:
        if FLAGS.dataprior:
          pert = np.random.normal(size=(q, 50, 50, 3))
          pert = np.array([cv2.resize(pert[i], adv_image.shape[:2],
                                      interpolation=cv2.INTER_NEAREST) for i in range(q)])
        else:
          pert = np.random.normal(size=(q,) + adv_image.shape)
        for i in range(q):
          if FLAGS.method in ['biased', 'fixed_biased']:
            pert[i] = pert[i] - np.sum(pert[i] * prior) * prior / np.maximum(1e-12,
              np.sum(prior * prior))
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
            # Avoid the numerical issue in finite difference
            sigma *= 5
            print("estimated grad == 0, multiply sigma by 5")
          else:
            break
        grad = grad / np.maximum(1e-12, np.sqrt(np.mean(np.square(grad))))

        if FLAGS.method == 'average':
          while True:
            diff_prior = (model.get_loss(adv_image + sigma * prior, label) - l)[0]
            total_q += 1
            diff_nes = (model.get_loss(adv_image + sigma * grad, label) - l)[0]
            total_q += 1
            diff_prior = max(0, diff_prior)
            if diff_prior == 0 and diff_nes == 0:
              sigma *= 2
              print("multiply sigma by 2")
            else:
              break
          final = prior * diff_prior + grad * diff_nes
          final = final / np.maximum(1e-12, np.sqrt(np.mean(np.square(final))))
          print("diff_prior = {}, diff_nes = {}".format(diff_prior, diff_nes))
        elif FLAGS.method == 'fixed_average':
          diff_prior = (model.get_loss(adv_image + sigma * prior, label) - l)[0]
          total_q += 1
          if diff_prior < 0:
            prior = -prior
          final = FLAGS.fixed_const * prior + (1 - FLAGS.fixed_const) * grad
          final = final / np.maximum(1e-12, np.sqrt(np.mean(np.square(final))))
        else:
          final = grad

        def print_loss(model, direction):
          length = [1e-4, 1e-3]
          les = []
          for ss in length:
            les.append((model.get_loss(adv_image + ss * direction, label) - l)[0])
          print("losses", les)

        if FLAGS.show_loss:
          if FLAGS.method in ['average', 'fixed_average']:
            lprior = model.get_loss(adv_image + lr * prior, label) - l
            print_loss(model, prior)
            lgrad = model.get_loss(adv_image + lr * grad, label) - l
            print_loss(model, grad)
            lfinal = model.get_loss(adv_image + lr * final, label) - l
            print_loss(model, final)
            print(lprior, lgrad, lfinal)
          elif FLAGS.method in ['biased', 'fixed_biased']:
            lprior = model.get_loss(adv_image + lr * prior, label) - l
            print_loss(model, prior)
            lgrad = model.get_loss(adv_image + lr * grad, label) - l
            print_loss(model, grad)
            print(lprior, lgrad)        
      else:
        final = prior

      if FLAGS.show_true and hasattr(model, 'get_grad'):
        if FLAGS.method in ['average', 'fixed_average'] and not return_prior:
          print("NES angle =", np.sum(true*grad) / np.maximum(1e-12, np.sqrt(np.sum(true*true) * np.sum(grad*grad))))
        print("angle =", np.sum(true*final) / np.maximum(1e-12, np.sqrt(np.sum(true*true) * np.sum(final*final))))

      if FLAGS.norm == 'l2':
        adv_image = adv_image + lr * final / np.maximum(1e-12, np.sqrt(np.mean(np.square(final))))
        norm = max(1e-12, np.linalg.norm(adv_image - image))
        factor = min(1, eps / norm)
        adv_image = image + (adv_image - image) * factor
      else:
        adv_image = adv_image + lr * np.sign(final)
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
    output_logging.close()

  if model.predictions_is_correct:
    total = FLAGS.number_images
  else:
    total = correct
  print('Success rate:', success / total, 'Queries', queries)
  output_logging = open(os.path.join(FLAGS.output_dir, 'logging'), 'a')
  output_logging.write('Success rate: ' + str(success / total)
                        + ', Queries on success: ' + str(np.mean(queries)))

  output_logging.close()


if __name__ == '__main__':
  tf.app.run()
