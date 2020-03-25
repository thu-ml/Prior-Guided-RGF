import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import os

# For JPEG
from PIL import Image
from scipy.misc import imread

# For Random
import random

# For Denoiser
import torch
import torch.autograd as autograd
import torch.nn.functional as F

from torch_nets.res152_wide import get_model as get_model1
from torch_nets.inres import get_model as  get_model2
from torch_nets.v3 import get_model as get_model3

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

class InceptionV3:
  def __init__(self):
    from nets import inception_v3

    self.image_size = 299
    self.num_classes = 1001
    self.predictions_is_correct = True
    self.use_larger_step_size = False
    self.use_smoothed_grad = False

    # For dataprior attacks. gamma = A^2 * D / d in the paper
    self.gamma = 3.5

    batch_shape = [None, self.image_size, self.image_size, 3]
    self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
    self.target_label = tf.placeholder(tf.int32, shape=[None])
    target_onehot = tf.one_hot(self.target_label, self.num_classes)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
      logits, end_points = inception_v3.inception_v3(
        self.x_input, num_classes=self.num_classes, is_training=False)

    self.predicted_labels = tf.argmax(end_points['Predictions'], 1)
    self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_onehot, logits=logits)
    self.grad = 2*tf.gradients(self.loss, self.x_input)[0]

    saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
    self.sess = tf.get_default_session()
    saver.restore(self.sess, 'inception_v3.ckpt')

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

class VGG16:
  def __init__(self):
    from nets import vgg

    self.image_size = 224
    self.num_classes = 1000
    self.predictions_is_correct = False
    self.use_larger_step_size = False
    self.use_smoothed_grad = False

    # For dataprior attacks. gamma = A^2 * D / d in the paper
    self.gamma = 4.5

    batch_shape = [None, self.image_size, self.image_size, 3]
    self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
    self.target_label = tf.placeholder(tf.int32, shape=[None])
    target_onehot = tf.one_hot(self.target_label, self.num_classes)

    with slim.arg_scope(vgg.vgg_arg_scope()):
      logits, end_points = vgg.vgg_16(
        self.x_input, num_classes=self.num_classes, is_training=False)

    self.predicted_labels = tf.argmax(end_points['vgg_16/fc8'], 1)
    #logits -= tf.reduce_min(logits)
    #real = tf.reduce_max(logits * target_onehot, 1)
    #other = tf.reduce_max(logits * (1 - target_onehot), 1)
    #self.loss = other - real
    self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_onehot, logits=logits)
    self.grad = 255.0 * tf.gradients(self.loss, self.x_input)[0]

    saver = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))
    self.sess = tf.get_default_session()
    saver.restore(self.sess, 'vgg_16.ckpt')

  def get_loss(self, images, labels):
    if len(images.shape) == 3:
      images = images[np.newaxis]

    images = images * 255.0
    images[:,:,:,0] -= _R_MEAN
    images[:,:,:,1] -= _G_MEAN
    images[:,:,:,2] -= _B_MEAN

    return self.sess.run(self.loss,
      feed_dict={self.x_input: images, self.target_label: labels})

  def get_grad(self, images, labels):
    if len(images.shape) == 3:
      images = images[np.newaxis]

    images = images * 255.0
    images[:,:,:,0] -= _R_MEAN
    images[:,:,:,1] -= _G_MEAN
    images[:,:,:,2] -= _B_MEAN

    return self.sess.run(self.grad,
      feed_dict={self.x_input: images, self.target_label: labels})

  def get_pred(self, images):
    if len(images.shape) == 3:
      images = images[np.newaxis]

    images = images * 255.0
    images[:,:,:,0] -= _R_MEAN
    images[:,:,:,1] -= _G_MEAN
    images[:,:,:,2] -= _B_MEAN

    return self.sess.run(self.predicted_labels,
      feed_dict={self.x_input: images})

class ResNet50:
  def __init__(self):
    from nets import resnet_v1

    self.image_size = 224
    self.num_classes = 1000
    self.predictions_is_correct = False
    self.use_larger_step_size = False
    self.use_smoothed_grad = False

    # For dataprior attacks. gamma = A^2 * D / d in the paper
    self.gamma = 2.7

    batch_shape = [None, self.image_size, self.image_size, 3]
    self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
    self.target_label = tf.placeholder(tf.int32, shape=[None])
    target_onehot = tf.one_hot(self.target_label, self.num_classes)

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      logits, end_points = resnet_v1.resnet_v1_50(
        self.x_input, num_classes=self.num_classes, is_training=False)

    self.predicted_labels = tf.argmax(end_points['predictions'], 1)
    #logits -= tf.reduce_min(logits)
    #real = tf.reduce_max(logits * target_onehot, 1)
    #other = tf.reduce_max(logits * (1 - target_onehot), 1)
    #self.loss = other - real
    self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_onehot, logits=logits)
    self.grad = 255.0 * tf.gradients(self.loss, self.x_input)[0]

    saver = tf.train.Saver(slim.get_model_variables(scope='resnet_v1'))
    self.sess = tf.get_default_session()
    saver.restore(self.sess, 'resnet_v1_50.ckpt')

  def get_loss(self, images, labels):
    if len(images.shape) == 3:
      images = images[np.newaxis]

    images = images * 255.0
    images[:,:,:,0] -= _R_MEAN
    images[:,:,:,1] -= _G_MEAN
    images[:,:,:,2] -= _B_MEAN

    return self.sess.run(self.loss,
      feed_dict={self.x_input: images, self.target_label: labels})

  def get_grad(self, images, labels):
    if len(images.shape) == 3:
      images = images[np.newaxis]

    images = images * 255.0
    images[:,:,:,0] -= _R_MEAN
    images[:,:,:,1] -= _G_MEAN
    images[:,:,:,2] -= _B_MEAN

    return self.sess.run(self.grad,
      feed_dict={self.x_input: images, self.target_label: labels})

  def get_pred(self, images):
    if len(images.shape) == 3:
      images = images[np.newaxis]

    images = images * 255.0
    images[:,:,:,0] -= _R_MEAN
    images[:,:,:,1] -= _G_MEAN
    images[:,:,:,2] -= _B_MEAN

    return self.sess.run(self.predicted_labels,
      feed_dict={self.x_input: images})

class Denoiser:
  def __init__(self):
    self.image_size = 299
    self.num_classes = 1000
    self.predictions_is_correct = False
    self.use_larger_step_size = False
    self.use_smoothed_grad = True

    # For dataprior attacks. gamma = A^2 * D / d in the paper
    self.gamma = 4.0

    self.mean_torch = autograd.Variable(torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1,3,1,1]).astype('float32')).cuda(), volatile=True)
    self.std_torch = autograd.Variable(torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1,3,1,1]).astype('float32')).cuda(), volatile=True)
    self.mean_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda(), volatile=True)
    self.std_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda(), volatile=True)
    
    config, resmodel = get_model1()
    config, inresmodel = get_model2()
    config, incepv3model = get_model3()
    self.net1 = resmodel.net    
    self.net2 = inresmodel.net
    self.net3 = incepv3model.net

    checkpoint = torch.load('denoise_res_015.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        resmodel.load_state_dict(checkpoint['state_dict'])
    else:
        resmodel.load_state_dict(checkpoint)

    checkpoint = torch.load('denoise_inres_014.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        inresmodel.load_state_dict(checkpoint['state_dict'])
    else:
        inresmodel.load_state_dict(checkpoint)

    checkpoint = torch.load('denoise_incepv3_012.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        incepv3model.load_state_dict(checkpoint['state_dict'])
    else:
        incepv3model.load_state_dict(checkpoint)

    inresmodel = inresmodel.cuda()
    resmodel = resmodel.cuda()
    incepv3model = incepv3model.cuda()
    inresmodel.eval()
    resmodel.eval()
    incepv3model.eval()

  def forward(self, input):
    images = input.permute(0, 3, 1, 2)
    input_tf = (images - self.mean_tf) / self.std_tf
    input_torch = (images - self.mean_torch) / self.std_torch

    logits1 = self.net1(input_torch,True)[-1]
    logits2 = self.net2(input_tf,True)[-1]
    logits3 = self.net3(input_tf,True)[-1]

    logits = (logits1 + logits2 + logits3) / 3
    return logits

  def get_loss(self, images, labels):
    if len(images.shape) == 3:
      images = images[np.newaxis]
    
    # to save GPU memory usage
    if images.shape[0] > 25:
      loss1 = self.get_loss(images[:25], labels[:25])
      loss2 = self.get_loss(images[25:], labels[25:])
      return np.concatenate([loss1, loss2], axis=0)

    with torch.no_grad():
      images = torch.tensor(images, dtype=torch.float)
      images = images.cuda()
      logits = self.forward(images)

      one_hot = torch.zeros([labels.shape[0], 1000])
      for i in range(labels.shape[0]):
        one_hot[i, labels[i]] = 1
      one_hot = one_hot.cuda()
      loss = torch.sum(- one_hot * F.log_softmax(logits, -1), -1)
    return loss.data.cpu().numpy()

  def get_grad(self, images, labels):
    if len(images.shape) == 3:
      images = images[np.newaxis]
    
    # to save GPU memory usage
    if images.shape[0] > 25:
      grad1 = self.get_grad(images[:25], labels[:25])
      grad2 = self.get_grad(images[25:], labels[25:])
      return np.concatenate([grad1, grad2], axis=0)

    images = torch.tensor(images, dtype=torch.float)
    images = images.cuda()
    images.requires_grad_()
    one_hot = torch.zeros([labels.shape[0], 1000])
    for i in range(labels.shape[0]):
      one_hot[i, labels[i]] = 1
    one_hot = one_hot.cuda()
    logits = self.forward(images)
    loss = torch.sum(-one_hot * F.log_softmax(logits, -1))
    loss.backward()
    grad = images.grad
    return grad.detach().cpu().numpy()

  def get_pred(self, images):
    if len(images.shape) == 3:
      images = images[np.newaxis]

    with torch.no_grad():
      images = torch.tensor(images, dtype=torch.float)
      images = images.cuda()
      logits = self.forward(images)

    return logits.max(1)[1].data.cpu().numpy()

class JPEG:
  def __init__(self):
    from nets import inception_v3

    self.image_size = 299
    self.num_classes = 1001
    self.predictions_is_correct = False
    self.use_larger_step_size = True
    self.use_smoothed_grad = True

    # For dataprior attacks. gamma = A^2 * D / d in the paper
    self.gamma = 4.0

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
    saver.restore(self.sess, 'inception_v3.ckpt')

  def get_loss(self, imgs, labels):
    images = imgs.copy()
    if len(images.shape) == 3:
      images = images[np.newaxis]
    for i in range(images.shape[0]):
      img = Image.fromarray((images[i] * 255.0).astype(np.uint8), 'RGB')
      img.save('temp.png', "JPEG", quality=75)
      images[i] = imread('temp.png').astype(np.float) / 255.0
    images = images * 2.0 - 1.0

    return self.sess.run(self.loss,
      feed_dict={self.x_input: images, self.target_label: labels})

  def get_pred(self, imgs):
    images = imgs.copy()
    if len(images.shape) == 3:
      images = images[np.newaxis]
    for i in range(images.shape[0]):
      img = Image.fromarray((images[i] * 255.0).astype(np.uint8), 'RGB')
      img.save('temp.png', "JPEG", quality=75)
      images[i] = imread('temp.png').astype(np.float) / 255.0
    images = images * 2.0 - 1.0

    return self.sess.run(self.predicted_labels,
      feed_dict={self.x_input: images})

def padding_layer_iyswim(inputs, shape, name=None):
  h_start = shape[0]
  w_start = shape[1]
  output_short = shape[2]
  input_shape = tf.shape(inputs)
  input_short = tf.reduce_min(input_shape[1:3])
  input_long = tf.reduce_max(input_shape[1:3])
  output_long = tf.to_int32(tf.ceil(
    1. * tf.to_float(output_short) * tf.to_float(input_long) / tf.to_float(input_short)))
  output_height = tf.to_int32(input_shape[1] >= input_shape[2]) * output_long +\
    tf.to_int32(input_shape[1] < input_shape[2]) * output_short
  output_width = tf.to_int32(input_shape[1] >= input_shape[2]) * output_short +\
    tf.to_int32(input_shape[1] < input_shape[2]) * output_long
  return tf.pad(inputs, tf.to_int32(tf.stack([[0, 0], [h_start, output_height - h_start - input_shape[1]], [w_start, output_width - w_start - input_shape[2]], [0, 0]])), name=name)

class Random:
  def __init__(self):
    from nets import inception_v3

    self.image_size = 299
    self.num_classes = 1001
    self.predictions_is_correct = False
    self.use_larger_step_size = True
    self.use_smoothed_grad = False

    # For dataprior attacks. gamma = A^2 * D / d in the paper
    self.gamma = 4.0

    batch_shape = [None, self.image_size, self.image_size, 3]
    self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
    self.img_resize_tensor = tf.placeholder(tf.int32, [2])
    x_input_resize = tf.image.resize_images(self.x_input, self.img_resize_tensor, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    self.shape_tensor = tf.placeholder(tf.int32, [3])
    padded_input = padding_layer_iyswim(x_input_resize, self.shape_tensor)

    self.image_resize = 331
    padded_input.set_shape((None, self.image_resize, self.image_resize, 3))

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
      logits, end_points = inception_v3.inception_v3(
        padded_input, num_classes=self.num_classes, is_training=False)

    self.pred = end_points['Predictions']

    saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
    self.sess = tf.get_default_session()
    saver.restore(self.sess, 'inception_v3.ckpt')

  def get_loss(self, images, labels):
    if len(images.shape) == 3:
      images = images[np.newaxis]
    images = images * 2.0 - 1.0

    final_preds = np.zeros([images.shape[0], self.num_classes, 10])
    for i in range(10):
      if np.random.randint(0, 2, size=1) == 1:
        images = images[:, :, ::-1, :]
      resize_shape_ = np.random.randint(310, 331)
      pred = self.sess.run(self.pred, feed_dict={self.x_input: images, 
                                                 self.img_resize_tensor: [resize_shape_]*2,
                                                 self.shape_tensor: np.array([random.randint(0, self.image_resize - resize_shape_), 
                                                                              random.randint(0, self.image_resize - resize_shape_), 
                                                                              self.image_resize])})
      final_preds[..., i] = pred
    final_probs = np.mean(final_preds, axis=-1)
    loss = -np.log(np.array([final_probs[i, labels[i]] for i in range(labels.shape[0])]))
    return loss

  def get_pred(self, images):
    if len(images.shape) == 3:
      images = images[np.newaxis]
    images = images * 2.0 - 1.0

    final_preds = np.zeros([images.shape[0], self.num_classes, 10])
    for i in range(10):
      if np.random.randint(0, 2, size=1) == 1:
        images = images[:, :, ::-1, :]
      resize_shape_ = np.random.randint(310, 331)
      pred = self.sess.run(self.pred, feed_dict={self.x_input: images, 
                                                 self.img_resize_tensor: [resize_shape_]*2,
                                                 self.shape_tensor: np.array([random.randint(0, self.image_resize - resize_shape_), 
                                                                              random.randint(0, self.image_resize - resize_shape_), 
                                                                              self.image_resize])})
      final_preds[..., i] = pred
    final_probs = np.sum(final_preds, axis=-1)
    labels = np.argmax(final_probs, 1)
    return labels