# Copyright 2022 Google LLC
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from absl import flags
import math

FLAGS = flags.FLAGS

def norm_layer():
  if FLAGS.global_bn:
    return tf.keras.layers.experimental.SyncBatchNormalization()
  else:
    return tf.keras.layers.BatchNormalization()


class FCN_backbone(tf.keras.Model):

  def __init__(self,
               hidden_dims,
               use_bn = True,
               use_relu=True):
    super(FCN_backbone, self).__init__()
    self.fcn_layers = []
    for dim in hidden_dims:
      self.fcn_layers.append(tf.keras.layers.Dense(dim))
      if use_bn:
        self.fcn_layers.append(norm_layer())
      if use_relu:
        self.fcn_layers.append(tf.keras.layers.ReLU())
    self.outputs = []

  def call(self, x, training):
    self.outputs = []
    feats = []
    # final_shape = 1
    # for i in range(tf.rank(x)):
    #   if i > 0:
    #     final_shape *= tf.shape(x)[i]
    # x = tf.reshape(x, (tf.shape(x)[0], final_shape))
    #tf.print('x shape:', tf.shape(x))
    x = tf.reshape(x, (tf.shape(x)[0], -1))
    for ind, layer in enumerate(self.fcn_layers):
      x = layer(x, training=training)
      if ind%3 == 0 and ((ind//3 in FLAGS.HSIC_feature_layers) or FLAGS.use_all_features_HSIC):
        self.outputs.append(x)
        feats.append(x)

    return x, feats

class CNN_backbone(tf.keras.Model):

  def __init__(self,
               filters,
               kernel_sizes,
               strides,
               use_bn = True,
               use_relu=True):
    super(CNN_backbone, self).__init__()
    self.cnn_layers = []
    for ind, filter in enumerate(filters):
      self.cnn_layers.append(tf.keras.layers.Conv2D(filter, kernel_sizes[ind], strides[ind], padding='same'))
      if use_bn:
        self.cnn_layers.append(norm_layer())
      if use_relu:
        self.cnn_layers.append(tf.keras.layers.ReLU())
    self.output_bf_avg_pool = None

  def call(self, x, training):
    feats = []
    for ind, layer in enumerate(self.cnn_layers):
      x = layer(x, training=training)
      if ind%3 == 0 and ((ind//3 in FLAGS.HSIC_feature_layers) or FLAGS.use_all_features_HSIC):
        if FLAGS.use_GAP_HSIC_features:
          feats.append(tf.reduce_mean(x, [1, 2]))
        elif FLAGS.use_random_projections:
          temp_input = tf.reshape(x, (tf.shape(x)[0], -1))
          final_shape = tf.constant([FLAGS.random_proj_dim], dtype=tf.int32)
          final_shape = tf.concat([final_shape, tf.shape(x)[1:]], axis=0)
          z = tf.random.normal(final_shape)
          z = tf.reshape(z, (FLAGS.random_proj_dim, -1))
          z = tf.transpose(z)
          q, _ = tf.linalg.qr(z)
          #z = z/tf.norm(z, axis=1, keepdims=True)
          #feats.append(tf.matmul(temp_input, tf.transpose(z)))
          feats.append(tf.matmul(temp_input, q))
        else:
          feats.append(x)

    self.output_bf_avg_pool = x
    x = tf.reduce_mean(x, axis=[1,2])
    return x, feats

class multi_base_model(tf.keras.Model):

  def __init__(self,
               base_models,
               num_classes):
    super(multi_base_model, self).__init__()
    self.base_models = base_models
    self.out = tf.keras.layers.Dense(num_classes)

  def call(self, x, training, only_head=False, only_linear_head=False, init=False, project_out_w=False, project_out_mat=None,
           gauss_noise_feats=False, sigma=1.0, rand_sigma=False):
    outs = []
    for model_ind, model in enumerate(self.base_models):
      if FLAGS.use_pretrained:
        if gauss_noise_feats:
          out = model.layers[0](x, training = training and not only_head)
          if model_ind == 0:
            if rand_sigma:
              std_devs = sigma * tf.random.uniform([tf.shape(out)[0]])
              final_shape = tf.constant([-1], dtype=tf.int32)
              final_shape = tf.concat([final_shape, tf.ones([tf.rank(out)-1], dtype=tf.int32)], axis=0)
              z = tf.reshape(std_devs, final_shape) * tf.random.normal(tf.shape(out))
            else:
              final_shape = tf.constant([-1], dtype=tf.int32)
              final_shape = tf.concat([final_shape, tf.ones([tf.rank(out)-1], dtype=tf.int32)], axis=0)
              z = sigma * tf.random.normal(tf.shape(out))
          out = out + tf.cast(z, out.dtype)
          for ind, layer in enumerate(model.layers):
            if ind > 0:
              out = layer(out, training = training and not only_head)
        else:
          out = model(x, training = training and not only_head)
        feat = [out]
      else:
        out, feat = model(x, training = training and not only_head)
      outs.append(out)

    inp = tf.concat(outs, axis=1)
    fin = self.out(inp)
    return inp, fin, feat

class base_multi_head_model(tf.keras.Model):

  def __init__(self,
               base_model,
               head_models,
               num_classes):
    super(base_multi_head_model, self).__init__()
    self.base_model = base_model
    self.head_models = head_models
    self.out = tf.keras.layers.Dense(num_classes)

  def call(self, x, training, only_head=False, only_linear_head=False, init=False, project_out_w=False, project_out_mat=None,
           gauss_noise_feats=False, sigma=1.0, rand_sigma=False):
    outs = []
    base_out = self.base_model(x, training = training and not only_head)
    if FLAGS.use_pretrained:
      if gauss_noise_feats:
        if rand_sigma:
          std_devs = sigma * tf.random.uniform([tf.shape(base_out)[0]])
          final_shape = tf.constant([-1], dtype=tf.int32)
          final_shape = tf.concat([final_shape, tf.ones([tf.rank(base_out)-1], dtype=tf.int32)], axis=0)
          z = tf.reshape(std_devs, final_shape) * tf.random.normal(tf.shape(base_out))
        else:
          final_shape = tf.constant([-1], dtype=tf.int32)
          final_shape = tf.concat([final_shape, tf.ones([tf.rank(base_out)-1], dtype=tf.int32)], axis=0)
          z = sigma * tf.random.normal(tf.shape(base_out))
        base_out = base_out + z
    for model_ind, model in enumerate(self.head_models):
      out = model(base_out, training = training and not only_head)
      feat = [out]
      outs.append(out)

    inp = tf.concat(outs, axis=1)
    fin = self.out(inp)
    return inp, fin, feat


class head_model(tf.keras.Model):

  def __init__(self,
               base_model,
               num_classes,
               proj_dim,
               proj_layer,
               head_dims,
               use_proj=True,
               use_bn=True,
               use_relu=True,
               dropout_rate=0.0):
    super(head_model, self).__init__()
    self.base_model = base_model
    if proj_layer > len(head_dims):
      raise ValueError('proj layer must be less than head dimensions')
    self.proj_head = []
    for i in range(proj_layer):
      self.proj_head.append(tf.keras.layers.Dense(head_dims[i]))
      if use_bn:
        self.proj_head.append(norm_layer())
      if use_relu:
        self.proj_head.append(tf.keras.layers.ReLU())

    if use_proj:
      self.proj_head.append(tf.keras.layers.Dense(proj_dim))

    self.head = []
    self.head.append(tf.keras.layers.Dropout(rate=dropout_rate))
    for i in range(len(head_dims) - proj_layer):
      self.head.append(tf.keras.layers.Dense(head_dims[proj_layer + i]))
      if use_bn:
        self.head.append(norm_layer())
      if use_relu:
        self.head.append(tf.keras.layers.ReLU())

    if FLAGS.use_chizat_init:
      if len(head_dims) > 1 or num_classes > 1:
        raise ValueError('Rich regime initialization only for 1 layer net and binary classification')
      else:
        self.out = tf.keras.layers.Dense(num_classes, use_bias=False)
    else:
      self.out = tf.keras.layers.Dense(num_classes)

    self.head_dims = head_dims

  def call(self, x, training, only_head=False, only_linear_head=False, init=False, project_out_w=False, project_out_mat=None,
           gauss_noise_feats=False, sigma=1.0, rand_sigma=False):
    if FLAGS.use_pretrained:
      x = self.base_model(x, training=training and not only_head)
      if gauss_noise_feats:
        if rand_sigma:
          std_devs = sigma * tf.random.uniform([tf.shape(x)[0]])
          final_shape = tf.constant([-1], dtype=tf.int32)
          final_shape = tf.concat([final_shape, tf.ones([tf.rank(x)-1], dtype=tf.int32)], axis=0)
          z = tf.reshape(std_devs, final_shape) * tf.random.normal(tf.shape(x))
        else:
          z = sigma * tf.random.normal(tf.shape(x))
        x = x + tf.cast(z, x.dtype)
      feat = []
    else:
      x, feat = self.base_model(x, training=training and not only_head)
    for layer in self.proj_head:
      x = layer(x, training=training and not only_linear_head)
    reps = x
    if project_out_w and FLAGS.use_pretrained:
      for ind, layer in enumerate(self.head):
        if isinstance(layer, tf.keras.layers.Dense):
          for var in layer.trainable_variables:
            if 'kernel' in var.name:
              W_norm = tf.norm(var, axis=0, keepdims=True)**2
              sample_ind = tf.random.categorical(tf.math.log(W_norm), 1)[0,0]
              sample_W = tf.reshape(var[:, sample_ind], [-1,1])
              ctx = tf.distribute.get_replica_context()
              sample_W_gather = ctx.all_gather(sample_W, axis=1)
              sample_W = tf.reshape(sample_W_gather[:,0], [-1,1])
              sample_W = sample_W/tf.norm(sample_W, axis=0, keepdims=True)
              proj_mat = tf.eye(tf.shape(var)[0]) - tf.linalg.matmul(sample_W, tf.transpose(sample_W))
              x = tf.transpose(tf.linalg.matmul(proj_mat, tf.transpose(x)))
          break
    elif project_out_mat is not None:
      pinv = tf.linalg.pinv(tf.linalg.matmul(tf.transpose(project_out_mat), project_out_mat))
      proj_mat = tf.linalg.matmul(project_out_mat, tf.linalg.matmul(pinv, tf.transpose(project_out_mat)))
      proj_x = tf.linalg.matmul(proj_mat, tf.transpose(x))
      x = x - tf.transpose(proj_x)
    curr_ind = 0
    for layer in self.head:
      if FLAGS.use_chizat_init and isinstance(layer, tf.keras.layers.Dense):
        if init:
          tf.print('assigned')
          z = tf.random.normal((tf.shape(x)[1]+1, self.head_dims[0]))
          z = z/tf.norm(z, axis=0, keepdims=True)
          layer.kernel.assign(z[0:tf.shape(x)[1], :])
          layer.bias.assign(z[tf.shape(x)[1], :])
      x = layer(x, training=training and not only_linear_head)
      if FLAGS.use_pretrained:
        if isinstance(layer, tf.keras.layers.Dense):
          if curr_ind in FLAGS.HSIC_feature_layers or FLAGS.use_all_features_HSIC:
            feat.append(x)
          curr_ind += 1
    if FLAGS.use_chizat_init:
      if init:
        tf.print('out assigned')
        samples = tf.random.categorical(tf.math.log([[0.5, 0.5]]), tf.shape(x)[1])
        self.out.kernel.assign(tf.transpose(2.0*tf.cast(samples, tf.float32) - 1.0))
    x = self.out(x, training=training)

    if FLAGS.use_chizat_init:
      x = x/tf.cast(self.head_dims[0], tf.float32)
    if FLAGS.use_all_features_HSIC or -1 in FLAGS.HSIC_feature_layers:
      feat.append(x)
    return reps, x, feat


class multihead_model(tf.keras.Model):

  def __init__(self,
               base_model,
               num_classes,
               proj_dim,
               proj_layer,
               head_dims,
               num_heads,
               use_proj=True,
               use_bn=True,
               use_relu=True,
               dropout_rate=0.0,
               normalized_reps=False):
    super(multihead_model, self).__init__()
    self.base_model = base_model
    self.proj_heads = []
    self.heads = []
    self.outs = []
    self.num_heads = num_heads
    self.normalized_reps = normalized_reps

    for i in range(num_heads):
      curr_proj_head = []
      for j in range(proj_layer):
        curr_proj_head.append(tf.keras.layers.Dense(head_dims[j]))
        if use_bn:
          curr_proj_head.append(norm_layer())
        if use_relu:
          curr_proj_head.append(tf.keras.layers.ReLU())
      if use_proj:
        curr_proj_head.append(tf.keras.layers.Dense(proj_dim))
      self.proj_heads.append(tf.keras.Sequential(layers = curr_proj_head))
      curr_head = []
      curr_head.append(tf.keras.layers.Dropout(rate=dropout_rate))
      for j in range(len(head_dims) - proj_layer):
        curr_head.append(tf.keras.layers.Dense(head_dims[j + proj_layer]))
        if use_bn:
          curr_head.append(norm_layer())
        if use_relu:
          curr_head.append(tf.keras.layers.ReLU())
      self.heads.append(tf.keras.Sequential(layers = curr_head))
      self.outs.append(tf.keras.layers.Dense(num_classes))

  def call(self, x, training, only_head=False, only_linear_head=False):
    x = self.base_model(x, training=training and not only_head)

    outs = []
    reps = []
    for i in range(self.num_heads):
      y = self.proj_heads[i](x, training=training and not only_linear_head)
      if self.normalized_reps:
        y = y/tf.norm(y, axis=1, keepdims=True)
      reps.append(y)
      y = self.heads[i](y, training=training and not only_linear_head)
      y = self.outs[i](y, training = training)
      outs.append(y)
    return reps, outs

class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies a warmup schedule on a given learning rate decay schedule."""

  def __init__(self, base_learning_rate, num_examples, name=None):
    super(WarmUpAndCosineDecay, self).__init__()
    self.base_learning_rate = base_learning_rate
    self.num_examples = num_examples
    self._name = name

  def __call__(self, step):
    with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
      steps_per_epoch = (self.num_examples // FLAGS.train_batch_size) + 1
      if FLAGS.warmup_steps is not None:
        warmup_steps = FLAGS.warmup_steps
      else:
        warmup_steps = int(FLAGS.warmup_epochs * steps_per_epoch)
      if FLAGS.learning_rate_scaling == 'linear':
        scaled_lr = self.base_learning_rate * FLAGS.train_batch_size / 256.
      elif FLAGS.learning_rate_scaling == 'sqrt':
        scaled_lr = self.base_learning_rate * math.sqrt(FLAGS.train_batch_size)
      else:
        raise ValueError('Unknown learning rate scaling {}'.format(
            FLAGS.learning_rate_scaling))
      learning_rate = (
          step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

      # Cosine decay learning rate schedule
      if FLAGS.train_steps > 0:
        total_steps = FLAGS.train_steps
      else:
        total_steps = int(FLAGS.train_epochs * steps_per_epoch)

      cosine_decay = tf.keras.experimental.CosineDecay(
          scaled_lr, total_steps - warmup_steps)
      learning_rate = tf.where(step < warmup_steps, learning_rate,
                               cosine_decay(step - warmup_steps))

      return learning_rate

  def get_config(self):
    return {
        'base_learning_rate': self.base_learning_rate,
        'num_examples': self.num_examples,
    }
