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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from absl import flags

FLAGS = flags.FLAGS

#Expected gradients
def expected_gradients_full(inputs, references, model, k=20, labels=None):
  '''
  Given a batch of inputs and labels, and a model,
  symbolically computes expected gradients with k references.
  Args:
      inputs: A [batch_size, ...]-shaped tensor. The input to a model.
      references: A numpy array representing background training data to sample from.
      model:  A tf.keras.Model object, or a subclass object thereof.
      k: The number of samples to use when computing expected gradients.
      index_true_class: Whether or not to take the gradients of the output with respect to the true
          class. True by default. This should be set to True in the multi-class setting, and False
          in the regression setting.
      labels: A [batch_size, num_classes]-shaped tensor.
              The true class labels in one-hot encoding,
              assuming a multi-class problem.
  Returns:
      A tensor the same shape as the input representing the expected gradients
      feature attributions with respect to the output predictions.
  '''
  final_shape = tf.constant([1], dtype=tf.int32)
  final_shape = tf.concat([final_shape, tf.shape(inputs)[1:]], axis=0)
  eg_array = tf.zeros(final_shape, dtype=inputs.dtype)
  #invariant_shape = []
  #for i in range(tf.rank(inputs)):
  #  invariant_shape.append(None)
  for i in tf.range(tf.shape(inputs)[0]):
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(eg_array, tf.TensorShape([None, None, None, None]))]
    )
    sample_references = references

    alpha_shape = tf.reshape(tf.shape(sample_references)[0], [-1])
    alpha_shape = tf.concat([alpha_shape, tf.ones([tf.rank(sample_references)-1], dtype=tf.int32)], axis=0)
    alphas = tf.random.uniform(shape=alpha_shape, minval=0.0, maxval=1.0, dtype=inputs.dtype)
    current_input  = tf.expand_dims(inputs[i], axis=0)

    interpolated_inputs = alphas * current_input + (1.0 - alphas) * sample_references
    with tf.GradientTape() as tape:
      tape.watch(interpolated_inputs)
      _, out, _ = model(interpolated_inputs, training=False)
      if FLAGS.binary_classification:
        out_diff = out
      else:
        out_diff = out - tf.reshape(out[:, labels[i]], (-1,1))
      # out_diff = -1.0*out_diff
      # out_diff = tf.math.log(tf.reduce_sum(tf.exp(out_diff), axis=1) - 1.0)
      out_diff = tf.reduce_sum(out_diff)

    input_gradients = tape.gradient(out_diff, interpolated_inputs)
    difference_from_reference = current_input - sample_references

    expected_gradients_samples = input_gradients * difference_from_reference
    expected_gradients = tf.reduce_mean(tf.abs(expected_gradients_samples), axis=0, keepdims=True)
    if i==0:
      eg_array = expected_gradients
    else:
      eg_array = tf.concat([eg_array, expected_gradients], axis=0)

  return eg_array

def expected_gradients_inter_intra_class(inputs, references, model, k=200, intra_class=False, inter_class=False, labels_inputs=None, labels_references=None):
    '''
    Given a batch of inputs and labels, and a model,
    symbolically computes expected gradients with k references.
    Args:
        inputs: A [batch_size, ...]-shaped tensor. The input to a model.
        references: A numpy array representing background training data to sample from.
        model:  A tf.keras.Model object, or a subclass object thereof.
        k: The number of samples to use when computing expected gradients.
        index_true_class: Whether or not to take the gradients of the output with respect to the true
            class. True by default. This should be set to True in the multi-class setting, and False
            in the regression setting.
        labels: A [batch_size, num_classes]-shaped tensor.
                The true class labels in one-hot encoding,
                assuming a multi-class problem.
    Returns:
        A tensor the same shape as the input representing the expected gradients
        feature attributions with respect to the output predictions.
    '''
    final_shape = tf.constant([1], dtype=tf.int32)
    final_shape = tf.concat([final_shape, tf.shape(inputs)[1:]], axis=0)
    eg_array = tf.zeros(final_shape, dtype=inputs.dtype)
    #invariant_shape = []
    #for i in range(tf.rank(inputs)):
    #  invariant_shape.append(None)
    for i in tf.range(tf.shape(inputs)[0]):
      tf.autograph.experimental.set_loop_options(
        shape_invariants=[(eg_array, tf.TensorShape([None, None, None, None]))]
      )
      if intra_class:
        sample_references = tf.boolean_mask(references, labels_references == labels_inputs[i], axis=0)
      elif inter_class:
        sample_references = tf.boolean_mask(references, labels_references != labels_inputs[i], axis=0)
      else:
        sample_references = references
      if tf.shape(sample_references)[0] > k:
        idxs = tf.range(tf.shape(sample_references)[0])
        ridxs = tf.random.shuffle(idxs)[:k]
        sample_references = tf.gather(sample_references, ridxs)

      final_shape = tf.reshape(tf.shape(sample_references)[0], [-1])
      final_shape = tf.concat([final_shape, tf.ones([tf.rank(sample_references)-1], dtype=tf.int32)], axis=0)
      alphas = tf.random.uniform(shape=final_shape, minval=0.0, maxval=1.0, dtype=inputs.dtype)
      current_input  = tf.expand_dims(inputs[i], axis=0)

      interpolated_inputs = alphas * current_input + (1.0 - alphas) * sample_references
      with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        _, out, _ = model(interpolated_inputs, training=False)
        if FLAGS.binary_classification:
          out_diff = out
        else:
          out_diff = tf.reshape(out[:, labels_inputs[i]], (-1,1)) - out
        out_diff = tf.reduce_sum(out_diff)

      input_gradients = tape.gradient(out_diff, interpolated_inputs)
      difference_from_reference = current_input - sample_references

      expected_gradients_samples = input_gradients * difference_from_reference
      expected_gradients = tf.reduce_mean(tf.abs(expected_gradients_samples), axis=0, keepdims=True)
      if i==0:
        eg_array = expected_gradients
      else:
        eg_array = tf.concat([eg_array, expected_gradients], axis=0)

    return eg_array

#Expected gradients
def expected_gradients_full_2d(inputs, references, model, k=20, labels=None):
  '''
  Given a batch of inputs and labels, and a model,
  symbolically computes expected gradients with k references.
  Args:
      inputs: A [batch_size, ...]-shaped tensor. The input to a model.
      references: A numpy array representing background training data to sample from.
      model:  A tf.keras.Model object, or a subclass object thereof.
      k: The number of samples to use when computing expected gradients.
      index_true_class: Whether or not to take the gradients of the output with respect to the true
          class. True by default. This should be set to True in the multi-class setting, and False
          in the regression setting.
      labels: A [batch_size, num_classes]-shaped tensor.
              The true class labels in one-hot encoding,
              assuming a multi-class problem.
  Returns:
      A tensor the same shape as the input representing the expected gradients
      feature attributions with respect to the output predictions.
  '''
  final_shape = tf.constant([1], dtype=tf.int32)
  final_shape = tf.concat([final_shape, tf.shape(inputs)[1:]], axis=0)
  eg_array = tf.zeros(final_shape, dtype=inputs.dtype)
  #invariant_shape = []
  #for i in range(tf.rank(inputs)):
  #  invariant_shape.append(None)
  for i in range(tf.shape(inputs)[0]):
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(eg_array, tf.TensorShape([None, None]))]
    )
    sample_references = references

    alpha_shape = tf.reshape(tf.shape(sample_references)[0], [-1])
    alpha_shape = tf.concat([alpha_shape, tf.ones([tf.rank(sample_references)-1], dtype=tf.int32)], axis=0)
    alphas = tf.random.uniform(shape=alpha_shape, minval=0.0, maxval=1.0, dtype=inputs.dtype)
    current_input  = tf.expand_dims(inputs[i], axis=0)

    interpolated_inputs = alphas * current_input + (1.0 - alphas) * sample_references
    with tf.GradientTape() as tape:
      tape.watch(interpolated_inputs)
      _, out, _ = model(interpolated_inputs, training=False)
      if FLAGS.binary_classification:
        out_diff = out
      else:
        out_diff = out - tf.reshape(out[:, labels[i]], (-1,1))
      # out_diff = -1.0*out_diff
      # out_diff = tf.math.log(tf.reduce_sum(tf.exp(out_diff), axis=1) - 1.0)
      out_diff = tf.reduce_sum(out_diff)

    input_gradients = tape.gradient(out_diff, interpolated_inputs)
    difference_from_reference = current_input - sample_references

    expected_gradients_samples = input_gradients * difference_from_reference
    expected_gradients = tf.reduce_mean(tf.abs(expected_gradients_samples), axis=0, keepdims=True)
    if i==0:
      eg_array = expected_gradients
    else:
      eg_array = tf.concat([eg_array, expected_gradients], axis=0)

  return eg_array

def expected_gradients_inter_intra_class_2d(inputs, references, model, k=200, intra_class=False, inter_class=False, labels_inputs=None, labels_references=None):
    '''
    Given a batch of inputs and labels, and a model,
    symbolically computes expected gradients with k references.
    Args:
        inputs: A [batch_size, ...]-shaped tensor. The input to a model.
        references: A numpy array representing background training data to sample from.
        model:  A tf.keras.Model object, or a subclass object thereof.
        k: The number of samples to use when computing expected gradients.
        index_true_class: Whether or not to take the gradients of the output with respect to the true
            class. True by default. This should be set to True in the multi-class setting, and False
            in the regression setting.
        labels: A [batch_size, num_classes]-shaped tensor.
                The true class labels in one-hot encoding,
                assuming a multi-class problem.
    Returns:
        A tensor the same shape as the input representing the expected gradients
        feature attributions with respect to the output predictions.
    '''
    final_shape = tf.constant([1], dtype=tf.int32)
    final_shape = tf.concat([final_shape, tf.shape(inputs)[1:]], axis=0)
    eg_array = tf.zeros(final_shape, dtype=inputs.dtype)
    #invariant_shape = []
    #for i in range(tf.rank(inputs)):
    #  invariant_shape.append(None)
    for i in range(tf.shape(inputs)[0]):
      tf.autograph.experimental.set_loop_options(
        shape_invariants=[(eg_array, tf.TensorShape([None, None]))]
      )
      if intra_class:
        sample_references = tf.boolean_mask(references, labels_references == labels_inputs[i], axis=0)
      elif inter_class:
        sample_references = tf.boolean_mask(references, labels_references != labels_inputs[i], axis=0)
      else:
        sample_references = references
      if tf.shape(sample_references)[0] > k:
        idxs = tf.range(tf.shape(sample_references)[0])
        ridxs = tf.random.shuffle(idxs)[:k]
        sample_references = tf.gather(sample_references, ridxs)

      final_shape = tf.reshape(tf.shape(sample_references)[0], [-1])
      final_shape = tf.concat([final_shape, tf.ones([tf.rank(sample_references)-1], dtype=tf.int32)], axis=0)
      alphas = tf.random.uniform(shape=final_shape, minval=0.0, maxval=1.0, dtype=inputs.dtype)
      current_input  = tf.expand_dims(inputs[i], axis=0)

      interpolated_inputs = alphas * current_input + (1.0 - alphas) * sample_references
      with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        _, out, _ = model(interpolated_inputs, training=False)
        if FLAGS.binary_classification:
          out_diff = out
        else:
          out_diff = tf.reshape(out[:, labels_inputs[i]], (-1,1)) - out
        out_diff = tf.reduce_sum(out_diff)

      input_gradients = tape.gradient(out_diff, interpolated_inputs)
      difference_from_reference = current_input - sample_references

      expected_gradients_samples = input_gradients * difference_from_reference
      expected_gradients = tf.reduce_mean(tf.abs(expected_gradients_samples), axis=0, keepdims=True)
      if i==0:
        eg_array = expected_gradients
      else:
        eg_array = tf.concat([eg_array, expected_gradients], axis=0)

    return eg_array

# Correlation between logits conditioned on class
def logit_correlation(model1, model2, data):
  _, logits1, _ = model1(data['image'], training=False)
  _, logits2, _ = model2(data['image'], training=False)
  ctx = tf.distribute.get_replica_context()
  labels_comb = ctx.all_gather(data['label'], axis=0)
  logits1_comb = ctx.all_gather(logits1, axis=0)
  logits2_comb = ctx.all_gather(logits2, axis=0)
  if FLAGS.binary_classification:
    num_labels = 2
  else:
    num_labels = tf.shape(logits1)[1]
  total_corr = 0.0
  total_ind = 0.0
  for label in range(num_labels):
    logits1_curr_label_comb = tf.gather(
            logits1_comb * 1,
            indices=tf.squeeze(
                tf.where(
                    tf.equal(
                        tf.cast(label, dtype='int64'),
                        labels_comb)),
                axis=1))
    logits2_curr_label_comb = tf.gather(
            logits2_comb * 1,
            indices=tf.squeeze(
                tf.where(
                    tf.equal(
                        tf.cast(label, dtype='int64'),
                        labels_comb)),
                axis=1))
    if FLAGS.binary_classification:
      temp_logits_1 = logits1_curr_label_comb
      temp_logits_2 = logits2_curr_label_comb
    else:
      temp_logits_1 = logits1_curr_label_comb - tf.reshape(logits1_curr_label_comb[:, label], (-1,1))
      # temp_logits_1 = -1.0*temp_logits_1
      # temp_logits_1 = tf.math.log(tf.reduce_sum(tf.exp(temp_logits_1), axis=1) - 1.0)
      temp_logits_1 = tf.reduce_sum(temp_logits_1, axis=1)

      temp_logits_2 = logits2_curr_label_comb - tf.reshape(logits2_curr_label_comb[:, label], (-1,1))
      # temp_logits_2 = -1.0*temp_logits_2
      # temp_logits_2 = tf.math.log(tf.reduce_sum(tf.exp(temp_logits_2), axis=1) - 1.0)
      temp_logits_2 = tf.reduce_sum(temp_logits_2, axis=1)

    if tf.shape(temp_logits_1)[0] >= 10 and tf.shape(temp_logits_2)[0] >= 10:
      total_corr += tf.abs(tfp.stats.correlation(tf.reshape(temp_logits_1, (-1,1)), tf.reshape(temp_logits_2, (-1,1)), sample_axis=0)[0,0])
      total_ind += 1.0

  return total_corr/total_ind

# Error diversity
def error_diversity(model1, model2, data, subgroup = 0):
  total_err_1 = 0.0
  total_err_2 = 0.0
  common_err = 0.0
  flip_err_div = False
  if subgroup==1:
    metadata = data['metadata']
    I1 = tf.math.logical_and(metadata[:, 0] == 0, metadata[:, 1] == 0)
    temp_image = data['image'][I1]
    temp_label = data['label'][I1]
  elif subgroup==2:
    metadata = data['metadata']
    I1 = tf.math.logical_and(metadata[:, 0] == 0, metadata[:, 1] == 1)
    temp_image = data['image'][I1]
    temp_label = data['label'][I1]
    if FLAGS.flip_err_div_for_minority:
      flip_err_div = True
  elif subgroup==3:
    metadata = data['metadata']
    I1 = tf.math.logical_and(metadata[:, 0] == 1, metadata[:, 1] == 0)
    temp_image = data['image'][I1]
    temp_label = data['label'][I1]
    if FLAGS.flip_err_div_for_minority:
      flip_err_div = True
  elif subgroup==4:
    metadata = data['metadata']
    I1 = tf.math.logical_and(metadata[:, 0] == 1, metadata[:, 1] == 1)
    temp_image = data['image'][I1]
    temp_label = data['label'][I1]
  else:
    temp_image = data['image']
    temp_label = data['label']
  _, logit1, _ = model1(temp_image, training=False)
  _, logit2, _ = model2(temp_image, training=False)
  if FLAGS.binary_classification:
    if flip_err_div:
      err1 = tf.cast(tf.reshape(logit1, [-1]) > 0, tf.int64) == temp_label
      err2 = tf.cast(tf.reshape(logit2, [-1]) > 0, tf.int64) == temp_label
    else:
      err1 = tf.cast(tf.reshape(logit1, [-1]) > 0, tf.int64) != temp_label
      err2 = tf.cast(tf.reshape(logit2, [-1]) > 0, tf.int64) != temp_label
  else:
    if flip_err_div:
      err1 = tf.argmax(logit1, axis=1) == temp_label
      err2 = tf.argmax(logit2, axis=1) == temp_label
    else:
      err1 = tf.argmax(logit1, axis=1) != temp_label
      err2 = tf.argmax(logit2, axis=1) != temp_label
  comm_err = err1 & err2
  total_err_1 += tf.reduce_sum(tf.cast(err1, tf.float32))
  total_err_2 += tf.reduce_sum(tf.cast(err2, tf.float32))
  common_err += tf.reduce_sum(tf.cast(comm_err, tf.float32))

  return tf.stack([total_err_1, total_err_2, common_err], axis=0)

#Gaussian noise robustness ensemble
def gauss_noise_robust(models, data):
  ctx = tf.distribute.get_replica_context()
  #accum_data_image = ctx.all_gather(data['image'], axis=0)
  accum_data_label = ctx.all_gather(data['label'], axis=0)
  std_devs = FLAGS.max_gauss_noise_std * tf.random.uniform([tf.shape(data['image'])[0]])
  if FLAGS.measure_feat_robust:
    if len(models) > 1:
      # assuming each model is a head model (not a multi base model)
      feats = models[0].layers[0](data['image'], training=False)
      final_shape = tf.constant([-1], dtype=tf.int32)
      final_shape = tf.concat([final_shape, tf.ones([tf.rank(feats)-1], dtype=tf.int32)], axis=0)
      z = tf.reshape(std_devs, final_shape) * tf.random.normal(tf.shape(feats))
      noise_feats = feats + tf.cast(z, feats.dtype)
      avg = 0.0
      for i in range(len(models)):
        x = noise_feats
        for ind, layer in enumerate(models[i].layers):
          if ind > 0:
            x = layer(x, training=False)
        avg = (i/(i+1))*avg + (1/(i+1))*x
    else:
      _, avg, _ = models[0](data['image'], training=False, gauss_noise_feats=True, sigma=FLAGS.max_gauss_noise_std, rand_sigma=True)
  else:
    final_shape = tf.constant([-1], dtype=tf.int32)
    final_shape = tf.concat([final_shape, tf.ones([tf.rank(data['image'])-1], dtype=tf.int32)], axis=0)
    z = tf.reshape(std_devs, final_shape) * tf.random.normal(tf.shape(data['image']))
    noise_image = data['image'] + tf.cast(z, data['image'].dtype)
    # accum_data_image = accum_data_image + tf.cast(z, accum_data_image.dtype)
    avg = 0.0
    for i in range(len(models)):
      _,out,_ = models[i](noise_image, training=False)
      avg = (i/(i+1))*avg + (1/(i+1))*out
  if FLAGS.binary_classification:
    corr = tf.cast(tf.reshape(avg, [-1]) > 0, tf.int64) == data['label']
  else:
    corr = tf.argmax(avg, axis=1) == data['label']

  accum_corr = ctx.all_gather(corr, axis=0)
  return tf.reduce_sum(tf.cast(accum_corr, tf.float32))/tf.cast(tf.shape(accum_data_label)[0], tf.float32)

#Gaussian noise robustness averaged over multiple std-dev
def gauss_noise_robust_2(models, data):
  ctx = tf.distribute.get_replica_context()
  accum_data_label = ctx.all_gather(data['label'], axis=0)
  std_devs = tf.range(0.0, FLAGS.max_gauss_noise_std, delta=0.1)
  total_corr = 0.0
  for std_dev in std_devs:
    # tf.autograph.experimental.set_loop_options(
    #     shape_invariants=[(total_corr, tf.TensorShape([None]))]
    # )
    if FLAGS.measure_feat_robust:
      if len(models) > 1:
        # assuming each model is a head model (not a multi base model)
        feats = models[0].layers[0](data['image'], training=False)
        z = std_dev * tf.random.normal(tf.shape(feats))
        noise_feats = feats + tf.cast(z, feats.dtype)
        avg = 0.0
        for i in range(len(models)):
          x = noise_feats
          for ind, layer in enumerate(models[i].layers):
            if ind > 0:
              x = layer(x, training=False)
          avg = (i/(i+1.0))*avg + (1.0/(i+1.0))*x
      else:
        _, avg, _ = models[0](data['image'], training=False, gauss_noise_feats=True, sigma=std_dev)
    else:
      z = std_dev * tf.random.normal(tf.shape(data['image']))
      noise_image = data['image'] + tf.cast(z, data['image'].dtype)
      avg = 0.0
      for i in range(len(models)):
        _,out,_ = models[i](noise_image, training=False)
        avg = (i/(i+1))*avg + (1/(i+1))*out
    if FLAGS.binary_classification:
      corr = tf.cast(tf.reshape(avg, [-1]) > 0, tf.int64) == data['label']
    else:
      corr = tf.argmax(avg, axis=1) == data['label']

    total_corr += tf.reduce_sum(tf.cast(corr, tf.float32))

  total_corr = ctx.all_reduce(tf.distribute.ReduceOp.SUM, total_corr)
  final_score = tf.reduce_sum(total_corr)/tf.cast(tf.shape(accum_data_label)[0], tf.float32)
  return final_score

#Masking noise robustness ensemble
def mask_noise_robust(models, data):
  ctx = tf.distribute.get_replica_context()
  accum_data_image = ctx.all_gather(data['image'], axis=0)
  accum_data_label = ctx.all_gather(data['label'], axis=0)
  temp = tf.identity(accum_data_image)
  probs = tf.range(0.0, 1.0, delta=0.01)
  baseline = 0.0
  final = 0.0
  for prob in probs:
    bern = tfp.distributions.Bernoulli(probs=1-prob)
    z = bern.sample(sample_shape=tf.shape(temp))
    temp2 = tf.cast(z, temp.dtype)*temp
    avg = 0.0
    for i in range(len(models)):
      _,out,_ = models[i](temp2, training=False)
      avg = (i/(i+1))*avg + (1/(i+1))*out
    if FLAGS.binary_classification:
      corr = tf.cast(tf.reshape(avg, [-1]) > 0, tf.int64) == accum_data_label
    else:
      corr = tf.argmax(avg, axis=1) == accum_data_label
    final += tf.reduce_sum(tf.cast(corr, tf.float32))
    if prob == 0:
      baseline = tf.reduce_sum(tf.cast(corr, tf.float32))*tf.cast(tf.shape(probs)[0], tf.float32)

  return final/baseline

# RDE based sparsest mask
def RDE_sparse_mask(models, data, thres=0.1):
  temp = tf.identity(data['image'])
  opt = tf.keras.optimizers.SGD(lr = 0.01)
  init_avg = 0.0
  for i in range(len(models)):
    _,out,_ = models[i](temp, training=False)
    init_avg = (i/(i+1))*init_avg + (1/(i+1))*out
  lambda_vals = np.arange(1.0, 0.0, -0.05)

  for curr_lambda in lambda_vals:
    v = tf.Variable(tf.ones(tf.shape(temp), dtype=tf.float32))

    @tf.function
    def train_step(input):
      with tf.GradientTape() as tape:
        avg = 0.0
        for i in range(len(models)):
          _, out, _ = models[i](input*v, training=False)
          avg = (i/(i+1))*avg + (1/(i+1))*out
        loss = (tf.norm(avg-init_avg))**2 + curr_lambda*tf.reduce_sum(tf.abs(v))
      grads = tape.gradient(loss, v)
      opt.apply_gradients([(grads,v)])
      return loss

    for step in range(100):
      l = train_step(temp)
    avg = 0.0
    for i in range(len(models)):
      _, out, _ = models[i](temp*v, training=False)
      avg = (i/(i+1))*avg + (1/(i+1))*out
    if tf.norm(init_avg - avg)/tf.norm(init_avg) < thres:
      return tf.reduce_sum(tf.abs(v))/tf.reduce_sum(tf.ones(tf.shape(temp)))
