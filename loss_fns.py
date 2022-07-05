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

"""Defines different loss functions."""
import typing
import tensorflow as tf
import tensorflow_probability as tfp
from absl import flags

FLAGS = flags.FLAGS


def grad_loss(logits, labels, features, tape):
  logits = logits - tf.gather_nd(logits, labels.reshape(-1,1), batch_dims=1)
  loss = tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.exp(logits), axis=1) - 1.0))
  grads = tape.gradient(loss, features)
  ctx = tf.distribute.get_replica_context()
  grad_comb = ctx.all_gather(grads, axis=0)
  final_loss = tf.var(tf.mean(tf.abs(grad_comb), axis=0))
  return tf.nn.scale_regularization_loss(final_loss)


def cross_correlation_matrix(reps1, reps2):
  return tfp.stats.correlation(reps1, y=reps2, sample_axis=0, event_axis=-1)


def compute_cross_redundancy_loss_GPU(reps1: tf.Tensor, reps2:tf.Tensor, rr_weight: float) -> float:
  """Function to compute redundancy loss.

  Args:
    reps: representations
    rr_weight: redundancy loss weight
  Returns:
    Loss value.
  """
  if tf.shape(reps1)[0] >= 10 and tf.shape(reps2)[0] >= 10:
    cc_matrix = tf.squeeze(cross_correlation_matrix(reps1, reps2))
    pow_mat = tf.ones(cc_matrix.shape) * 2
    cc_matrix = tf.math.pow(cc_matrix, pow_mat)  # dim X dim matrix
    # adding all the elements to get the redundancy loss and scaling it
    redundancy_loss = tf.nn.scale_regularization_loss(
        rr_weight*tf.reduce_sum(cc_matrix))
    return redundancy_loss
  else:
    return 0.0

def compute_cross_redundancy_loss_TPU(reps1: tf.Tensor, reps2:tf.Tensor, rr_weight: float) -> float:
  """Function to compute redundancy loss.

  Args:
    reps: representations
    rr_weight: redundancy loss weight
  Returns:
    Loss value.
  """
  cc_matrix = tf.squeeze(cross_correlation_matrix(reps1, reps2))
  pow_mat = tf.ones(cc_matrix.shape) * 2
  cc_matrix = tf.math.pow(cc_matrix, pow_mat)  # dim X dim matrix
  # adding all the elements to get the redundancy loss and scaling it
  redundancy_loss = tf.nn.scale_regularization_loss(
      rr_weight*tf.reduce_sum(cc_matrix))
  return redundancy_loss

def correlation_matrix(reps):
  return tfp.stats.correlation(reps, y=None, sample_axis=0, event_axis=-1)


def compute_redundancy_loss_GPU(reps: tf.Tensor, rr_weight: float) -> float:
  """Function to compute redundancy loss.

  Args:
    reps: representations
    rr_weight: redundancy loss weight
  Returns:
    Loss value.
  """
  if tf.shape(reps)[0] >= 10:
    cc_matrix = tf.squeeze(correlation_matrix(reps))
    dim = tf.shape(cc_matrix)[0]
    cc_matrix_diff = (cc_matrix - tf.eye(dim))
    pow_mat = tf.ones(cc_matrix_diff.shape) * 2
    cc_matrix_diff = tf.math.pow(cc_matrix_diff, pow_mat)  # dim X dim matrix
    weighted_mat = (tf.ones(cc_matrix.shape) - tf.eye(dim)) * rr_weight
    # multiplying the off-diagnol elements with effective rr weight
    cc_matrix_diff = tf.math.multiply(cc_matrix_diff, weighted_mat)
    # adding all the elements to get the redundancy loss and scaling it
    redundancy_loss = tf.nn.scale_regularization_loss(
        tf.reduce_sum(cc_matrix_diff))
    if redundancy_loss != redundancy_loss:
      return 0.0
    else:
      return redundancy_loss
  else:
    return 0.0

def compute_redundancy_loss_TPU(reps: tf.Tensor, rr_weight: float) -> float:
  """Function to compute redundancy loss.

  Args:
    reps: representations
    rr_weight: redundancy loss weight
  Returns:
    Loss value.
  """
  cc_matrix = tf.squeeze(correlation_matrix(reps))
  dim = tf.shape(cc_matrix)[0]
  cc_matrix_diff = (cc_matrix - tf.eye(dim))
  pow_mat = tf.ones(cc_matrix_diff.shape) * 2
  cc_matrix_diff = tf.math.pow(cc_matrix_diff, pow_mat)  # dim X dim matrix
  weighted_mat = (tf.ones(cc_matrix.shape) - tf.eye(dim)) * rr_weight
  # multiplying the off-diagnol elements with effective rr weight
  cc_matrix_diff = tf.math.multiply(cc_matrix_diff, weighted_mat)
  # adding all the elements to get the redundancy loss and scaling it
  redundancy_loss = tf.nn.scale_regularization_loss(
      tf.reduce_sum(cc_matrix_diff))
  return redundancy_loss

def compute_explained_variance(reps: tf.Tensor, exp_reps_mat: tf.Tensor, rr_weight: float):
  temp_mat = tf.linalg.matmul(exp_reps_mat, tf.linalg.pinv(exp_reps_mat))
  overall_loss = 0.0
  for i in range(reps.shape[1]):
    exp_var = tf.norm(tf.linalg.matmul(temp_mat, tf.reshape(reps[:,i], [-1,1])))**2
    overall_loss += exp_var/(tf.norm(reps[:,i])**2)
  return tf.nn.scale_regularization_loss(rr_weight * overall_loss)

def compute_MI(prob1, prob2, rr_weight, num_sq=True):
  marginal1 = tf.reduce_mean(prob1, axis=0, keepdims=True)
  marginal2 = tf.reduce_mean(prob2, axis=0, keepdims=True)
  marginal_prob = tf.linalg.matmul(tf.transpose(marginal1), marginal2)
  joint_prob = (tf.linalg.matmul(tf.transpose(prob1), prob2))/(tf.cast(tf.shape(prob1)[0], tf.float32))
  MI = tf.reduce_sum(joint_prob * (tf.math.log(joint_prob) - tf.math.log(marginal_prob)))
  if FLAGS.normalize_MI:
    if FLAGS.normalize_MI_random:
      prob1_shuff = tf.gather(prob1, tf.random.shuffle(tf.range(tf.shape(prob1)[0])))
      prob2_shuff = tf.gather(prob2, tf.random.shuffle(tf.range(tf.shape(prob2)[0])))
      joint_prob_shuff = (tf.linalg.matmul(tf.transpose(prob1_shuff), prob2_shuff))/(tf.cast(tf.shape(prob1)[0], tf.float32))
      MI_shuff = tf.reduce_sum(joint_prob_shuff * (tf.math.log(joint_prob_shuff) - tf.math.log(marginal_prob)))
      if FLAGS.use_num_sq_MI and num_sq:
        if FLAGS.use_stop_grad:
          MI = (MI**2)/tf.stop_gradient(MI_shuff)
        else:
          MI = (MI**2)/MI_shuff
      else:
        if FLAGS.use_stop_grad:
          MI = MI/tf.stop_gradient(MI_shuff)
        else:
          MI = MI/MI_shuff
    else:
      ent1 = -tf.reduce_sum(marginal1*tf.math.log(marginal1))
      MI = MI/ent1
  if FLAGS.use_sq_MI:
    MI = MI**2
  return tf.nn.scale_regularization_loss(rr_weight * MI)

def get_diff_sq_mat(X):
  X_repeat_col = tf.repeat(X, repeats=[tf.shape(X)[0]], axis=1)
  X_repeat_row = tf.repeat(tf.reshape(X, (1, tf.shape(X)[0])), repeats=[tf.shape(X)[0]], axis=0)
  return (X_repeat_col - X_repeat_row)**2

def get_diff_sq_mat_multivariate(X):
  X_repeat = tf.repeat(X, repeats=[tf.shape(X)[1]], axis=2)
  X_repeat_T = tf.transpose(X_repeat, perm=[0,2,1])
  return (X_repeat - X_repeat_T)**2

def est_HSIC(X, Y):
  X = tf.expand_dims(X, axis=-1)
  Y = tf.expand_dims(Y, axis=-1)
  var1 = tfp.stats.variance(X)
  var2 = tfp.stats.variance(Y)
  mat1 = (-1.0/var1)*get_diff_sq_mat(X)
  mat2 = (-1.0/var2)*get_diff_sq_mat(Y)
  mat1 = tf.math.exp(mat1)
  mat2 = tf.math.exp(mat2)
  H = tf.eye(tf.shape(X)[0]) - (1.0/tf.cast(tf.shape(X)[0], tf.float32))*tf.ones((tf.shape(X)[0], tf.shape(X)[0]))

  fin_mat = tf.linalg.matmul(mat1, H)
  fin_mat = tf.linalg.matmul(fin_mat, mat2)
  fin_mat = tf.linalg.matmul(fin_mat, H)
  return tf.linalg.trace(fin_mat)/(tf.cast(tf.shape(X)[0], tf.float32)**2)

def est_HSIC_multivariate(X, Y):
  var1 = tfp.stats.variance(X, sample_axis=0)
  var2 = tfp.stats.variance(Y, sample_axis=0)
  # transpose BxF matrix to FxB matrix
  X = tf.transpose(X)
  Y = tf.transpose(Y)
  X = tf.expand_dims(X, axis=-1)
  Y = tf.expand_dims(Y, axis=-1)
  var1 = tf.reshape(var1, (-1,1,1))
  var2 = tf.reshape(var2, (-1,1,1))
  mat1 = (-1.0/var1)*get_diff_sq_mat_multivariate(X)
  mat2 = (-1.0/var2)*get_diff_sq_mat_multivariate(Y)
  mat1 = tf.reduce_mean(mat1, axis=0)
  mat2 = tf.reduce_mean(mat2, axis=0)
  mat1 = tf.math.exp(mat1)
  mat2 = tf.math.exp(mat2)
  H = tf.eye(tf.shape(mat1)[0]) - (1.0/tf.cast(tf.shape(mat1)[0], tf.float32))*tf.ones((tf.shape(mat1)[0], tf.shape(mat1)[0]))

  fin_mat = tf.linalg.matmul(mat1, H)
  fin_mat = tf.linalg.matmul(fin_mat, mat2)
  fin_mat = tf.linalg.matmul(fin_mat, H)
  return tf.linalg.trace(fin_mat)/(tf.cast(tf.shape(mat1)[0], tf.float32)**2)

# estimate HSIC in case X and Y were independent
def est_HSIC_ind(X, Y):
  var1 = tfp.stats.variance(X, sample_axis=0)
  var2 = tfp.stats.variance(Y, sample_axis=0)
  # transpose BxF matrix to FxB matrix
  X = tf.transpose(X)
  Y = tf.transpose(Y)
  X = tf.expand_dims(X, axis=-1)
  Y = tf.expand_dims(Y, axis=-1)
  var1 = tf.reshape(var1, (-1,1,1))
  var2 = tf.reshape(var2, (-1,1,1))
  mat1 = (-1.0/var1)*get_diff_sq_mat_multivariate(X)
  mat2 = (-1.0/var2)*get_diff_sq_mat_multivariate(Y)
  mat1 = tf.reduce_mean(mat1, axis=0)
  mat2 = tf.reduce_mean(mat2, axis=0)
  mat1 = tf.math.exp(mat1)
  mat2 = tf.math.exp(mat2)

  mu_x_2 = tf.reduce_mean(mat1)
  mu_y_2 = tf.reduce_mean(mat2)

  return (1.0 + mu_x_2*mu_y_2 - mu_x_2 - mu_y_2)/tf.cast(tf.shape(mat1)[0], tf.float32)

def compute_HSIC_loss_ind(X, Y, rr_weight):
  l1 = est_HSIC_ind(X, Y)
  l2 = est_HSIC_multivariate(X, X)
  l3 = est_HSIC_multivariate(Y, Y)
  loss = l1/(tf.sqrt(l2*l3))
  if FLAGS.use_sq_HSIC:
    loss = loss**2
  return tf.nn.scale_regularization_loss(rr_weight * loss), tf.nn.scale_regularization_loss(l1), tf.nn.scale_regularization_loss(l2), tf.nn.scale_regularization_loss(l3)

def compute_HSIC_loss(X, Y, rr_weight):
  l1 = est_HSIC(X, Y)
  l2 = est_HSIC(X, X)
  l3 = est_HSIC(Y, Y)
  loss = l1/(tf.sqrt(l2*l3))
  if FLAGS.use_sq_HSIC:
    loss = loss**2
  return tf.nn.scale_regularization_loss(rr_weight * loss), tf.nn.scale_regularization_loss(l1), tf.nn.scale_regularization_loss(l2), tf.nn.scale_regularization_loss(l3)

def compute_HSIC_loss_multivariate(X, Y, rr_weight):
  l1 = est_HSIC_multivariate(X, Y)
  l2 = est_HSIC_multivariate(X, X)
  l3 = est_HSIC_multivariate(Y, Y)
  loss = l1/(tf.sqrt(l2*l3))
  if FLAGS.use_sq_HSIC:
    loss = loss**2
  return tf.nn.scale_regularization_loss(rr_weight * loss), tf.nn.scale_regularization_loss(l1), tf.nn.scale_regularization_loss(l2), tf.nn.scale_regularization_loss(l3)

def compute_HSIC_loss_ratio_multivariate(X, Y, rr_weight):
  l1 = est_HSIC_multivariate(X, Y)
  l2 = est_HSIC_ind(X, Y)
  return tf.nn.scale_regularization_loss(rr_weight * l1/l2), tf.nn.scale_regularization_loss(l1)

def compute_disagreement(prob1, prob2, rr_weight):
  # prod_prob = tf.linalg.matmul(prob1, tf.transpose(prob2))
  # prod_prob = tf.linalg.diag_part(prod_prob)
  # disagreement_loss = -tf.math.log(1 - prod_prob)
  prob_diff = tf.math.abs(prob1 - prob2)
  disagreement_loss = -tf.reduce_mean(prob_diff)
  return tf.nn.scale_regularization_loss(rr_weight * disagreement_loss)

def compute_seq_multihead_exp_variance_loss(labels: tf.Tensor,
                                            logits: tf.Tensor,
                                  class_specific_rr: bool = True,
                                  reps: typing.Optional[tf.Tensor] = None,
                                  reps_arr = None,
                                  rr_weight: typing.Optional[float] = None) -> float:
  # Redundancy loss
  overall_redundancy_loss = tf.constant(0.0)
  if rr_weight is None:
    raise ValueError('RR weight not specified.')

  if class_specific_rr:
    raise ValueError('Dont use explaining away variance loss when class conditioned. Representation size becomes bigger than data points')

  ctx = tf.distribute.get_replica_context()
  labels_comb = ctx.all_gather(labels, axis=0)
  reps_comb = ctx.all_gather(reps, axis=0)
  reps_comb = reps_comb - tf.reduce_mean(reps_comb, axis=0, keepdims=1)

  for i, reps1 in enumerate(reps_arr):
    reps1_comb = ctx.all_gather(reps1, axis=0)
    reps1_comb = reps1_comb - tf.reduce_mean(reps1_comb, axis=0, keepdims=1)
    redundancy_loss = compute_explained_variance(reps1_comb, reps_comb, rr_weight)
    overall_redundancy_loss += redundancy_loss

  return overall_redundancy_loss


def compute_batch_redundancy_loss(labels: tf.Tensor,
                                  logits: tf.Tensor,
                                  class_specific_rr: bool = True,
                                  reps: typing.Optional[tf.Tensor] = None,
                                  rr_weight: typing.Optional[float] = None) -> float:
  if rr_weight is None:
    raise ValueError('RR weight not specified.')

  ctx = tf.distribute.get_replica_context()
  # Check that the gather operation is being done along the correct axis
  reps_comb = ctx.all_gather(reps, axis=0)
  labels_comb = ctx.all_gather(labels, axis=0)
  if class_specific_rr:
    num_labels = logits.shape[1]
    redundancy_loss = tf.constant(0.0)
    avg_over = tf.constant(0.0)
    for label in range(num_labels):
      reps_curr_label_comb = tf.gather(
          reps_comb * 1,
          indices=tf.squeeze(
              tf.where(
                  tf.equal(
                      tf.cast(tf.constant(label), dtype='int64'),
                      labels_comb)),
              axis=1))

      if FLAGS.platform == 'GPU':
        redundancy_loss += compute_redundancy_loss_GPU(reps=reps_curr_label_comb, rr_weight=rr_weight)
      else:
        redundancy_loss += compute_redundancy_loss_TPU(reps=reps_curr_label_comb, rr_weight=rr_weight)

    redundancy_loss = redundancy_loss / num_labels
  else:
    redundancy_loss = compute_redundancy_loss_TPU(reps_comb, rr_weight)

  return redundancy_loss


def compute_multihead_batch_redundancy_loss(labels: tf.Tensor,
                                            logits_arr,
                                  class_specific_rr: bool = True,
                                  reps_arr = None,
                                  rr_weight: typing.Optional[float] = None) -> float:
  # Redundancy loss
  overall_redundancy_loss = tf.constant(0.0)
  if rr_weight is None:
    raise ValueError('RR weight not specified.')

  ctx = tf.distribute.get_replica_context()
  labels_comb = ctx.all_gather(labels, axis=0)

  for i, reps1 in enumerate(reps_arr):
    reps1_comb = ctx.all_gather(reps1, axis=0)
    for j in range(i+1, len(reps_arr)):
      reps2_comb = ctx.all_gather(reps_arr[j], axis=0)
      if class_specific_rr:
        num_labels = logits_arr[0].shape[1]
        redundancy_loss = tf.constant(0.0)
        for label in range(num_labels):
          reps1_curr_label_comb = tf.gather(
              reps1_comb * 1,
              indices=tf.squeeze(
                  tf.where(
                      tf.equal(
                          tf.cast(tf.constant(label), dtype='int64'),
                          labels_comb)),
                  axis=1))
          reps2_curr_label_comb = tf.gather(
              reps2_comb * 1,
              indices=tf.squeeze(
                  tf.where(
                      tf.equal(
                          tf.cast(tf.constant(label), dtype='int64'),
                          labels_comb)),
                  axis=1))

          if FLAGS.platform == 'GPU':
            redundancy_loss += compute_cross_redundancy_loss_GPU(reps1=reps1_curr_label_comb, reps2=reps2_curr_label_comb, rr_weight=rr_weight)
          else:
            redundancy_loss += compute_cross_redundancy_loss_TPU(reps1=reps1_curr_label_comb, reps2=reps2_curr_label_comb, rr_weight=rr_weight)

        redundancy_loss = redundancy_loss / num_labels
      else:
        redundancy_loss = compute_cross_redundancy_loss_TPU(reps1=reps1_comb, reps2=reps2_comb, rr_weight=rr_weight)

      overall_redundancy_loss += redundancy_loss

  return overall_redundancy_loss

def compute_seq_multihead_batch_redundancy_loss(labels: tf.Tensor,
                                            logits: tf.Tensor,
                                  class_specific_rr: bool = True,
                                  reps: typing.Optional[tf.Tensor] = None,
                                  reps_arr = None,
                                  rr_weight: typing.Optional[float] = None) -> float:
  # Redundancy loss
  overall_redundancy_loss = tf.constant(0.0)
  if rr_weight is None:
    raise ValueError('RR weight not specified.')

  if FLAGS.lin_scale_rr_weight and len(reps_arr) > 0:
    rr_weight = rr_weight/len(reps_arr)

  ctx = tf.distribute.get_replica_context()
  labels_comb = ctx.all_gather(labels, axis=0)
  reps_comb = ctx.all_gather(reps, axis=0)

  for i, reps1 in enumerate(reps_arr):
    reps1_comb = ctx.all_gather(reps1, axis=0)
    if class_specific_rr:
      num_labels = logits.shape[1]
      redundancy_loss = tf.constant(0.0)
      for label in range(num_labels):
        reps1_curr_label_comb = tf.gather(
            reps1_comb * 1,
            indices=tf.squeeze(
                tf.where(
                    tf.equal(
                        tf.cast(tf.constant(label), dtype='int64'),
                        labels_comb)),
                axis=1))
        reps_curr_label_comb = tf.gather(
            reps_comb * 1,
            indices=tf.squeeze(
                tf.where(
                    tf.equal(
                        tf.cast(tf.constant(label), dtype='int64'),
                        labels_comb)),
                axis=1))

        if FLAGS.platform == 'GPU':
          redundancy_loss += compute_cross_redundancy_loss_GPU(reps1=reps1_curr_label_comb, reps2=reps_curr_label_comb, rr_weight=rr_weight)
        else:
          redundancy_loss += compute_cross_redundancy_loss_TPU(reps1=reps1_curr_label_comb, reps2=reps_curr_label_comb, rr_weight=rr_weight)

      redundancy_loss = redundancy_loss / num_labels
    else:
      redundancy_loss = compute_cross_redundancy_loss_TPU(reps1=reps1_comb, reps2=reps_comb, rr_weight=rr_weight)

    overall_redundancy_loss += redundancy_loss

  return overall_redundancy_loss

def compute_multihead_MI_loss(labels: tf.Tensor,
                        logits_arr,
                        class_specific_rr: bool = True,
                        rr_weight: typing.Optional[float] = None) -> float:
  # MI loss
  overall_MI_loss = tf.constant(0.0)
  if rr_weight is None:
    raise ValueError('RR weight not specified.')

  ctx = tf.distribute.get_replica_context()
  labels_comb = ctx.all_gather(labels, axis=0)

  for i, logits1 in enumerate(logits_arr):
    logits1_comb = ctx.all_gather(logits1, axis=0)
    probs1_comb = tf.nn.softmax(logits1_comb, axis=1)
    for j in range(i+1, len(logits_arr)):
      logits2_comb = ctx.all_gather(logits_arr[j], axis=0)
      probs2_comb = tf.nn.softmax(logits2_comb, axis=1)
      if class_specific_rr:
        num_labels = logits1.shape[1]
        MI_loss = tf.constant(0.0)
        for label in range(num_labels):
          probs1_curr_label_comb = tf.gather(
              probs1_comb * 1,
              indices=tf.squeeze(
                  tf.where(
                      tf.equal(
                          tf.cast(tf.constant(label), dtype='int64'),
                          labels_comb)),
                  axis=1))
          probs2_curr_label_comb = tf.gather(
              probs2_comb * 1,
              indices=tf.squeeze(
                  tf.where(
                      tf.equal(
                          tf.cast(tf.constant(label), dtype='int64'),
                          labels_comb)),
                  axis=1))
          logits1_curr_label_comb = tf.gather(
              logits1_comb * 1,
              indices=tf.squeeze(
                  tf.where(
                      tf.equal(
                          tf.cast(tf.constant(label), dtype='int64'),
                          labels_comb)),
                  axis=1))
          logits2_curr_label_comb = tf.gather(
              logits2_comb * 1,
              indices=tf.squeeze(
                  tf.where(
                      tf.equal(
                          tf.cast(tf.constant(label), dtype='int64'),
                          labels_comb)),
                  axis=1))

          if FLAGS.use_disagr_loss:
            MI_loss += compute_disagreement(probs1_curr_label_comb, probs2_curr_label_comb, rr_weight)
          elif FLAGS.use_HSIC_loss:
            temp_MI_loss,_,_,_ = compute_HSIC_loss(logits1_curr_label_comb[:, label], logits2_curr_label_comb[:, label], rr_weight)
            MI_loss += temp_MI_loss
          else:
            MI_loss += compute_MI(probs1_curr_label_comb, probs2_curr_label_comb, rr_weight)

        MI_loss = MI_loss / num_labels
      else:
        if FLAGS.use_disagr_loss:
          MI_loss = compute_disagreement(probs1_comb, probs2_comb, rr_weight)
        else:
          MI_loss = compute_MI(probs1_comb, probs2_comb, rr_weight)

      overall_MI_loss += MI_loss

  return overall_MI_loss

def est_seq_HSIC_loss_ind(labels: tf.Tensor,
                          logits: tf.Tensor,
                          reps: tf.Tensor,
                          class_specific_rr: bool = True,
                          reps_arr = None,
                          rr_weight: typing.Optional[float] = None):
  # HSIC loss
  overall_HSIC_loss = tf.constant(0.0)
  overall_HSIC_XY = tf.constant(0.0)
  overall_HSIC_XX = tf.constant(0.0)
  overall_HSIC_YY = tf.constant(0.0)
  overall_th_hsic_loss = tf.constant(0.0)
  overall_th_hsic_xy = tf.constant(0.0)
  if rr_weight is None:
    raise ValueError('RR weight not specified.')

  if FLAGS.lin_scale_rr_weight and len(reps_arr) > 0:
    rr_weight = rr_weight/len(reps_arr)

  ctx = tf.distribute.get_replica_context()
  labels_comb = ctx.all_gather(labels, axis=0)
  # if FLAGS.sep_short_direct_branch:
  reps_comb = []
  for i in range(len(reps)):
    reps_comb.append(ctx.all_gather(reps[i], axis=0))
  # else:
  # reps_comb = ctx.all_gather(reps, axis=0)

  for i, reps1 in enumerate(reps_arr):
    reps1_comb = ctx.all_gather(reps1, axis=0)
    if class_specific_rr:
      if FLAGS.binary_classification:
        num_labels = 2
      else:
        num_labels = logits.shape[1]
      HSIC_loss = tf.constant(0.0)
      HSIC_XY = tf.constant(0.0)
      HSIC_XX = tf.constant(0.0)
      HSIC_YY = tf.constant(0.0)
      th_hsic_loss = tf.constant(0.0)
      th_hsic_xy = tf.constant(0.0)
      for label in range(num_labels):
        reps1_curr_label_comb = tf.gather(
            reps1_comb * 1,
            indices=tf.squeeze(
                tf.where(
                    tf.equal(
                        tf.cast(tf.constant(label), dtype='int64'),
                        labels_comb)),
                axis=1))
        reps1_curr_label_comb = tf.stop_gradient(tf.random.shuffle(reps1_curr_label_comb))
        if FLAGS.sep_short_direct_branch:
          for j in range(len(reps)):
            reps_curr_label_comb = tf.gather(
                reps_comb[j] * 1,
                indices=tf.squeeze(
                    tf.where(
                        tf.equal(
                            tf.cast(tf.constant(label), dtype='int64'),
                            labels_comb)),
                    axis=1))

            if tf.shape(reps1_curr_label_comb)[0] >= 10 and tf.shape(reps_curr_label_comb)[0] >= 10:
              if FLAGS.use_GAP_HSIC_features:
                HSIC_loss += compute_HSIC_loss_multivariate(reps1_curr_label_comb, reps_curr_label_comb, rr_weight)
              else:
                # Assuming using HSIC with respect to previous models logits
                z = tf.random.normal((tf.shape(reps[j])[1], tf.shape(reps[j])[2]))
                z = z/tf.norm(z)
                z = tf.reshape(z, (1,1,-1))
                temp_reps = tf.transpose(reps_curr_label_comb, perm=[0,3,1,2])
                temp_reps = tf.reshape(temp_reps, (tf.shape(temp_reps)[0], tf.shape(temp_reps)[1], -1))
                fin_rep = tf.reduce_sum(z*temp_reps, axis=2)
                HSIC_loss += compute_HSIC_loss_multivariate(reps1_curr_label_comb, fin_rep, rr_weight)
        else:
          for j in range(len(reps)):
            reps_curr_label_comb = tf.gather(
                reps_comb[j] * 1,
                indices=tf.squeeze(
                    tf.where(
                        tf.equal(
                            tf.cast(tf.constant(label), dtype='int64'),
                            labels_comb)),
                    axis=1))

            # Comment below line for working on TPUs (Need to find a fix for this)
            # if tf.shape(reps1_curr_label_comb)[0] >= 10 and tf.shape(reps_curr_label_comb)[0] >= 10:
            if FLAGS.use_GAP_HSIC_features or FLAGS.use_random_projections:
              if FLAGS.eig_cutoff_factor == 0.0:
                temp_HSIC_loss, temp_HSIC_XY, temp_HSIC_XX, temp_HSIC_YY = compute_HSIC_loss_multivariate(reps1_curr_label_comb, reps_curr_label_comb, rr_weight)
                temp_th_hsic_loss, temp_th_hsic_xy, _, _ = compute_HSIC_loss_ind(reps1_curr_label_comb, reps_curr_label_comb, rr_weight)
              else:
                reps1_curr_label_comb = reps1_curr_label_comb - tf.reduce_mean(reps1_curr_label_comb, axis=0, keepdims=True)
                cov1 = tfp.stats.covariance(reps1_curr_label_comb, sample_axis=0, event_axis=1)
                e1, v1 = tf.linalg.eigh(cov1)
                e1_max = tf.reduce_max(e1)
                e1_cutoff = FLAGS.eig_cutoff_factor * e1_max
                reps1_curr_label_comb_diag = tf.linalg.matmul(reps1_curr_label_comb, v1)
                reps1_curr_label_comb_diag_cutoff = tf.boolean_mask(reps1_curr_label_comb_diag, e1 >= e1_cutoff, axis=1)
                reps_curr_label_comb = reps_curr_label_comb - tf.reduce_mean(reps_curr_label_comb, axis=0, keepdims=True)
                cov = tfp.stats.covariance(reps_curr_label_comb, sample_axis=0, event_axis=1)
                e, v = tf.linalg.eigh(cov)
                e_max = tf.reduce_max(e)
                e_cutoff = FLAGS.eig_cutoff_factor * e_max
                reps_curr_label_comb_diag = tf.linalg.matmul(reps_curr_label_comb, v)
                reps_curr_label_comb_diag_cutoff = tf.boolean_mask(reps_curr_label_comb_diag, e >= e_cutoff, axis=1)
                temp_HSIC_loss, temp_HSIC_XY, temp_HSIC_XX, temp_HSIC_YY = compute_HSIC_loss_multivariate(reps1_curr_label_comb_diag_cutoff, reps_curr_label_comb_diag_cutoff, rr_weight)
                temp_th_hsic_loss, temp_th_hsic_xy, _, _ = compute_HSIC_loss_ind(reps1_curr_label_comb_diag_cutoff, reps_curr_label_comb_diag_cutoff, rr_weight)
              HSIC_loss += temp_HSIC_loss/tf.cast(len(reps), tf.float32)
              HSIC_XY += temp_HSIC_XY/tf.cast(len(reps), tf.float32)
              HSIC_XX += temp_HSIC_XX/tf.cast(len(reps), tf.float32)
              HSIC_YY += temp_HSIC_YY/tf.cast(len(reps), tf.float32)
              th_hsic_loss += temp_th_hsic_loss/tf.cast(len(reps), tf.float32)
              th_hsic_xy += temp_th_hsic_xy/tf.cast(len(reps), tf.float32)
            else:
              # Assuming using HSIC with respect to previous models logits
              z = tf.random.normal((tf.shape(reps)[1], tf.shape(reps)[2]))
              z = z/tf.norm(z)
              z = tf.reshape(z, (1,1,-1))
              temp_reps = tf.transpose(reps_curr_label_comb, perm=[0,3,1,2])
              temp_reps = tf.reshape(temp_reps, (tf.shape(temp_reps)[0], tf.shape(temp_reps)[1], -1))
              fin_rep = tf.reduce_sum(z*temp_reps, axis=2)
              HSIC_loss += compute_HSIC_loss_multivariate(reps1_curr_label_comb, fin_rep, rr_weight)/tf.cast(len(reps), tf.float32)

      HSIC_loss = HSIC_loss / num_labels
      HSIC_XY = HSIC_XY / num_labels
      HSIC_XX = HSIC_XX / num_labels
      HSIC_YY = HSIC_YY / num_labels
      th_hsic_xy = th_hsic_xy / num_labels
      th_hsic_loss = th_hsic_loss / num_labels
    else:
      reps1_comb = tf.stop_gradient(tf.random.shuffle(reps1_comb))
      HSIC_loss = compute_HSIC_loss_multivariate(reps1_comb, reps_comb, rr_weight)

    overall_HSIC_loss += HSIC_loss
    overall_HSIC_XY += HSIC_XY
    overall_HSIC_XX += HSIC_XX
    overall_HSIC_YY += HSIC_YY
    overall_th_hsic_xy += th_hsic_xy
    overall_th_hsic_loss += th_hsic_loss

  return overall_HSIC_loss, overall_HSIC_XY, overall_HSIC_XX, overall_HSIC_YY, overall_th_hsic_loss, overall_th_hsic_xy

def compute_seq_HSIC_loss(labels: tf.Tensor,
                          logits: tf.Tensor,
                          reps: tf.Tensor,
                          class_specific_rr: bool = True,
                          reps_arr = None,
                          rr_weight: typing.Optional[float] = None):
  # HSIC loss
  overall_HSIC_loss = tf.constant(0.0)
  overall_HSIC_XY = tf.constant(0.0)
  if rr_weight is None:
    raise ValueError('RR weight not specified.')

  if FLAGS.lin_scale_rr_weight and len(reps_arr) > 0:
    rr_weight = rr_weight/len(reps_arr)

  ctx = tf.distribute.get_replica_context()
  labels_comb = ctx.all_gather(labels, axis=0)
  # if FLAGS.sep_short_direct_branch:
  reps_comb = []
  for i in range(len(reps)):
    reps_comb.append(ctx.all_gather(reps[i], axis=0))
  #else:
  #  reps_comb = ctx.all_gather(reps, axis=0)
  for i, reps1 in enumerate(reps_arr):
    reps1_comb = ctx.all_gather(reps1, axis=0)
    if class_specific_rr:
      if FLAGS.binary_classification:
        num_labels = 2
      else:
        num_labels = logits.shape[1]
      HSIC_loss = tf.constant(0.0)
      HSIC_XY = tf.constant(0.0)
      for label in range(num_labels):
        reps1_curr_label_comb = tf.gather(
            reps1_comb * 1,
            indices=tf.squeeze(
                tf.where(
                    tf.equal(
                        tf.cast(tf.constant(label), dtype='int64'),
                        labels_comb)),
                axis=1))
        if FLAGS.sep_short_direct_branch:
          for j in range(len(reps)):
            reps_curr_label_comb = tf.gather(
                reps_comb[j] * 1,
                indices=tf.squeeze(
                    tf.where(
                        tf.equal(
                            tf.cast(tf.constant(label), dtype='int64'),
                            labels_comb)),
                    axis=1))

            if tf.shape(reps1_curr_label_comb)[0] >= 10 and tf.shape(reps_curr_label_comb)[0] >= 10:
              if FLAGS.use_GAP_HSIC_features:
                HSIC_loss += compute_HSIC_loss_multivariate(reps1_curr_label_comb, reps_curr_label_comb, rr_weight)
              else:
                # Assuming using HSIC with respect to previous models logits
                z = tf.random.normal((tf.shape(reps[j])[1], tf.shape(reps[j])[2]))
                z = z/tf.norm(z)
                z = tf.reshape(z, (1,1,-1))
                temp_reps = tf.transpose(reps_curr_label_comb, perm=[0,3,1,2])
                temp_reps = tf.reshape(temp_reps, (tf.shape(temp_reps)[0], tf.shape(temp_reps)[1], -1))
                fin_rep = tf.reduce_sum(z*temp_reps, axis=2)
                HSIC_loss += compute_HSIC_loss_multivariate(reps1_curr_label_comb, fin_rep, rr_weight)
        else:
          for j in range(len(reps)):
            reps_curr_label_comb = tf.gather(
                reps_comb[j] * 1,
                indices=tf.squeeze(
                    tf.where(
                        tf.equal(
                            tf.cast(tf.constant(label), dtype='int64'),
                            labels_comb)),
                    axis=1))

            # Comment this when worrking on TPU, need to fix this
            #if tf.shape(reps1_curr_label_comb)[0] >= 10 and tf.shape(reps_curr_label_comb)[0] >= 10:
            if FLAGS.use_GAP_HSIC_features or FLAGS.use_random_projections:
              if FLAGS.eig_cutoff_factor == 0.0:
                if FLAGS.use_HSIC_ratio:
                  temp_HSIC_loss, temp_HSIC_XY = compute_HSIC_loss_ratio_multivariate(reps1_curr_label_comb, reps_curr_label_comb, rr_weight)
                else:
                  temp_HSIC_loss, temp_HSIC_XY, _, _ = compute_HSIC_loss_multivariate(reps1_curr_label_comb, reps_curr_label_comb, rr_weight)
              else:
                reps1_curr_label_comb = reps1_curr_label_comb - tf.reduce_mean(reps1_curr_label_comb, axis=0, keepdims=True)
                cov1 = tfp.stats.covariance(reps1_curr_label_comb, sample_axis=0, event_axis=1)
                e1, v1 = tf.linalg.eigh(cov1)
                e1_max = tf.reduce_max(e1)
                e1_cutoff = FLAGS.eig_cutoff_factor * e1_max
                reps1_curr_label_comb_diag = tf.linalg.matmul(reps1_curr_label_comb, v1)
                reps1_curr_label_comb_diag_cutoff = tf.boolean_mask(reps1_curr_label_comb_diag, e1 >= e1_cutoff, axis=1)
                reps_curr_label_comb = reps_curr_label_comb - tf.reduce_mean(reps_curr_label_comb, axis=0, keepdims=True)
                cov = tfp.stats.covariance(reps_curr_label_comb, sample_axis=0, event_axis=1)
                e, v = tf.linalg.eigh(cov)
                e_max = tf.reduce_max(e)
                e_cutoff = FLAGS.eig_cutoff_factor * e_max
                reps_curr_label_comb_diag = tf.linalg.matmul(reps_curr_label_comb, v)
                reps_curr_label_comb_diag_cutoff = tf.boolean_mask(reps_curr_label_comb_diag, e >= e_cutoff, axis=1)
                if FLAGS.use_HSIC_ratio:
                  temp_HSIC_loss, temp_HSIC_XY = compute_HSIC_loss_ratio_multivariate(reps1_curr_label_comb_diag_cutoff, reps_curr_label_comb_diag_cutoff, rr_weight)
                else:
                  temp_HSIC_loss, temp_HSIC_XY, _, _ = compute_HSIC_loss_multivariate(reps1_curr_label_comb_diag_cutoff, reps_curr_label_comb_diag_cutoff, rr_weight)

              HSIC_loss += temp_HSIC_loss/tf.cast(len(reps), tf.float32)
              HSIC_XY += temp_HSIC_XY/tf.cast(len(reps), tf.float32)
            # elif FLAGS.use_random_projections:
            #   # Assuming using prev models logits
            #   final_shape = tf.constant([FLAGS.random_proj_dim], dtype=tf.int32)
            #   final_shape = tf.concat([final_shape, tf.shape(reps[j])[1:]], axis=0)
            #   z = tf.random.normal(final_shape)
            #   z = tf.reshape(z, (FLAGS.random_proj_dim, -1))
            #   z = z/tf.norm(z, axis=1, keepdims=True)
            #   temp_reps = tf.reshape(reps_curr_label_comb, (tf.shape(reps_curr_label_comb)[0], -1))
            #   # Use the method of projection only if projecting onto a sufficiently low dimensional space
            #   # as the method adds noise in the optimization process
            #   # final_reps = tf.cond(tf.shape(temp_reps)[1] > 2*FLAGS.random_proj_dim, lambda: tf.matmul(temp_reps, tf.transpose(z)),
            #   #                     lambda: temp_reps)
            #   # if tf.shape(temp_reps)[1] > 2*FLAGS.random_proj_dim:
            #   #   final_reps =
            #   # else:
            #   #   final_reps = temp_reps
            #   # HSIC_loss += compute_HSIC_loss_multivariate(reps1_curr_label_comb, final_reps, rr_weight)/tf.cast(len(reps), tf.float32)
            #   loss_curr = tf.cond(tf.shape(temp_reps)[1] > 2*FLAGS.random_proj_dim,
            #                       lambda: compute_HSIC_loss_multivariate(reps1_curr_label_comb, tf.matmul(temp_reps, tf.transpose(z)), rr_weight),
            #                       lambda: compute_HSIC_loss_multivariate(reps1_curr_label_comb, temp_reps, rr_weight))
            #   HSIC_loss += loss_curr/tf.cast(len(reps), tf.float32)
            else:
              # Assuming using HSIC with respect to previous models logits
              z = tf.random.normal((tf.shape(reps[j])[1], tf.shape(reps[j])[2]))
              z = z/tf.norm(z)
              z = tf.reshape(z, (1,1,-1))
              temp_reps = tf.transpose(reps_curr_label_comb, perm=[0,3,1,2])
              temp_reps = tf.reshape(temp_reps, (tf.shape(temp_reps)[0], tf.shape(temp_reps)[1], -1))
              fin_rep = tf.reduce_sum(z*temp_reps, axis=2)
              HSIC_loss += compute_HSIC_loss_multivariate(reps1_curr_label_comb, fin_rep, rr_weight)/tf.cast(len(reps), tf.float32)

      HSIC_loss = HSIC_loss / num_labels
      HSIC_XY = HSIC_XY / num_labels
    else:
      HSIC_loss = compute_HSIC_loss_multivariate(reps1_comb, reps_comb, rr_weight)

    overall_HSIC_loss += HSIC_loss
    overall_HSIC_XY += HSIC_XY

  return overall_HSIC_loss, overall_HSIC_XY

def est_seq_HSIC_logits_loss_ind(labels: tf.Tensor,
                        logits: tf.Tensor,
                        class_specific_rr: bool = True,
                        logits_arr = None,
                        rr_weight: typing.Optional[float] = None):
  overall_HSIC_ind_loss = tf.constant(0.0)
  overall_HSIC_XY = tf.constant(0.0)
  overall_HSIC_XX = tf.constant(0.0)
  overall_HSIC_YY = tf.constant(0.0)
  overall_th_hsic_loss = tf.constant(0.0)
  overall_th_hsic_xy = tf.constant(0.0)
  if rr_weight is None:
    raise ValueError('RR weight not specified.')

  if FLAGS.lin_scale_rr_weight and len(logits_arr) > 0:
    rr_weight = rr_weight/len(logits_arr)

  ctx = tf.distribute.get_replica_context()
  labels_comb = ctx.all_gather(labels, axis=0)
  logits_comb = ctx.all_gather(logits, axis=0)

  for i, logits1 in enumerate(logits_arr):
    logits1_comb = ctx.all_gather(logits1, axis=0)
    HSIC_loss = tf.constant(0.0)
    HSIC_XY = tf.constant(0.0)
    HSIC_XX = tf.constant(0.0)
    HSIC_YY = tf.constant(0.0)
    th_hsic_loss = tf.constant(0.0)
    th_hsic_xy = tf.constant(0.0)
    if class_specific_rr:
      num_labels = logits.shape[1]
      for label in range(num_labels):
        logits1_curr_label_comb = tf.gather(
            logits1_comb * 1,
            indices=tf.squeeze(
                tf.where(
                    tf.equal(
                        tf.cast(tf.constant(label), dtype='int64'),
                        labels_comb)),
                axis=1))
        logits_curr_label_comb = tf.gather(
            logits_comb * 1,
            indices=tf.squeeze(
                tf.where(
                    tf.equal(
                        tf.cast(tf.constant(label), dtype='int64'),
                        labels_comb)),
                axis=1))

        if tf.shape(logits_curr_label_comb)[0] >= 10 and tf.shape(logits1_curr_label_comb)[0] >= 10:
          if FLAGS.use_HSIC_diff:
            temp_logits_1 = logits1_curr_label_comb - tf.reshape(logits1_curr_label_comb[:, label], (-1,1))
            # temp_logits_1 = -1.0*temp_logits_1
            # temp_logits_1 = tf.math.log(tf.reduce_sum(tf.exp(temp_logits_1), axis=1) - 1.0)
            temp_logits_1 = tf.reduce_sum(temp_logits_1, axis=1, keepdims=True)
            temp_logits = logits_curr_label_comb - tf.reshape(logits_curr_label_comb[:, label], (-1,1))
            # temp_logits = -1.0*temp_logits
            # temp_logits = tf.math.log(tf.reduce_sum(tf.exp(temp_logits), axis=1) - 1.0)
            temp_logits = tf.reduce_sum(temp_logits, axis=1, keepdims=True)
            temp_logits = tf.stop_gradient(tf.random.shuffle(temp_logits))
            temp_HSIC_loss, temp_HSIC_XY, temp_HSIC_XX, temp_HSIC_YY = compute_HSIC_loss_multivariate(temp_logits_1, temp_logits, rr_weight)
            temp_th_hsic_loss, temp_th_hsic_xy, _, _ = compute_HSIC_loss_ind(temp_logits_1, temp_logits, rr_weight)
          else:
            temp_logits = tf.reshape(logits1_curr_label_comb[:, label], (-1,1))
            temp_logits = tf.stop_gradient(tf.random.shuffle(temp_logits))
            temp_HSIC_loss, temp_HSIC_XY, temp_HSIC_XX, temp_HSIC_YY = compute_HSIC_loss_multivariate(temp_logits, tf.reshape(logits_curr_label_comb[:, label], (-1,1)), rr_weight)
            temp_th_hsic_loss, temp_th_hsic_xy, _, _ = compute_HSIC_loss_ind(temp_logits, tf.reshape(logits_curr_label_comb[:, label], (-1,1)), rr_weight)
          HSIC_loss += temp_HSIC_loss
          HSIC_XY += temp_HSIC_XY
          HSIC_XX += temp_HSIC_XX
          HSIC_YY += temp_HSIC_YY
          th_hsic_xy += temp_th_hsic_xy
          th_hsic_loss += temp_th_hsic_loss

      HSIC_loss = HSIC_loss / num_labels
      HSIC_XY = HSIC_XY / num_labels
      HSIC_XX = HSIC_XX / num_labels
      HSIC_YY = HSIC_YY / num_labels
      th_hsic_xy = th_hsic_xy / num_labels
      th_hsic_loss = th_hsic_loss / num_labels
    else:
      temp_labels_comb = tf.reshape(labels_comb, (-1,1))
      if FLAGS.use_HSIC_diff:
        temp_logits_1 = logits1_comb - tf.reshape(tf.gather_nd(logits1_comb, temp_labels_comb, batch_dims=1), (-1,1))
        temp_logits_1 = tf.reduce_sum(temp_logits_1, axis=1, keepdims=True)
        temp_logits = logits_comb - tf.reshape(tf.gather_nd(logits_comb, temp_labels_comb, batch_dims=1), (-1,1))
        temp_logits = tf.reduce_sum(temp_logits, axis=1, keepdims=True)
        temp_logits = tf.stop_gradient(tf.random.shuffle(temp_logits))
        temp_HSIC_loss, temp_HSIC_XY, temp_HSIC_XX, temp_HSIC_YY = compute_HSIC_loss_multivariate(temp_logits_1, temp_logits, rr_weight)
        temp_th_hsic_loss, temp_th_hsic_xy, _, _ = compute_HSIC_loss_ind(temp_logits_1, temp_logits, rr_weight)
      else:
        temp_logits_1 = tf.reshape(tf.gather_nd(logits1_comb, temp_labels_comb, batch_dims=1), (-1,1))
        temp_logits = tf.reshape(tf.gather_nd(logits_comb, temp_labels_comb, batch_dims=1), (-1,1))
        temp_logits = tf.stop_gradient(tf.random.shuffle(temp_logits))
        temp_HSIC_loss, temp_HSIC_XY, temp_HSIC_XX, temp_HSIC_YY = compute_HSIC_loss_multivariate(temp_logits, temp_logits_1, rr_weight)
        temp_th_hsic_loss, temp_th_hsic_xy, _, _ = compute_HSIC_loss_ind(temp_logits, temp_logits_1, rr_weight)
      HSIC_loss += temp_HSIC_loss
      HSIC_XY += temp_HSIC_XY
      HSIC_XX += temp_HSIC_XX
      HSIC_YY += temp_HSIC_YY
      th_hsic_xy += temp_th_hsic_xy
      th_hsic_loss += temp_th_hsic_loss

    overall_HSIC_ind_loss += HSIC_loss
    overall_HSIC_XY += HSIC_XY
    overall_HSIC_XX += HSIC_XX
    overall_HSIC_YY += HSIC_YY
    overall_th_hsic_xy += th_hsic_xy
    overall_th_hsic_loss += th_hsic_loss

  return overall_HSIC_ind_loss, overall_HSIC_XY, overall_HSIC_XX, overall_HSIC_YY, overall_th_hsic_loss, overall_th_hsic_xy

def compute_seq_MI_loss(labels: tf.Tensor,
                        logits: tf.Tensor,
                        class_specific_rr: bool = True,
                        logits_arr = None,
                        rr_weight: typing.Optional[float] = None,
                        num_sq = True):
  # MI loss
  overall_MI_loss = tf.constant(0.0)
  overall_HSIC_XY = tf.constant(0.0)
  if rr_weight is None:
    raise ValueError('RR weight not specified.')

  if FLAGS.lin_scale_rr_weight and len(logits_arr) > 0:
    rr_weight = rr_weight/len(logits_arr)

  ctx = tf.distribute.get_replica_context()
  labels_comb = ctx.all_gather(labels, axis=0)
  logits_comb = ctx.all_gather(logits, axis=0)
  probs_comb = tf.nn.softmax(logits_comb, axis=1)

  for i, logits1 in enumerate(logits_arr):
    logits1_comb = ctx.all_gather(logits1, axis=0)
    probs1_comb = tf.nn.softmax(logits1_comb, axis=1)
    if class_specific_rr:
      num_labels = logits.shape[1]
      MI_loss = tf.constant(0.0)
      HSIC_XY = tf.constant(0.0)
      for label in range(num_labels):
        probs1_curr_label_comb = tf.gather(
            probs1_comb * 1,
            indices=tf.squeeze(
                tf.where(
                    tf.equal(
                        tf.cast(tf.constant(label), dtype='int64'),
                        labels_comb)),
                axis=1))
        probs_curr_label_comb = tf.gather(
            probs_comb * 1,
            indices=tf.squeeze(
                tf.where(
                    tf.equal(
                        tf.cast(tf.constant(label), dtype='int64'),
                        labels_comb)),
                axis=1))
        logits1_curr_label_comb = tf.gather(
            logits1_comb * 1,
            indices=tf.squeeze(
                tf.where(
                    tf.equal(
                        tf.cast(tf.constant(label), dtype='int64'),
                        labels_comb)),
                axis=1))
        logits_curr_label_comb = tf.gather(
            logits_comb * 1,
            indices=tf.squeeze(
                tf.where(
                    tf.equal(
                        tf.cast(tf.constant(label), dtype='int64'),
                        labels_comb)),
                axis=1))

        if FLAGS.use_disagr_loss:
          MI_loss += compute_disagreement(probs1_curr_label_comb, probs_curr_label_comb, rr_weight)
        elif FLAGS.use_HSIC_loss:
          if tf.shape(logits_curr_label_comb)[0] >= 10 and tf.shape(logits1_curr_label_comb)[0] >= 10:
            if FLAGS.use_HSIC_diff:
              temp_logits_1 = logits1_curr_label_comb - tf.reshape(logits1_curr_label_comb[:, label], (-1,1))
              # temp_logits_1 = -1.0*temp_logits_1
              # temp_logits_1 = tf.math.log(tf.reduce_sum(tf.exp(temp_logits_1), axis=1) - 1.0)
              temp_logits_1 = tf.reduce_sum(temp_logits_1, axis=1)
              temp_logits = logits_curr_label_comb - tf.reshape(logits_curr_label_comb[:, label], (-1,1))
              # temp_logits = -1.0*temp_logits
              # temp_logits = tf.math.log(tf.reduce_sum(tf.exp(temp_logits), axis=1) - 1.0)
              temp_logits = tf.reduce_sum(temp_logits, axis=1)
              if FLAGS.use_HSIC_ratio:
                temp_logits = tf.reshape(temp_logits, (-1,1))
                temp_logits_1 = tf.reshape(temp_logits_1, (-1,1))
                temp_MI_loss, temp_HSIC_XY = compute_HSIC_loss_ratio_multivariate(temp_logits_1, temp_logits, rr_weight)
              else:
                temp_MI_loss,temp_HSIC_XY,_,_ = compute_HSIC_loss(temp_logits_1, temp_logits, rr_weight)
            else:
              if FLAGS.use_HSIC_ratio:
                temp_logits_1 = tf.reshape(logits1_curr_label_comb[:, label], (-1,1))
                temp_logits = tf.reshape(logits_curr_label_comb[:, label], (-1,1))
                temp_MI_loss, temp_HSIC_XY = compute_HSIC_loss_ratio_multivariate(temp_logits_1, temp_logits, rr_weight)
              else:
                temp_MI_loss,temp_HSIC_XY,_,_ = compute_HSIC_loss(logits1_curr_label_comb[:, label], logits_curr_label_comb[:, label], rr_weight)
            MI_loss += temp_MI_loss
            HSIC_XY += temp_HSIC_XY
        else:
          MI_loss += compute_MI(probs1_curr_label_comb, probs_curr_label_comb, rr_weight, num_sq=num_sq)

      MI_loss = MI_loss / num_labels
      HSIC_XY = HSIC_XY / num_labels
    else:
      if FLAGS.use_disagr_loss:
        MI_loss = compute_disagreement(probs1_comb, probs_comb, rr_weight)
      elif FLAGS.use_HSIC_loss:
        temp_labels_comb = tf.reshape(labels_comb, (-1,1))
        if FLAGS.use_HSIC_diff:
          temp_logits_1 = logits1_comb - tf.reshape(tf.gather_nd(logits1_comb, temp_labels_comb, batch_dims=1), (-1,1))
          temp_logits_1 = tf.reduce_sum(temp_logits_1, axis=1)
          temp_logits = logits_comb - tf.reshape(tf.gather_nd(logits_comb, temp_labels_comb, batch_dims=1), (-1,1))
          temp_logits = tf.reduce_sum(temp_logits, axis=1)
          if FLAGS.use_HSIC_ratio:
            temp_logits = tf.reshape(temp_logits, (-1,1))
            temp_logits_1 = tf.reshape(temp_logits_1, (-1,1))
            temp_MI_loss, temp_HSIC_XY = compute_HSIC_loss_ratio_multivariate(temp_logits_1, temp_logits, rr_weight)
          else:
            temp_MI_loss,temp_HSIC_XY,_,_ = compute_HSIC_loss(temp_logits_1, temp_logits, rr_weight)
        else:
          if FLAGS.use_HSIC_ratio:
            temp_logits_1 = tf.reshape(tf.gather_nd(logits1_comb, temp_labels_comb, batch_dims=1), (-1,1))
            temp_logits = tf.reshape(tf.gather_nd(logits_comb, temp_labels_comb, batch_dims=1), (-1,1))
            temp_MI_loss, temp_HSIC_XY = compute_HSIC_loss_ratio_multivariate(temp_logits_1, temp_logits, rr_weight)
          else:
            temp_logits_1 = tf.reshape(tf.gather_nd(logits1_comb, temp_labels_comb, batch_dims=1), (-1,1))
            temp_logits = tf.reshape(tf.gather_nd(logits_comb, temp_labels_comb, batch_dims=1), (-1,1))
            temp_MI_loss,temp_HSIC_XY,_,_ = compute_HSIC_loss(temp_logits_1, temp_logits, rr_weight)
        MI_loss = temp_MI_loss
        HSIC_XY = temp_HSIC_XY
      else:
        MI_loss = compute_MI(probs1_comb, probs_comb, rr_weight)

    overall_MI_loss += MI_loss
    overall_HSIC_XY += HSIC_XY

  return overall_MI_loss, overall_HSIC_XY

def compute_loss(labels: tf.Tensor,
                 logits: tf.Tensor,
                 batch_size: int,
                 use_rr_loss: bool = False,
                 class_specific_rr: bool = True,
                 reps: typing.Optional[tf.Tensor] = None,
                 rr_weight: typing.Optional[float] = None):
  """Function to compute Categorical Cross Entropy + redundancy reduction loss.

  Args:
    labels: True labels in sparse form.
    logits: logits as logits.
    batch_size: Global batch size.
    use_rr_loss: Whether to use RR loss.
    class_specific_rr: Whether to use class specific RR loss.
    reps: Representations on which we apply RR loss.
    rr_weight: effective weight for RR loss.
  Returns:
    Loss value.
  """
  # CE loss function
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  per_example_loss = loss_fn(labels, logits)
  total_loss = tf.nn.compute_average_loss(
      per_example_loss, global_batch_size=batch_size)

  redundancy_loss = compute_batch_redundancy_loss(labels, logits, class_specific_rr, reps, rr_weight)
  if use_rr_loss:
    total_loss += redundancy_loss

  return  total_loss, redundancy_loss

def compute_multihead_loss(labels: tf.Tensor,
                 logits_arr,
                 batch_size: int,
                 use_rr_loss: bool = False,
                 class_specific_rr: bool = True,
                 reps_arr = None,
                 rr_weight: typing.Optional[float] = None,
                 labels2 = None,
                 logits2 = None):
  """Function to compute Categorical Cross Entropy + redundancy reduction loss.

  Args:
    labels: True labels in sparse form.
    logits: logits as logits.
    batch_size: Global batch size.
    use_rr_loss: Whether to use RR loss.
    class_specific_rr: Whether to use class specific RR loss.
    reps: Representations on which we apply RR loss.
    rr_weight: effective weight for RR loss.
  Returns:
    Loss value.
  """
  # CE loss function
  total_loss = 0.0
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  for logits in logits_arr:
    per_example_loss = loss_fn(labels, logits)
    total_loss += tf.nn.compute_average_loss(
        per_example_loss, global_batch_size=batch_size)

  if FLAGS.use_MI_loss or FLAGS.use_disagr_loss or FLAGS.use_HSIC_loss:
    if FLAGS.use_val_for_MI:
      redundancy_loss = compute_multihead_MI_loss(labels2, logits2, class_specific_rr, rr_weight)
    else:
      redundancy_loss = compute_multihead_MI_loss(labels, logits_arr, class_specific_rr, rr_weight)
  else:
    redundancy_loss = compute_multihead_batch_redundancy_loss(labels, logits_arr, class_specific_rr, reps_arr, rr_weight)

  if use_rr_loss:
    if FLAGS.lowerbound_rr:
      num_classes = logits_arr[0].shape[1]
      num_features = reps_arr[0].shape[1]
      if class_specific_rr:
        exp_value = rr_weight*(num_features**2)*(num_classes*1.0)/(batch_size*1.0)
      else:
        exp_value = rr_weight*(num_features**2)/(batch_size*1.0)
      lowerbound = tf.nn.scale_regularization_loss(FLAGS.lowerbound_factor * exp_value)
      if redundancy_loss > lowerbound:
        total_loss += redundancy_loss
    else:
      total_loss += redundancy_loss

  return  total_loss, redundancy_loss

def compute_seq_multihead_loss(labels: tf.Tensor,
                 logits: tf.Tensor,
                 batch_size: int,
                 use_rr_loss: bool = False,
                 class_specific_rr: bool = True,
                 reps: typing.Optional[tf.Tensor] = None,
                 reps_arr = None,
                 rr_weight: typing.Optional[float] = None,
                 logits_arr = None,
                 logits2 = None,
                 labels2 = None):
  """Function to compute Categorical Cross Entropy + redundancy reduction loss.

  Args:
    labels: True labels in sparse form.
    logits: logits as logits.
    batch_size: Global batch size.
    use_rr_loss: Whether to use RR loss.
    class_specific_rr: Whether to use class specific RR loss.
    reps: Representations on which we apply RR loss.
    rr_weight: effective weight for RR loss.
  Returns:
    Loss value.
  """
  # CE loss function
  total_loss = 0.0
  if FLAGS.binary_classification:
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    per_example_loss = loss_fn(tf.reshape(labels, (-1,1)), logits)
  else:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    per_example_loss = loss_fn(labels, logits)
  total_loss += tf.nn.compute_average_loss(
      per_example_loss, global_batch_size=batch_size)

  if FLAGS.use_exp_var_loss:
    redundancy_loss = compute_seq_multihead_exp_variance_loss(labels, logits, class_specific_rr, reps, reps_arr, rr_weight)
  elif FLAGS.use_MI_loss or FLAGS.use_disagr_loss or FLAGS.use_HSIC_loss:
    if FLAGS.use_HSIC_loss and FLAGS.use_HSIC_on_features:
      redundancy_loss, _ = compute_seq_HSIC_loss(labels, logits, reps, class_specific_rr, reps_arr, rr_weight)
    elif FLAGS.use_val_for_MI:
      redundancy_loss,_ = compute_seq_MI_loss(labels2, logits2, class_specific_rr, logits_arr, rr_weight)
    else:
      redundancy_loss,_ = compute_seq_MI_loss(labels, logits, class_specific_rr, logits_arr, rr_weight)
  elif FLAGS.use_logit_decorr:
    redundancy_loss = compute_seq_multihead_batch_redundancy_loss(labels, logits, class_specific_rr, logits, logits_arr, rr_weight)
  elif FLAGS.use_prob_decorr:
    prob1 = tf.nn.softmax(logits, axis=1)
    prob_arr = []
    for logit in logits_arr:
      prob_arr.append(tf.nn.softmax(logit, axis=1))
    redundancy_loss = compute_seq_multihead_batch_redundancy_loss(labels, logits, class_specific_rr, prob1, prob_arr, rr_weight)
  else:
    redundancy_loss = compute_seq_multihead_batch_redundancy_loss(labels, logits, class_specific_rr, reps, reps_arr, rr_weight)

  if use_rr_loss:
    if FLAGS.lowerbound_rr:
      num_classes = logits.shape[1]
      num_features = reps_arr[0].shape[1]
      if class_specific_rr:
        exp_value = rr_weight*(num_features**2)*(num_classes*1.0)/(batch_size*1.0)
      else:
        exp_value = rr_weight*(num_features**2)/(batch_size*1.0)
      lowerbound = tf.nn.scale_regularization_loss(FLAGS.lowerbound_factor * exp_value)
      if redundancy_loss > lowerbound:
        total_loss += redundancy_loss
    else:
      total_loss += redundancy_loss

  return  total_loss, redundancy_loss

def compute_seq_GAN_loss(labels: tf.Tensor,
                 logits: tf.Tensor,
                 batch_size: int,
                 use_rr_loss: bool = False,
                 reps_curr = None,
                 reps_prev = None,
                 rr_weight: typing.Optional[float] = None,
                 ):
  """Function to compute Categorical Cross Entropy + redundancy reduction loss.

  Args:
    labels: True labels in sparse form.
    logits: logits as logits.
    batch_size: Global batch size.
    use_rr_loss: Whether to use RR loss.
    class_specific_rr: Whether to use class specific RR loss.
    reps: Representations on which we apply RR loss.
    rr_weight: effective weight for RR loss.
  Returns:
    Loss value.
  """
  # CE loss function
  total_loss = 0.0
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  per_example_loss = loss_fn(labels, logits)
  total_loss += tf.nn.compute_average_loss(
      per_example_loss, global_batch_size=batch_size)

  ctx = tf.distribute.get_replica_context()
  reps_curr_comb = []
  reps_prev_comb = []
  for i in range(len(reps_curr)):
    reps_curr_comb.append(ctx.all_gather(reps_curr[i], axis=0))
  for i in range(len(reps_prev)):
    temp_reps_comb = []
    for j in range(len(reps_prev[i])):
      temp_reps_comb.append(ctx.all_gather(reps_prev[i][j], axis=0))
    reps_prev_comb.append(temp_reps_comb)
  num_labels = tf.shape(logits)[1]
  redundancy_loss = 0.0
  for i in range(len(reps_prev)):
    for j in range(len(reps_curr_comb)):
      if tf.shape(reps_curr_comb[j])[0] >= 10 and tf.shape(reps_prev_comb[i][j])[0] >= 10:
        tf.print('reps_curr_comb', tf.shape(reps_curr_comb[j]))
        tf.print('reps_prev_comb', tf.shape(reps_prev_comb[i][j]))
        redundancy_loss += 0.0
        #redundancy_loss += compute_cross_redundancy_loss_GPU(reps_curr_comb[j], reps_prev_comb[i][j], rr_weight)
  redundancy_loss  = redundancy_loss/tf.cast(len(reps_prev)*num_labels, tf.float32)

  if use_rr_loss:
    total_loss += redundancy_loss

  return  total_loss, redundancy_loss
