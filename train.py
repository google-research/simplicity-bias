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

"""Trains a classifier and evaluates finetuning performance."""

from absl import flags
from absl import logging
import numpy as np  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow import keras


from loss_fns import *
from diverse_metrics import *

FLAGS = flags.FLAGS


def train(
    model,
    ds_train,
    ds_val,
    ds_test,
    train_len,
    optimizer,
    run=0,
    only_head=False,
    only_linear_head=False,
    finetune=False,  #extra flag to decide number of epochs and disable rr loss and clipping
    extra_dsets=[],
    num_heads=1,
    ckpt_manager=None,
    prev_models=[],
    MI_ds=None,
    prefix='',
    add_gauss_noise=False,
    optimizer2=None,
    W=None):
  """Train function."""

  strategy = tf.distribute.get_strategy()

  if FLAGS.project_out_rank > 0:
    global_proj = tf.Variable(tf.random.normal((1, FLAGS.project_out_rank), dtype=tf.float64), shape=[None,FLAGS.project_out_rank], dtype=tf.float64)

  # Required as need to access model's trainable variables
  # depending on only_head argument
  def instantiate_model(inputs):
    images = inputs['image']
    if FLAGS.project_out_rank > 0:
      temp_images = tf.reshape(images, (tf.shape(images)[0], -1))
      temp_proj = tf.random.normal((tf.shape(temp_images)[1], FLAGS.project_out_rank), dtype=tf.float64)
      temp_proj = temp_proj/tf.norm(temp_proj, axis=0, keepdims=True)
      global_proj.assign(temp_proj)
    logits = model(images, training=False)
    for prev_model in prev_models:
      logits = prev_model(images, training=False)
    return logits

  # Required as need to access model's trainable variables
  # depending on only_head argument
  def instantiate_model_init(inputs, init):
    images = inputs['image']
    if FLAGS.project_out_rank > 0:
      temp_images = tf.reshape(images, (tf.shape(images)[0], -1))
      temp_proj = tf.random.normal((tf.shape(temp_images)[1], FLAGS.project_out_rank), dtype=tf.float64)
      temp_proj = temp_proj/tf.norm(temp_proj, axis=0, keepdims=True)
      global_proj.assign(temp_proj)
    logits = model(images, training=False, init=init)
    for prev_model in prev_models:
      logits = prev_model(images, training=False)
    return logits

  @tf.function
  def distributed_instantiate_model(dataset_inputs):
    return strategy.run(instantiate_model, args=(dataset_inputs,))

  @tf.function
  def distributed_instantiate_model_init(dataset_inputs, init):
    return strategy.run(instantiate_model_init, args=(dataset_inputs, init))

  logging.info('Instantiating the model.')
  for i, data in enumerate(ds_train):
    logging.info('Iteration %d', i)
    if i==0:
      distributed_instantiate_model(data)
    elif i==1:
      distributed_instantiate_model_init(data, True)
    if i == 1:
      break
  logging.info('Finished instantiating the model.')

  use_rr_loss = FLAGS.use_rr_loss
  trainable_variables = model.trainable_variables
  prev_models_trainable_variables = []
  for i in range(len(prev_models)):
    prev_models_trainable_variables.append(prev_models[i].trainable_variables)
  clip = FLAGS.clip_norm
  weight_decay = FLAGS.weight_decay
  if finetune:
    use_rr_loss = False
    clip = None
    weight_decay = 0.0
  if only_head:
    # use_rr_loss = False
    trainable_variables = []
    for ind, layer in enumerate(model.layers):
      if ind != 0:
        trainable_variables.append(layer.trainable_variables)
    trainable_variables = [
        item for sublist in trainable_variables for item in sublist
    ]
    #model.layers[0].trainable = False
    prev_models_trainable_variables = []
    for i in range(len(prev_models)):
      prev_trainable_variables = []
      for ind, layer in enumerate(prev_models[i].layers):
        if ind != 0:
          prev_trainable_variables.append(layer.trainable_variables)
      prev_trainable_variables = [
          item for sublist in prev_trainable_variables for item in sublist
      ]
      prev_models_trainable_variables.append(prev_trainable_variables)
    if only_linear_head:
      # for ind, layer in enumerate(model.layers):
      #   if ind != len(model.layers) - 1:
      #     layer.trainable = False
      trainable_variables = model.layers[-1].trainable_variables
      prev_models_trainable_variables = []
      for i in range(len(prev_models)):
        prev_models_trainable_variables.append(prev_models[i].layers[-1].trainable_variables)

  tf.print('trainable_variables:', trainable_variables)
  for i in range(len(prev_models)):
    tf.print('prev_{}_trainable'.format(str(i)), prev_models_trainable_variables[i])

  project_out_mat = None
  if FLAGS.project_out_prev_w:
    for i in range(len(prev_models)):
      for var in prev_models_trainable_variables[i]:
        if 'kernel' in var.name:
          uncentered_outer_prod = tf.linalg.matmul(var, tf.transpose(var))
          e1, v1 = tf.linalg.eigh(uncentered_outer_prod)
          if project_out_mat is None:
            project_out_mat = v1[:, -FLAGS.project_out_vecs[i]:]
          else:
            project_out_mat = tf.concat([project_out_mat, v1[:, -FLAGS.project_out_vecs[i]:]], axis=1)
          break

  def train_step(inputs, inputs2, use_rr_loss):
    images = inputs['image']
    labels = inputs['label']

    if FLAGS.project_out_rank > 0 and not finetune:
      tf.print('global_proj', global_proj)
      if W is not None:
        tf.print('dot_prod', tf.linalg.matmul(W, global_proj/tf.norm(global_proj, axis=0, keepdims=True)))
        temp_dot = tf.linalg.matmul(W, global_proj/tf.norm(global_proj, axis=0, keepdims=True))
        for i in range(W.shape[0]):
          proj_dot_prod[i].update_state(temp_dot[i,0])

    if add_gauss_noise and not FLAGS.measure_feat_robust:
      std_devs = FLAGS.max_gauss_noise_std * tf.random.uniform([tf.shape(images)[0]])
      final_shape = tf.constant([-1], dtype=tf.int32)
      final_shape = tf.concat([final_shape, tf.ones([tf.rank(images)-1], dtype=tf.int32)], axis=0)
      z = tf.reshape(std_devs, final_shape) * tf.random.normal(tf.shape(images))
      images = images + tf.cast(z, images.dtype)

    with tf.GradientTape(persistent=True) as tape:
      if FLAGS.project_out_rank > 0 and not finetune:
        temp_images = tf.reshape(images, (tf.shape(images)[0],-1))
        pinv = tf.linalg.pinv(tf.linalg.matmul(tf.transpose(global_proj), global_proj))
        proj_mat = tf.linalg.matmul(global_proj, tf.linalg.matmul(pinv, tf.transpose(global_proj)))
        proj_x = tf.linalg.matmul(proj_mat, tf.transpose(temp_images))
        temp_images = tf.transpose(tf.transpose(temp_images) - (1.0-FLAGS.project_out_factor)*proj_x)
        images = tf.reshape(temp_images, tf.shape(images))
      reps, logits, feats = model(
          images,
          training=True,
          only_head=only_head,
          only_linear_head=only_linear_head,
          project_out_w=FLAGS.project_out_w,
          project_out_mat=project_out_mat,
          gauss_noise_feats=add_gauss_noise and FLAGS.measure_feat_robust,
          sigma=FLAGS.max_gauss_noise_std,
          rand_sigma=True)
      if FLAGS.use_HSIC_loss and FLAGS.use_HSIC_on_features:
        reps = feats
      if num_heads > 1:
        labels2 = None
        logits2 = None
        if FLAGS.use_val_for_MI:
          images2 = inputs2['image']
          labels2 = inputs2['label']
          _, logits2, _ = model(images2, training=False)
        loss, red_loss = compute_multihead_loss(
            labels,
            logits,
            FLAGS.train_batch_size,
            use_rr_loss=use_rr_loss,
            reps_arr=reps,
            rr_weight=FLAGS.rr_weight,
            class_specific_rr=FLAGS.class_specific_rr_loss,
            labels2 = labels2,
            logits2 = logits2)
      elif FLAGS.use_seq_rr:
        reps_arr = []
        logits_arr = []
        labels2 = None
        if FLAGS.use_val_for_MI:
          images2 = inputs2['image']
          labels2 = inputs2['label']
        for curr_model in prev_models:
          if FLAGS.use_val_for_MI:
            curr_reps, curr_logits, curr_feats = curr_model(images2, training=False)
          else:
            curr_reps, curr_logits, curr_feats = curr_model(images, training=False)
          if FLAGS.use_HSIC_loss and FLAGS.use_HSIC_on_features:
            if FLAGS.use_prev_logits_HSIC_features:
              reps_arr.append(curr_logits)
            else:
              reps_arr.append(curr_feats)
          else:
            reps_arr.append(curr_reps)
          logits_arr.append(curr_logits)
        logits2 = None
        if FLAGS.use_val_for_MI:
          _, logits2, _ = model(images2, training=False)
        loss, red_loss = compute_seq_multihead_loss(
            labels,
            logits,
            FLAGS.train_batch_size,
            use_rr_loss=use_rr_loss,
            class_specific_rr=FLAGS.class_specific_rr_loss,
            reps=reps,
            reps_arr=reps_arr,
            rr_weight=FLAGS.rr_weight,
            logits_arr=logits_arr,
            logits2 = logits2,
            labels2 = labels2)
        if FLAGS.use_HSIC_loss:
          if FLAGS.use_HSIC_on_features:
            HSIC_loss_ind, HSIC_XY_ind, HSIC_XX, HSIC_YY, th_hsic_loss_ind, th_hsic_xy_ind = est_seq_HSIC_loss_ind(labels, logits, reps, FLAGS.class_specific_rr_loss, reps_arr, FLAGS.rr_weight)
            _, HSIC_XY = compute_seq_HSIC_loss(labels, logits, reps, FLAGS.class_specific_rr_loss, reps_arr, FLAGS.rr_weight)
          else:
            HSIC_loss_ind, HSIC_XY_ind, HSIC_XX, HSIC_YY, th_hsic_loss_ind, th_hsic_xy_ind = est_seq_HSIC_logits_loss_ind(labels, logits, FLAGS.class_specific_rr_loss, logits_arr, FLAGS.rr_weight)
            _, HSIC_XY = compute_seq_MI_loss(labels, logits, FLAGS.class_specific_rr_loss, logits_arr, FLAGS.rr_weight)
          ctx = tf.distribute.get_replica_context()
          if FLAGS.use_HSIC_ratio:
            act_ratio = ctx.num_replicas_in_sync * red_loss / FLAGS.rr_weight
            act_ind_loss = ctx.num_replicas_in_sync * th_hsic_loss_ind
            train_HSIC_loss.update_state(act_ind_loss * act_ratio)
          else:
            train_HSIC_loss.update_state(ctx.num_replicas_in_sync * red_loss)
          train_HSIC_ind_loss.update_state(ctx.num_replicas_in_sync * HSIC_loss_ind)
          train_HSIC_th_ind_loss.update_state(ctx.num_replicas_in_sync * th_hsic_loss_ind)
          train_HSIC_XY.update_state(ctx.num_replicas_in_sync * HSIC_XY)
          train_HSIC_XY_ind.update_state(ctx.num_replicas_in_sync * HSIC_XY_ind)
          train_HSIC_XY_th_ind.update_state(ctx.num_replicas_in_sync * th_hsic_xy_ind)
          train_HSIC_XX.update_state(ctx.num_replicas_in_sync * HSIC_XX)
          train_HSIC_YY.update_state(ctx.num_replicas_in_sync * HSIC_YY)
        if FLAGS.use_num_sq_MI:
          if FLAGS.use_val_for_MI:
            red_loss_2 = compute_seq_MI_loss(labels2, logits2, FLAGS.class_specific_rr_loss, logits_arr, FLAGS.rr_weight, num_sq=False)
          else:
            red_loss_2 = compute_seq_MI_loss(labels, logits, FLAGS.class_specific_rr_loss, logits_arr, FLAGS.rr_weight, num_sq=False)
      else:
        loss, red_loss = compute_loss(
            labels,
            logits,
            FLAGS.train_batch_size,
            use_rr_loss=use_rr_loss,
            reps=reps,
            rr_weight=FLAGS.rr_weight,
            class_specific_rr=FLAGS.class_specific_rr_loss)
      if use_rr_loss:
        ce_loss = loss - red_loss
      else:
        ce_loss = loss
      grad_asc_loss = -1.0*ce_loss
      logit_loss = FLAGS.logit_decay * tf.nn.l2_loss(logits)
      loss = loss + logit_loss
      reg_loss = 0.0
      for weight in trainable_variables:
        if 'kernel' in weight.name:
          if FLAGS.use_L4_reg:
            reg_loss += weight_decay * tf.reduce_sum(weight**4)
          else:
            reg_loss += weight_decay * tf.nn.l2_loss(weight)
      loss = loss + tf.nn.scale_regularization_loss(reg_loss)
      if FLAGS.use_EG_loss and len(prev_models) > 0:
        if len(tf.shape(images))==2:
          EG1 = expected_gradients_inter_intra_class_2d(images, images, model, FLAGS.num_ref_EG_loss, intra_class=True, labels_inputs=labels, labels_references=labels)
          EG2 = expected_gradients_inter_intra_class_2d(images, images, model, FLAGS.num_ref_EG_loss, inter_class=True, labels_inputs=labels, labels_references=labels)
        else:
          EG1 = expected_gradients_inter_intra_class(images, images, model, FLAGS.num_ref_EG_loss, intra_class=True, labels_inputs=labels, labels_references=labels)
          EG2 = expected_gradients_inter_intra_class(images, images, model, FLAGS.num_ref_EG_loss, inter_class=True, labels_inputs=labels, labels_references=labels)
        EG1 = tf.reshape(EG1, (tf.shape(EG1)[0], -1))
        EG2 = tf.reshape(EG2, (tf.shape(EG2)[0], -1))
        dot_prod = tf.linalg.matmul(EG1, tf.transpose(EG2))
        EG1_norm = tf.linalg.norm(EG1, axis=-1, keepdims=True)
        EG2_norm = tf.linalg.norm(EG2, axis=-1, keepdims=True)
        norm_prod = tf.linalg.matmul(EG1_norm, tf.transpose(EG2_norm))
        dot_prod = tf.linalg.diag_part(dot_prod)
        norm_prod = tf.linalg.diag_part(norm_prod)
        mask = norm_prod != 0.0
        dot_prod = dot_prod[mask]
        norm_prod = norm_prod[mask]
        norm_dot = dot_prod/norm_prod
        EG_loss = -1.0*tf.reduce_mean(norm_dot)
        loss = loss + FLAGS.EG_loss_weight * tf.cast(EG_loss, tf.float32)

    gradients = tape.gradient(loss, trainable_variables)
    if FLAGS.use_EG_loss and len(prev_models) > 0:
      train_EG_loss.update_state(EG_loss)
    elif FLAGS.monitor_EG_loss or FLAGS.use_EG_loss:
      if len(tf.shape(images))==2:
        EG1 = expected_gradients_inter_intra_class_2d(images, images, model, FLAGS.num_ref_EG_loss, intra_class=True, labels_inputs=labels, labels_references=labels)
        EG2 = expected_gradients_inter_intra_class_2d(images, images, model, FLAGS.num_ref_EG_loss, inter_class=True, labels_inputs=labels, labels_references=labels)
      else:
        EG1 = expected_gradients_inter_intra_class(images, images, model, FLAGS.num_ref_EG_loss, intra_class=True, labels_inputs=labels, labels_references=labels)
        EG2 = expected_gradients_inter_intra_class(images, images, model, FLAGS.num_ref_EG_loss, inter_class=True, labels_inputs=labels, labels_references=labels)
      EG1 = tf.reshape(EG1, (tf.shape(EG1)[0], -1))
      EG2 = tf.reshape(EG2, (tf.shape(EG2)[0], -1))
      dot_prod = tf.linalg.matmul(EG1, tf.transpose(EG2))
      EG1_norm = tf.linalg.norm(EG1, axis=-1, keepdims=True)
      EG2_norm = tf.linalg.norm(EG2, axis=-1, keepdims=True)
      norm_prod = tf.linalg.matmul(EG1_norm, tf.transpose(EG2_norm))
      dot_prod = tf.linalg.diag_part(dot_prod)
      norm_prod = tf.linalg.diag_part(norm_prod)
      mask = norm_prod != 0.0
      dot_prod = dot_prod[mask]
      norm_prod = norm_prod[mask]
      norm_dot = dot_prod/norm_prod
      EG_loss = -1.0*tf.reduce_mean(norm_dot)
      train_EG_loss.update_state(EG_loss)
    if FLAGS.monitor_rr_grad_norms:
      grad1 = tape.gradient(red_loss, trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
      if FLAGS.use_val_for_MI:
        grad2 = tape.gradient(red_loss, logits2, unconnected_gradients=tf.UnconnectedGradients.ZERO)
      else:
        grad2 = tape.gradient(red_loss, logits, unconnected_gradients=tf.UnconnectedGradients.ZERO)
      ctx = tf.distribute.get_replica_context()
      accum_grad1 = ctx.all_reduce(tf.distribute.ReduceOp.SUM, grad1)
      accum_grad2 = ctx.all_reduce(tf.distribute.ReduceOp.SUM, grad2)
      global_norm_1 = tf.math.reduce_sum([tf.nn.l2_loss(t) for t in accum_grad1])
      if num_heads > 1:
        global_norm_2 = tf.math.reduce_sum([tf.nn.l2_loss(t) for t in accum_grad2])
      else:
        global_norm_2 = tf.nn.l2_loss(accum_grad2)
    if clip is not None:
      ctx = tf.distribute.get_replica_context()
      accum_gradients = ctx.all_reduce(tf.distribute.ReduceOp.SUM, gradients)
      gradients, _ = tf.clip_by_global_norm(accum_gradients, FLAGS.clip_norm)
      optimizer.apply_gradients(
          zip(gradients, trainable_variables),
          experimental_aggregate_gradients=False)
    else:
      optimizer.apply_gradients(zip(gradients, trainable_variables))
    train_accuracy.update_state(labels, logits)
    if FLAGS.project_out_rank > 0 and not finetune:
      grads = tape.gradient(grad_asc_loss, global_proj)
      optimizer2.apply_gradients([[grads, global_proj]])
    if FLAGS.monitor_rr_grad_norms:
      return tf.stack([loss, ce_loss, red_loss, reg_loss, logit_loss, global_norm_1, global_norm_2], axis=0)
    else:
      return tf.stack([loss, ce_loss, red_loss, reg_loss, logit_loss], axis=0)

  def val_step(inputs):
    images = inputs['image']
    labels = inputs['label']

    reps, logits, feats = model(images, training=False, project_out_mat=project_out_mat)
    if FLAGS.use_HSIC_loss and FLAGS.use_HSIC_on_features:
      reps = feats
    if FLAGS.binary_classification:
      loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    else:
      loss_fn = keras.losses.SparseCategoricalCrossentropy(
          from_logits=True, reduction=keras.losses.Reduction.NONE)
    HSIC_ind_loss = 0.0

    ctx = tf.distribute.get_replica_context()
    if num_heads > 1:
      for ind, logit in enumerate(logits):
        loss = loss_fn(labels, logit)
        val_ce_loss.update_state(loss)
        val_accuracy.update_state(labels, logit)
        val_accuracy_heads[ind].update_state(labels, logit)
      if FLAGS.use_MI_loss or FLAGS.use_disagr_loss or FLAGS.use_HSIC_loss:
        red_loss = compute_multihead_MI_loss(labels, logits, FLAGS.class_specific_rr_loss, FLAGS.rr_weight)
      else:
        red_loss = compute_multihead_batch_redundancy_loss(labels, logits, FLAGS.class_specific_rr_loss, reps, FLAGS.rr_weight)
      val_rr_loss.update_state(red_loss * ctx.num_replicas_in_sync)
    elif FLAGS.use_seq_rr:
      loss = loss_fn(labels, logits)
      val_accuracy.update_state(labels, logits)
      val_ce_loss.update_state(loss)
      reps_arr = []
      logits_arr = []
      for curr_model in prev_models:
        curr_reps, curr_logits, curr_feats = curr_model(images, training=False)
        if FLAGS.use_HSIC_loss and FLAGS.use_HSIC_on_features:
          if FLAGS.use_prev_logits_HSIC_features:
            reps_arr.append(curr_logits)
          else:
            reps_arr.append(curr_feats)
        else:
          reps_arr.append(curr_reps)
        logits_arr.append(curr_logits)
      if FLAGS.use_exp_var_loss:
        red_loss = compute_seq_multihead_exp_variance_loss(labels, logits, FLAGS.class_specific_rr_loss, reps, reps_arr, FLAGS.rr_weight)
      elif FLAGS.use_MI_loss or FLAGS.use_disagr_loss or FLAGS.use_HSIC_loss:
        if FLAGS.use_HSIC_loss and FLAGS.use_HSIC_on_features:
          red_loss, HSIC_XY = compute_seq_HSIC_loss(labels, logits, reps, FLAGS.class_specific_rr_loss, reps_arr, FLAGS.rr_weight)
          HSIC_loss_ind, HSIC_XY_ind, HSIC_XX, HSIC_YY, th_hsic_loss_ind, th_hsic_xy_ind = est_seq_HSIC_loss_ind(labels, logits, reps, FLAGS.class_specific_rr_loss, reps_arr, FLAGS.rr_weight)
        else:
          red_loss, HSIC_XY = compute_seq_MI_loss(labels, logits, FLAGS.class_specific_rr_loss, logits_arr, FLAGS.rr_weight)
          if FLAGS.use_HSIC_loss:
            HSIC_loss_ind, HSIC_XY_ind, HSIC_XX, HSIC_YY, th_hsic_loss_ind, th_hsic_xy_ind = est_seq_HSIC_logits_loss_ind(labels, logits, FLAGS.class_specific_rr_loss, logits_arr, FLAGS.rr_weight)
        if FLAGS.use_HSIC_ratio:
          act_ratio = ctx.num_replicas_in_sync * red_loss / FLAGS.rr_weight
          act_ind_loss = ctx.num_replicas_in_sync * th_hsic_loss_ind
          val_HSIC_loss.update_state(act_ind_loss * act_ratio)
        else:
          val_HSIC_loss.update_state(ctx.num_replicas_in_sync * red_loss)
        val_HSIC_ind_loss.update_state(ctx.num_replicas_in_sync * HSIC_loss_ind)
        val_HSIC_th_ind_loss.update_state(ctx.num_replicas_in_sync * th_hsic_loss_ind)
        val_HSIC_XY.update_state(ctx.num_replicas_in_sync * HSIC_XY)
        val_HSIC_XY_ind.update_state(ctx.num_replicas_in_sync * HSIC_XY_ind)
        val_HSIC_XY_th_ind.update_state(ctx.num_replicas_in_sync * th_hsic_xy_ind)
        val_HSIC_XX.update_state(ctx.num_replicas_in_sync * HSIC_XX)
        val_HSIC_YY.update_state(ctx.num_replicas_in_sync * HSIC_YY)
      elif FLAGS.use_logit_decorr:
        red_loss = compute_seq_multihead_batch_redundancy_loss(labels, logits, FLAGS.class_specific_rr_loss, logits, logits_arr, FLAGS.rr_weight)
      elif FLAGS.use_prob_decorr:
        prob = tf.nn.softmax(logits, axis=1)
        prob_arr = []
        for logit in logits_arr:
          prob_arr.append(tf.nn.softmax(logit, axis=1))
        red_loss = compute_seq_multihead_batch_redundancy_loss(labels, logits, FLAGS.class_specific_rr_loss, prob, prob_arr, FLAGS.rr_weight)
      else:
        red_loss = compute_seq_multihead_batch_redundancy_loss(labels, logits, FLAGS.class_specific_rr_loss, reps, reps_arr, FLAGS.rr_weight)
      val_rr_loss.update_state(red_loss * ctx.num_replicas_in_sync)
      if FLAGS.use_num_sq_MI:
        red_loss_2 = compute_seq_MI_loss(labels, logits, FLAGS.class_specific_rr_loss, logits_arr, FLAGS.rr_weight, num_sq=False)
        val_no_sq_rr_loss.update_state(red_loss_2 * ctx.num_replicas_in_sync)
    else:
      loss = loss_fn(labels, logits)
      val_accuracy.update_state(labels, logits)
      val_ce_loss.update_state(loss)
      red_loss = compute_batch_redundancy_loss(labels, logits, FLAGS.class_specific_rr_loss, reps, FLAGS.rr_weight)
      val_rr_loss.update_state(red_loss * ctx.num_replicas_in_sync)
    if FLAGS.task_id == 'DRO':
      metadata = inputs['metadata']
      I1 = (metadata[:, 0] == 0) & (metadata[:, 1] == 0)
      val_00_accuracy.update_state(labels[I1], logits[I1])
      I2 = (metadata[:, 0] == 0) & (metadata[:, 1] == 1)
      val_01_accuracy.update_state(labels[I2], logits[I2])
      I3 = (metadata[:, 0] == 1) & (metadata[:, 1] == 0)
      val_10_accuracy.update_state(labels[I3], logits[I3])
      I4 = (metadata[:, 0] == 1) & (metadata[:, 1] == 1)
      val_11_accuracy.update_state(labels[I4], logits[I4])
    if FLAGS.monitor_EG_loss or FLAGS.use_EG_loss:
      if len(tf.shape(images))==2:
        EG1 = expected_gradients_inter_intra_class_2d(images, images, model, FLAGS.num_ref_EG_loss, intra_class=True, labels_inputs=labels, labels_references=labels)
        EG2 = expected_gradients_inter_intra_class_2d(images, images, model, FLAGS.num_ref_EG_loss, inter_class=True, labels_inputs=labels, labels_references=labels)
      else:
        EG1 = expected_gradients_inter_intra_class(images, images, model, FLAGS.num_ref_EG_loss, intra_class=True, labels_inputs=labels, labels_references=labels)
        EG2 = expected_gradients_inter_intra_class(images, images, model, FLAGS.num_ref_EG_loss, inter_class=True, labels_inputs=labels, labels_references=labels)
      EG1 = tf.reshape(EG1, (tf.shape(EG1)[0], -1))
      EG2 = tf.reshape(EG2, (tf.shape(EG2)[0], -1))
      dot_prod = tf.linalg.matmul(EG1, tf.transpose(EG2))
      EG1_norm = tf.linalg.norm(EG1, axis=-1, keepdims=True)
      EG2_norm = tf.linalg.norm(EG2, axis=-1, keepdims=True)
      norm_prod = tf.linalg.matmul(EG1_norm, tf.transpose(EG2_norm))
      dot_prod = tf.linalg.diag_part(dot_prod)
      norm_prod = tf.linalg.diag_part(norm_prod)
      mask = norm_prod != 0.0
      dot_prod = dot_prod[mask]
      norm_prod = norm_prod[mask]
      norm_dot = dot_prod/norm_prod
      EG_loss = -1.0*tf.reduce_mean(norm_dot)
      val_EG_loss.update_state(EG_loss)
    return loss

  def test_step(inputs):
    images = inputs['image']
    labels = inputs['label']

    reps, logits, feats = model(images, training=False, project_out_mat=project_out_mat)
    if FLAGS.use_HSIC_loss and FLAGS.use_HSIC_on_features:
      reps = feats
    if FLAGS.binary_classification:
      loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    else:
      loss_fn = keras.losses.SparseCategoricalCrossentropy(
          from_logits=True, reduction=keras.losses.Reduction.NONE)
    HSIC_ind_loss = 0.0
    loss = 0.0

    ctx = tf.distribute.get_replica_context()
    if num_heads > 1:
      for ind, logit in enumerate(logits):
        loss = loss_fn(labels, logit)
        test_ce_loss.update_state(loss)
        test_accuracy.update_state(labels, logit)
        test_accuracy_heads[ind].update_state(labels, logit)
      if FLAGS.use_MI_loss or FLAGS.use_disagr_loss or FLAGS.use_HSIC_loss:
        red_loss = compute_multihead_MI_loss(labels, logits, FLAGS.class_specific_rr_loss, FLAGS.rr_weight)
      else:
        red_loss = compute_multihead_batch_redundancy_loss(labels, logits, FLAGS.class_specific_rr_loss, reps, FLAGS.rr_weight)
      test_rr_loss.update_state(red_loss * ctx.num_replicas_in_sync)
    elif FLAGS.use_seq_rr:
      loss = loss_fn(labels, logits)
      test_accuracy.update_state(labels, logits)
      test_ce_loss.update_state(loss)
      reps_arr = []
      logits_arr = []
      for curr_model in prev_models:
        curr_reps, curr_logits, curr_feats = curr_model(images, training=False)
        if FLAGS.use_HSIC_loss and FLAGS.use_HSIC_on_features:
          if FLAGS.use_prev_logits_HSIC_features:
            reps_arr.append(curr_logits)
          else:
            reps_arr.append(curr_feats)
        else:
          reps_arr.append(curr_reps)
        logits_arr.append(curr_logits)
      if FLAGS.use_exp_var_loss:
        red_loss = compute_seq_multihead_exp_variance_loss(labels, logits, FLAGS.class_specific_rr_loss, reps, reps_arr, FLAGS.rr_weight)
      elif FLAGS.use_MI_loss or FLAGS.use_disagr_loss or FLAGS.use_HSIC_loss:
        if FLAGS.use_HSIC_loss and FLAGS.use_HSIC_on_features:
          red_loss, HSIC_XY = compute_seq_HSIC_loss(labels, logits, reps, FLAGS.class_specific_rr_loss, reps_arr, FLAGS.rr_weight)
          HSIC_loss_ind, HSIC_XY_ind, HSIC_XX, HSIC_YY, th_hsic_loss_ind, th_hsic_xy_ind = est_seq_HSIC_loss_ind(labels, logits, reps, FLAGS.class_specific_rr_loss, reps_arr, FLAGS.rr_weight)
        else:
          red_loss, HSIC_XY = compute_seq_MI_loss(labels, logits, FLAGS.class_specific_rr_loss, logits_arr, FLAGS.rr_weight)
          if FLAGS.use_HSIC_loss:
            HSIC_loss_ind, HSIC_XY_ind, HSIC_XX, HSIC_YY, th_hsic_loss_ind, th_hsic_xy_ind = est_seq_HSIC_logits_loss_ind(labels, logits, FLAGS.class_specific_rr_loss, logits_arr, FLAGS.rr_weight)
        if FLAGS.use_HSIC_ratio:
          act_ratio = ctx.num_replicas_in_sync * red_loss / FLAGS.rr_weight
          act_ind_loss = ctx.num_replicas_in_sync * th_hsic_loss_ind
          test_HSIC_loss.update_state(act_ind_loss * act_ratio)
        else:
          test_HSIC_loss.update_state(ctx.num_replicas_in_sync * red_loss)
        test_HSIC_ind_loss.update_state(ctx.num_replicas_in_sync * HSIC_loss_ind)
        test_HSIC_th_ind_loss.update_state(ctx.num_replicas_in_sync * th_hsic_loss_ind)
        test_HSIC_XY.update_state(ctx.num_replicas_in_sync * HSIC_XY)
        test_HSIC_XY_ind.update_state(ctx.num_replicas_in_sync * HSIC_XY_ind)
        test_HSIC_XY_th_ind.update_state(ctx.num_replicas_in_sync * th_hsic_xy_ind)
        test_HSIC_XX.update_state(ctx.num_replicas_in_sync * HSIC_XX)
        test_HSIC_YY.update_state(ctx.num_replicas_in_sync * HSIC_YY)
      elif FLAGS.use_logit_decorr:
        red_loss = compute_seq_multihead_batch_redundancy_loss(labels, logits, FLAGS.class_specific_rr_loss, logits, logits_arr, FLAGS.rr_weight)
      elif FLAGS.use_prob_decorr:
        prob = tf.nn.softmax(logits, axis=1)
        prob_arr = []
        for logit in logits_arr:
          prob_arr.append(tf.nn.softmax(logit, axis=1))
        red_loss = compute_seq_multihead_batch_redundancy_loss(labels, logits, FLAGS.class_specific_rr_loss, prob, prob_arr, FLAGS.rr_weight)
      else:
        red_loss = compute_seq_multihead_batch_redundancy_loss(labels, logits, FLAGS.class_specific_rr_loss, reps, reps_arr, FLAGS.rr_weight)
      test_rr_loss.update_state(red_loss * ctx.num_replicas_in_sync)
      if FLAGS.use_num_sq_MI:
        red_loss_2 = compute_seq_MI_loss(labels, logits, FLAGS.class_specific_rr_loss, logits_arr, FLAGS.rr_weight, num_sq=False)
        test_no_sq_rr_loss.update_state(red_loss_2 * ctx.num_replicas_in_sync)
    else:
      loss = loss_fn(labels, logits)
      test_accuracy.update_state(labels, logits)
      test_ce_loss.update_state(loss)
      red_loss = compute_batch_redundancy_loss(labels, logits, FLAGS.class_specific_rr_loss, reps, FLAGS.rr_weight)
      test_rr_loss.update_state(red_loss * ctx.num_replicas_in_sync)
    if FLAGS.task_id == 'DRO':
      metadata = inputs['metadata']
      I1 = tf.math.logical_and(metadata[:, 0] == 0, metadata[:, 1] == 0)
      test_00_accuracy.update_state(labels[I1], logits[I1])
      I2 = tf.math.logical_and(metadata[:, 0] == 0, metadata[:, 1] == 1)
      test_01_accuracy.update_state(labels[I2], logits[I2])
      I3 = tf.math.logical_and(metadata[:, 0] == 1, metadata[:, 1] == 0)
      test_10_accuracy.update_state(labels[I3], logits[I3])
      I4 = tf.math.logical_and(metadata[:, 0] == 1, metadata[:, 1] == 1)
      test_11_accuracy.update_state(labels[I4], logits[I4])
    if FLAGS.monitor_EG_loss or FLAGS.use_EG_loss:
      if len(tf.shape(images))==2:
        EG1 = expected_gradients_inter_intra_class_2d(images, images, model, FLAGS.num_ref_EG_loss, intra_class=True, labels_inputs=labels, labels_references=labels)
        EG2 = expected_gradients_inter_intra_class_2d(images, images, model, FLAGS.num_ref_EG_loss, inter_class=True, labels_inputs=labels, labels_references=labels)
      else:
        EG1 = expected_gradients_inter_intra_class(images, images, model, FLAGS.num_ref_EG_loss, intra_class=True, labels_inputs=labels, labels_references=labels)
        EG2 = expected_gradients_inter_intra_class(images, images, model, FLAGS.num_ref_EG_loss, inter_class=True, labels_inputs=labels, labels_references=labels)
      EG1 = tf.reshape(EG1, (tf.shape(EG1)[0], -1))
      EG2 = tf.reshape(EG2, (tf.shape(EG2)[0], -1))
      dot_prod = tf.linalg.matmul(EG1, tf.transpose(EG2))
      EG1_norm = tf.linalg.norm(EG1, axis=-1, keepdims=True)
      EG2_norm = tf.linalg.norm(EG2, axis=-1, keepdims=True)
      norm_prod = tf.linalg.matmul(EG1_norm, tf.transpose(EG2_norm))
      dot_prod = tf.linalg.diag_part(dot_prod)
      norm_prod = tf.linalg.diag_part(norm_prod)
      mask = norm_prod != 0.0
      dot_prod = dot_prod[mask]
      norm_prod = norm_prod[mask]
      norm_dot = dot_prod/norm_prod
      EG_loss = -1.0*tf.reduce_mean(norm_dot)
      test_EG_loss.update_state(EG_loss)
    return loss

  def extra_dset_step(inputs, i):
    images = inputs['image']
    labels = inputs['label']

    _, logits, _ = model(images, training=False)
    if FLAGS.binary_classification:
      loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    else:
      loss_fn = keras.losses.SparseCategoricalCrossentropy(
          from_logits=True, reduction=keras.losses.Reduction.NONE)

    if num_heads > 1:
      for ind, logit in enumerate(logits):
        loss = loss_fn(labels, logit)
        extra_dsets_accuracy[extra_dset_index].update_state(labels, logit)
        extra_dsets_accuracy_heads[ind][extra_dset_index].update_state(labels, logit)
    else:
      loss = loss_fn(labels, logits)
      extra_dsets_accuracy[extra_dset_index].update_state(labels, logits)
    return loss

  def EG_step(inputs, references, model, model2, i, j):
    if len(tf.shape(inputs['image']))==2:
      EG1 = expected_gradients_full_2d(inputs['image'], references['image'], model, k=200, labels=inputs['label'])
      EG2 = expected_gradients_full_2d(inputs['image'], references['image'], model2, k=200, labels=inputs['label'])
    else:
      EG1 = expected_gradients_full(inputs['image'], references['image'], model, k=200, labels=inputs['label'])
      EG2 = expected_gradients_full(inputs['image'], references['image'], model2, k=200, labels=inputs['label'])
    EG1 = tf.reshape(EG1, (tf.shape(EG1)[0], -1))
    EG2 = tf.reshape(EG2, (tf.shape(EG2)[0], -1))
    dot_prod = tf.linalg.matmul(EG1, tf.transpose(EG2))
    EG1_norm = tf.linalg.norm(EG1, axis=-1, keepdims=True)
    EG2_norm = tf.linalg.norm(EG2, axis=-1, keepdims=True)
    norm_prod = tf.linalg.matmul(EG1_norm, tf.transpose(EG2_norm))
    dot_prod = tf.linalg.diag_part(dot_prod)
    norm_prod = tf.linalg.diag_part(norm_prod)
    mask = norm_prod != 0.0
    dot_prod = dot_prod[mask]
    norm_prod = norm_prod[mask]
    norm_dot = dot_prod/norm_prod
    if i==0:
      train_EG_overlap[j].update_state(norm_dot)
    elif i==1:
      val_EG_overlap[j].update_state(norm_dot)
    else:
      test_EG_overlap[j].update_state(norm_dot)
    return norm_dot

  def noise_robust(inputs, i, gauss):
    for j in range(4):
      if j==0 or j==2:
        models = [model]
      else:
        models = []
        for prev_model in prev_models:
          models.append(prev_model)
        models.append(model)
      if gauss:
        if j < 2:
          robust = gauss_noise_robust(models, inputs)
        else:
          robust = gauss_noise_robust_2(models, inputs)
      else:
        robust = mask_noise_robust(models, inputs)

      if i==0:
        if gauss:
          train_gauss_robust[j].update_state(robust)
        else:
          train_mask_robust[j].update_state(robust)
      elif i==1:
        if gauss:
          val_gauss_robust[j].update_state(robust)
        else:
          val_mask_robust[j].update_state(robust)
      else:
        if gauss:
          test_gauss_robust[j].update_state(robust)
        else:
          test_mask_robust[j].update_state(robust)

    return robust

  def logit_corr(inputs, i):
    corr = tf.constant(0.0)
    for j in range(len(prev_models)):
      corr = logit_correlation(model, prev_models[j], inputs)
      if i==0:
        train_logit_correlation[j].update_state(corr)
      elif i==1:
        val_logit_correlation[j].update_state(corr)
      else:
        test_logit_correlation[j].update_state(corr)

    return corr

  def error_diversity_step(data, subg):
    return error_diversity(model, prev_models[err_div_prev_model_ind], data, subg)

  @tf.function
  def distributed_train_step(dataset_inputs, dataset_inputs_2, use_rr_loss):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs, dataset_inputs_2, use_rr_loss))
    return strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

  @tf.function
  def distributed_val_step(dataset_inputs):
    return strategy.run(val_step, args=(dataset_inputs,))

  @tf.function
  def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))

  @tf.function
  def distributed_extra_dset_step(dataset_inputs, i):
    return strategy.run(extra_dset_step, args=(dataset_inputs, i))

  @tf.function
  def distributed_EG_step(inputs, references, model, model2, i, j):
    return strategy.run(EG_step, args=(inputs, references, model, model2, i, j))

  @tf.function
  def distributed_noise_step(inputs, i, j):
    return strategy.run(noise_robust, args=(inputs, i, j))

  @tf.function
  def distributed_logit_corr(inputs, i):
    return strategy.run(logit_corr, args=(inputs, i))

  @tf.function
  def distributed_error_diversity(data, subg):
    per_replica_errs = strategy.run(error_diversity_step, args=(data,subg))
    return strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_errs, axis=None)

  with strategy.scope():
    # Instantiate metrics
    train_HSIC_loss = keras.metrics.Mean(name='{}train_HSIC_loss'.format(prefix))
    train_HSIC_ind_loss = keras.metrics.Mean(name='{}train_HSIC_ind_loss'.format(prefix))
    train_HSIC_XY = keras.metrics.Mean(name='{}train_HSIC_XY'.format(prefix))
    train_HSIC_XX = keras.metrics.Mean(name='{}train_HSIC_XX'.format(prefix))
    train_HSIC_YY = keras.metrics.Mean(name='{}train_HSIC_YY'.format(prefix))
    train_HSIC_XY_ind = keras.metrics.Mean(name='{}train_HSIC_XY_ind'.format(prefix))
    train_HSIC_XY_th_ind = keras.metrics.Mean(name='{}train_HSIC_XY_th_ind'.format(prefix))
    train_HSIC_th_ind_loss = keras.metrics.Mean(name='{}train_HSIC_th_ind_loss'.format(prefix))
    train_EG_loss = keras.metrics.Mean(name='{}train_EG_loss'.format(prefix))
    val_ce_loss = keras.metrics.Mean(name='{}val_ce_loss'.format(prefix))
    val_rr_loss = keras.metrics.Mean(name='{}val_rr_loss'.format(prefix))
    val_HSIC_loss = keras.metrics.Mean(name='{}val_HSIC_loss'.format(prefix))
    val_HSIC_ind_loss = keras.metrics.Mean(name='{}val_HSIC_ind_loss'.format(prefix))
    val_HSIC_XY = keras.metrics.Mean(name='{}val_HSIC_XY'.format(prefix))
    val_HSIC_XX = keras.metrics.Mean(name='{}val_HSIC_XX'.format(prefix))
    val_HSIC_YY = keras.metrics.Mean(name='{}val_HSIC_YY'.format(prefix))
    val_HSIC_XY_ind = keras.metrics.Mean(name='{}val_HSIC_XY_ind'.format(prefix))
    val_HSIC_XY_th_ind = keras.metrics.Mean(name='{}val_HSIC_XY_th_ind'.format(prefix))
    val_HSIC_th_ind_loss = keras.metrics.Mean(name='{}val_HSIC_th_ind_loss'.format(prefix))
    val_EG_loss = keras.metrics.Mean(name='{}val_EG_loss'.format(prefix))
    test_ce_loss = keras.metrics.Mean(name='{}test_ce_loss'.format(prefix))
    test_rr_loss = keras.metrics.Mean(name='{}test_rr_loss'.format(prefix))
    test_HSIC_loss = keras.metrics.Mean(name='{}test_HSIC_loss'.format(prefix))
    test_HSIC_ind_loss = keras.metrics.Mean(name='{}test_HSIC_ind_loss'.format(prefix))
    test_HSIC_XY = keras.metrics.Mean(name='{}test_HSIC_XY'.format(prefix))
    test_HSIC_XX = keras.metrics.Mean(name='{}test_HSIC_XX'.format(prefix))
    test_HSIC_YY = keras.metrics.Mean(name='{}test_HSIC_YY'.format(prefix))
    test_HSIC_XY_ind = keras.metrics.Mean(name='{}test_HSIC_XY_ind'.format(prefix))
    test_HSIC_XY_th_ind = keras.metrics.Mean(name='{}test_HSIC_XY_th_ind'.format(prefix))
    test_HSIC_th_ind_loss = keras.metrics.Mean(name='{}test_HSIC_th_ind_loss'.format(prefix))
    test_EG_loss = keras.metrics.Mean(name='{}test_EG_loss'.format(prefix))
    if FLAGS.monitor_EG_overlap:
      train_EG_overlap = []
      val_EG_overlap = []
      test_EG_overlap = []
      for i in range(len(prev_models)):
        train_EG_overlap.append(keras.metrics.Mean(name='{}train_EG_overlap_{}'.format(prefix,i)))
        val_EG_overlap.append(keras.metrics.Mean(name='{}val_EG_overlap_{}'.format(prefix,i)))
        test_EG_overlap.append(keras.metrics.Mean(name='{}test_EG_overlap_{}'.format(prefix,i)))
    if FLAGS.monitor_logit_correlation:
      train_logit_correlation = []
      val_logit_correlation = []
      test_logit_correlation = []
      for i in range(len(prev_models)):
        train_logit_correlation.append(keras.metrics.Mean(name='{}train_logit_correlation_{}'.format(prefix,i)))
        val_logit_correlation.append(keras.metrics.Mean(name='{}val_logit_correlation_{}'.format(prefix,i)))
        test_logit_correlation.append(keras.metrics.Mean(name='{}test_logit_correlation_{}'.format(prefix,i)))
    if FLAGS.monitor_robustness_measures:
      # array of size 4, first represents the current model, second represents the ensemble
      # The last 2 for Gaussian robustness of a different kind
      train_gauss_robust = []
      val_gauss_robust = []
      test_gauss_robust = []
      train_mask_robust = []
      val_mask_robust = []
      test_mask_robust = []
      for i in range(4):
        train_gauss_robust.append(keras.metrics.Mean(name='{}train_gauss_robust_{}'.format(prefix, i)))
        val_gauss_robust.append(keras.metrics.Mean(name='{}val_gauss_robust_{}'.format(prefix, i)))
        test_gauss_robust.append(keras.metrics.Mean(name='{}test_gauss_robust_{}'.format(prefix, i)))
        train_mask_robust.append(keras.metrics.Mean(name='{}train_mask_robust_{}'.format(prefix, i)))
        val_mask_robust.append(keras.metrics.Mean(name='{}val_mask_robust_{}'.format(prefix, i)))
        test_mask_robust.append(keras.metrics.Mean(name='{}test_mask_robust_{}'.format(prefix, i)))
    if FLAGS.use_num_sq_MI:
      val_no_sq_rr_loss = keras.metrics.Mean(name='{}val_no_sq_rr_loss'.format(prefix))
      test_no_sq_rr_loss = keras.metrics.Mean(name='{}test_no_sq_rr_loss'.format(prefix))
    if FLAGS.binary_classification:
      train_accuracy = keras.metrics.BinaryAccuracy(
        name='{}train_accuracy'.format(prefix), threshold=0.0)
      val_accuracy = keras.metrics.BinaryAccuracy(
        name='{}val_accuracy'.format(prefix), threshold=0.0)
      test_accuracy = keras.metrics.BinaryAccuracy(
        name='{}test_accuracy'.format(prefix), threshold=0.0)
      extra_dsets_accuracy = []
      for i in range(len(extra_dsets)):
        extra_dsets_accuracy.append(
            keras.metrics.BinaryAccuracy(
                name='{}extra_dset_accuracy_{}'.format(prefix, i), threshold=0.0))
    else:
      train_accuracy = keras.metrics.SparseCategoricalAccuracy(
          name='{}train_accuracy'.format(prefix))
      val_accuracy = keras.metrics.SparseCategoricalAccuracy(
          name='{}val_accuracy'.format(prefix))
      test_accuracy = keras.metrics.SparseCategoricalAccuracy(
          name='{}test_accuracy'.format(prefix))
      extra_dsets_accuracy = []
      for i in range(len(extra_dsets)):
        extra_dsets_accuracy.append(
            keras.metrics.SparseCategoricalAccuracy(
                name='{}extra_dset_accuracy_{}'.format(prefix, i)))
    if num_heads > 1:
      val_accuracy_heads = []
      test_accuracy_heads = []
      extra_dsets_accuracy_heads = []
      for i in range(num_heads):
        val_accuracy_heads.append(keras.metrics.SparseCategoricalAccuracy(
                name='{}val_accuracy_head_{}'.format(prefix, i)))
        test_accuracy_heads.append(keras.metrics.SparseCategoricalAccuracy(
                name='{}test_accuracy_head_{}'.format(prefix, i)))
        extra_dsets_accuracy_curr_head = []
        for j in range(len(extra_dsets)):
          extra_dsets_accuracy_curr_head.append(
              keras.metrics.SparseCategoricalAccuracy(
                  name='{}extra_dset_accuracy_{}_head_{}'.format(prefix, j, i)))
        extra_dsets_accuracy_heads.append(extra_dsets_accuracy_curr_head)
    if FLAGS.task_id == 'DRO':
      if FLAGS.binary_classification:
        test_00_accuracy = keras.metrics.BinaryAccuracy(
            name='{}test_00_accuracy'.format(prefix), threshold=0.0)
        test_10_accuracy = keras.metrics.BinaryAccuracy(
            name='{}test_10_accuracy'.format(prefix), threshold=0.0)
        test_01_accuracy = keras.metrics.BinaryAccuracy(
            name='{}test_01_accuracy'.format(prefix), threshold=0.0)
        test_11_accuracy = keras.metrics.BinaryAccuracy(
            name='{}test_11_accuracy'.format(prefix), threshold=0.0)
        val_00_accuracy = keras.metrics.BinaryAccuracy(
            name='{}val_00_accuracy'.format(prefix), threshold=0.0)
        val_10_accuracy = keras.metrics.BinaryAccuracy(
            name='{}val_10_accuracy'.format(prefix), threshold=0.0)
        val_01_accuracy = keras.metrics.BinaryAccuracy(
            name='{}val_01_accuracy'.format(prefix), threshold=0.0)
        val_11_accuracy = keras.metrics.BinaryAccuracy(
            name='{}val_11_accuracy'.format(prefix), threshold=0.0)
      else:
        test_00_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name='{}test_00_accuracy'.format(prefix))
        test_10_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name='{}test_10_accuracy'.format(prefix))
        test_01_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name='{}test_01_accuracy'.format(prefix))
        test_11_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name='{}test_11_accuracy'.format(prefix))
        val_00_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name='{}val_00_accuracy'.format(prefix))
        val_10_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name='{}val_10_accuracy'.format(prefix))
        val_01_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name='{}val_01_accuracy'.format(prefix))
        val_11_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name='{}val_11_accuracy'.format(prefix))
    if FLAGS.project_out_rank > 0 and not finetune and W is not None:
      proj_dot_prod = []
      for i in range(tf.shape(W)[0]):
        proj_dot_prod.append(keras.metrics.Mean(name='{}proj_dot_prod_{}'.format(prefix, i)))

  best_val_acc = 0.0
  best_test_acc = 0.0
  best_extra_dset_acc = []
  for i in range(len(extra_dsets)):
    best_extra_dset_acc.append(0.0)
  if FLAGS.monitor_robustness_measures:
    best_train_gauss_robust = []
    best_val_gauss_robust = []
    best_test_gauss_robust = []
    best_train_mask_robust = []
    best_val_mask_robust = []
    best_test_mask_robust = []
    for i in range(4):
      best_train_gauss_robust.append(0.0)
      best_val_gauss_robust.append(0.0)
      best_test_gauss_robust.append(0.0)
      best_train_mask_robust.append(0.0)
      best_val_mask_robust.append(0.0)
      best_test_mask_robust.append(0.0)
  if FLAGS.monitor_EG_overlap:
    best_train_EG_overlap = []
    best_val_EG_overlap = []
    best_test_EG_overlap = []
    for i in range(len(prev_models)):
      best_train_EG_overlap.append(0.0)
      best_val_EG_overlap.append(0.0)
      best_test_EG_overlap.append(0.0)
  if FLAGS.monitor_logit_correlation:
    best_train_logit_correlation = []
    best_val_logit_correlation = []
    best_test_logit_correlation = []
    for i in range(len(prev_models)):
      best_train_logit_correlation.append(0.0)
      best_val_logit_correlation.append(0.0)
      best_test_logit_correlation.append(0.0)
  if FLAGS.monitor_error_diversity:
    best_val_error_diversity = []
    best_test_error_diversity = []
    for i in range(len(prev_models)):
      best_val_error_diversity.append(0.0)
      best_test_error_diversity.append(0.0)
    if FLAGS.task_id == 'DRO':
      best_val_error_diversity_subgroup = []
      best_test_error_diversity_subgroup = []
      for i in range(4):
        subgroup_error_val_div = []
        subgroup_error_test_div = []
        for j in range(len(prev_models)):
          subgroup_error_val_div.append(0.0)
          subgroup_error_test_div.append(0.0)
        best_val_error_diversity_subgroup.append(subgroup_error_val_div)
        best_test_error_diversity_subgroup.append(subgroup_error_test_div)
  if num_heads > 1:
    best_val_acc_best_head = 0.0
    best_test_acc_best_head = 0.0
    best_extra_dset_acc_best_head = []
    for i in range(len(extra_dsets)):
      best_extra_dset_acc_best_head.append(0.0)
    best_val_acc_heads = []
    best_test_acc_heads = []
    best_extra_dset_acc_heads = []
    for i in range(num_heads):
      best_val_acc_heads.append(0.0)
      best_test_acc_heads.append(0.0)
      best_extra_dset_acc_curr_head = []
      for j in range(len(extra_dsets)):
        best_extra_dset_acc_curr_head.append(0.0)
      best_extra_dset_acc_heads.append(best_extra_dset_acc_curr_head)
  per_epoch_steps = int(train_len / FLAGS.train_batch_size) + 1
  if finetune:
    if FLAGS.finetune_val_steps_gap > 0:
      per_val_run_steps = FLAGS.finetune_val_steps_gap
    else:
      per_val_run_steps = FLAGS.finetune_val_epochs_gap * per_epoch_steps
  else:
    if FLAGS.val_steps_gap > 0:
      per_val_run_steps = FLAGS.val_steps_gap
    else:
      per_val_run_steps = FLAGS.val_epochs_gap * per_epoch_steps
  iterator = iter(ds_train)
  if FLAGS.use_val_for_MI:
    iterator2 = iter(MI_ds)
  if finetune:
    if FLAGS.train_steps_finetune > 0:
      total_steps = FLAGS.train_steps_finetune
    else:
      total_steps = per_epoch_steps * FLAGS.train_epochs_finetune
  else:
    if FLAGS.train_steps > 0:
      total_steps = FLAGS.train_steps
    else:
      total_steps = per_epoch_steps * FLAGS.train_epochs

  #initially checking val loss and test loss
  for data in ds_test:
    #tf.print(data)
    distributed_test_step(data)

  for data in ds_val:
    distributed_val_step(data)

  test_ce_loss.reset_states()
  test_rr_loss.reset_states()
  test_accuracy.reset_states()
  val_ce_loss.reset_states()
  val_rr_loss.reset_states()
  val_accuracy.reset_states()
  if num_heads > 1:
    for i in range(num_heads):
      val_accuracy_heads[i].reset_states()
      test_accuracy_heads[i].reset_states()
  if FLAGS.task_id == 'DRO':
    test_00_accuracy.reset_states()
    test_01_accuracy.reset_states()
    test_10_accuracy.reset_states()
    test_11_accuracy.reset_states()
    val_00_accuracy.reset_states()
    val_01_accuracy.reset_states()
    val_10_accuracy.reset_states()
    val_11_accuracy.reset_states()

  temp_use_rr_loss = use_rr_loss
  for step in range(total_steps):
    if step % per_epoch_steps == 0:
      epoch = int(step/per_epoch_steps)
      logging.info('Starting epoch %d', int(step / per_epoch_steps) + 1)
      total_loss = 0.0
      total_ce_loss = 0.0
      total_rr_loss = 0.0
      total_reg_loss = 0.0
      total_logit_loss = 0.0
      train_HSIC_loss.reset_states()
      train_HSIC_th_ind_loss.reset_states()
      train_HSIC_XY_th_ind.reset_states()
      train_HSIC_XY_ind.reset_states()
      train_HSIC_YY.reset_states()
      train_HSIC_XX.reset_states()
      train_HSIC_XY.reset_states()
      train_HSIC_ind_loss.reset_states()
      if FLAGS.monitor_rr_grad_norms:
        total_grad_norm_vars = 0.0
        total_grad_norm_logits = 0.0
      num_batches = 0
      if ckpt_manager is not None and FLAGS.save_model and epoch % FLAGS.checkpoint_epochs == 0:
        ckpt_manager.checkpoint.epoch.assign(epoch)
        ckpt_manager.save()
      if FLAGS.finetune_only_head and not only_linear_head:
        curr_var_ind = 0
        for ind, var in enumerate(trainable_variables):
          if curr_var_ind >= FLAGS.num_head_layers:
            break
          if 'kernel' in var.name:
            uncentered_outer_prod = tf.linalg.matmul(var, tf.transpose(var))
            centered_var = var - tf.reduce_mean(var, axis=1, keepdims=True)
            centered_outer_prod = tf.linalg.matmul(centered_var, tf.transpose(centered_var))
            e1, v1 = tf.linalg.eigh(uncentered_outer_prod)
            e2, v2 = tf.linalg.eigh(centered_outer_prod)
            e1_norm = e1/tf.reduce_sum(e1)
            e1_norm_clip = tf.maximum(e1_norm, 1e-9)
            SVD_eff_rank = tf.math.exp(-1.0*tf.reduce_sum(e1_norm_clip*tf.math.log(e1_norm_clip)))
            e2_norm = e2/tf.reduce_sum(e2)
            e2_norm_clip = tf.maximum(e2_norm, 1e-9)
            PCA_eff_rank = tf.math.exp(-1.0*tf.reduce_sum(e2_norm_clip*tf.math.log(e2_norm_clip)))
            for i in range(len(prev_models)):
              var_prev_model = prev_models_trainable_variables[i][ind]
              uncentered_outer_prod = tf.linalg.matmul(var_prev_model, tf.transpose(var_prev_model))
              centered_var_prev_model = var_prev_model - tf.reduce_mean(var_prev_model, axis=1, keepdims=True)
              centered_outer_prod = tf.linalg.matmul(centered_var_prev_model, tf.transpose(centered_var_prev_model))
              e1_prev, v1_prev = tf.linalg.eigh(uncentered_outer_prod)
              e2_prev, v2_prev = tf.linalg.eigh(centered_outer_prod)
              for j in range(FLAGS.check_ranks_max):
                v1_reshape = tf.reshape(v1[:,-(j+1):], (tf.shape(v1)[0], -1))
                v2_reshape = tf.reshape(v2[:,-(j+1):], (tf.shape(v2)[0], -1))
                v1_prev_reshape = tf.reshape(v1_prev[:,-(j+1):], (tf.shape(v1_prev)[0], -1))
                v2_prev_reshape = tf.reshape(v2_prev[:,-(j+1):], (tf.shape(v2_prev)[0], -1))
                val1 = tf.linalg.matmul(tf.transpose(v1_reshape), v1_prev_reshape)
                val2 = tf.linalg.matmul(tf.transpose(v2_reshape), v2_prev_reshape)
            curr_var_ind += 1

    use_rr_loss = temp_use_rr_loss and (step > FLAGS.use_rr_after_frac * total_steps)

    def f(val):
      def rr_loss_dist(value_ctx):
        return tf.constant(val, dtype=tf.bool)
      return rr_loss_dist

    def rr_loss_dist(value_ctx):
      return tf.constant(use_rr_loss, dtype=tf.bool)

    def MI_ds_dist(value_ctx):
      return tf.constant(0.0, dtype=tf.float32)

    if FLAGS.check_tf_func:
      distributed_rr_vals = strategy.experimental_distribute_values_from_function(rr_loss_dist)
    else:
      distributed_rr_vals = strategy.experimental_distribute_values_from_function(f(use_rr_loss))
    distributed_MI_ds_vals = strategy.experimental_distribute_values_from_function(MI_ds_dist)

    if FLAGS.use_val_for_MI:
      if FLAGS.monitor_rr_grad_norms:
        curr_loss, curr_ce_loss, curr_rr_loss, curr_reg_loss, curr_logit_loss, curr_grad_norm_var, curr_grad_norm_logit = distributed_train_step(next(iterator), next(iterator2), distributed_rr_vals)
      else:
        curr_loss, curr_ce_loss, curr_rr_loss, curr_reg_loss, curr_logit_loss = distributed_train_step(next(iterator), next(iterator2), distributed_rr_vals)
    else:
      if FLAGS.monitor_rr_grad_norms:
        curr_loss, curr_ce_loss, curr_rr_loss, curr_reg_loss, curr_logit_loss, curr_grad_norm_var, curr_grad_norm_logit = distributed_train_step(next(iterator), distributed_MI_ds_vals, distributed_rr_vals)
      else:
        curr_loss, curr_ce_loss, curr_rr_loss, curr_reg_loss, curr_logit_loss = distributed_train_step(next(iterator), distributed_MI_ds_vals, distributed_rr_vals)
    total_loss += curr_loss
    total_ce_loss += curr_ce_loss
    total_rr_loss += curr_rr_loss
    total_reg_loss += curr_reg_loss
    total_logit_loss += curr_logit_loss
    if FLAGS.monitor_rr_grad_norms:
      total_grad_norm_vars += curr_grad_norm_var
      total_grad_norm_logits += curr_grad_norm_logit
    num_batches += 1
    if FLAGS.project_out_rank > 0 and not finetune and W is not None:
      for i in range(tf.shape(W)[0]):
        proj_dot_prod[i].reset_states()
    if FLAGS.monitor_EG_loss or FLAGS.use_EG_loss:
      train_EG_loss.reset_states()
      val_EG_loss.reset_states()
      test_EG_loss.reset_states()
    if ckpt_manager is not None and FLAGS.save_model and step == total_steps - 1:
      ckpt_manager.checkpoint.epoch.assign(epoch)
      ckpt_manager.save()

    if (step + 1) % per_epoch_steps == 0 or step == total_steps - 1:
      epoch = int(step / per_epoch_steps)
      train_loss = total_loss / num_batches
      train_ce_loss = total_ce_loss / num_batches
      train_rr_loss = total_rr_loss / num_batches
      train_reg_loss = total_reg_loss / num_batches
      train_logit_loss = total_logit_loss / num_batches
      if FLAGS.monitor_rr_grad_norms:
        train_grad_norm_vars = total_grad_norm_vars / num_batches
        train_grad_norm_logits = total_grad_norm_logits / num_batches

      if (step + 1) % per_val_run_steps == 0 or step == total_steps - 1:
        # TEST LOOP
        for data in ds_test:
          distributed_test_step(data)

        for data in ds_val:
          distributed_val_step(data)

        for i, dset in enumerate(extra_dsets):
          extra_dset_index = i
          for data in dset:
            distributed_extra_dset_step(data, i)

        if FLAGS.monitor_EG_overlap:
          for ind, prev_model in enumerate(prev_models):
            distributed_EG_step(next(iterator), next(iterator), model, prev_model, 0, ind)
            for ind1, data in enumerate(ds_val):
              if ind1 == 0:
                temp1 = data
              if ind1 == 1:
                temp2 = data
                break
            if ind1==0:
              distributed_EG_step(temp1, temp1, model, prev_model, 1, ind)
            else:
              distributed_EG_step(temp1, temp2, model, prev_model, 1, ind)
            for ind1, data in enumerate(ds_test):
              if ind1 == 0:
                temp1 = data
              if ind1 == 1:
                temp2 = data
                break
            if ind1==0:
              distributed_EG_step(temp1, temp1, model, prev_model, 2, ind)
            else:
              distributed_EG_step(temp1, temp2, model, prev_model, 2, ind)

        if FLAGS.monitor_robustness_measures:
          distributed_noise_step(next(iterator), 0, True)
          #distributed_noise_step(next(iterator), 0, False)
          for ind1, data in enumerate(ds_val):
            if ind1 == 0:
              temp1 = data
              break
          distributed_noise_step(temp1, 1, True)
          #distributed_noise_step(temp1, 1, False)
          for ind1, data in enumerate(ds_test):
            if ind1 == 0:
              temp1 = data
              break
          distributed_noise_step(temp1, 2, True)
          #distributed_noise_step(temp1, 2, False)
          for i in range(4):
            train_gauss_robust[i].reset_states()
            test_gauss_robust[i].reset_states()
            val_gauss_robust[i].reset_states()

        if (FLAGS.use_early_stopping or step == total_steps - 1) and val_accuracy.result() > best_val_acc:
          best_val_acc = val_accuracy.result()
          best_test_acc = test_accuracy.result()
          for i in range(len(extra_dsets)):
            best_extra_dset_acc[i] = extra_dsets_accuracy[i].result()
          if FLAGS.monitor_EG_overlap:
            for i in range(len(prev_models)):
              best_train_EG_overlap[i] = train_EG_overlap[i].result()
              best_val_EG_overlap[i] = val_EG_overlap[i].result()
              best_test_EG_overlap[i] = test_EG_overlap[i].result()
          if FLAGS.monitor_robustness_measures:
            distributed_noise_step(next(iterator), 0, True)
            #distributed_noise_step(next(iterator), 0, False)
            for ind1, data in enumerate(ds_val):
              if ind1 == 0:
                temp1 = data
                break
            distributed_noise_step(temp1, 1, True)
            #distributed_noise_step(temp1, 1, False)
            for ind1, data in enumerate(ds_test):
              if ind1 == 0:
                temp1 = data
                break
            distributed_noise_step(temp1, 2, True)
            #distributed_noise_step(temp1, 2, False)
            for i in range(4):
              best_train_gauss_robust[i] = train_gauss_robust[i].result()
              best_val_gauss_robust[i] = val_gauss_robust[i].result()
              best_test_gauss_robust[i] = test_gauss_robust[i].result()
              best_train_mask_robust[i] = train_mask_robust[i].result()
              best_val_mask_robust[i] = val_mask_robust[i].result()
              best_test_mask_robust[i] = test_mask_robust[i].result()


          if FLAGS.monitor_logit_correlation:
            distributed_logit_corr(next(iterator), 0)
            for ind1, data in enumerate(ds_val):
              if ind1==0:
                temp1 = data
                break
            distributed_logit_corr(temp1, 1)
            for ind1, data in enumerate(ds_test):
              if ind1==0:
                temp1 = data
                break
            distributed_logit_corr(temp1, 2)
            for ind, prev_model in enumerate(prev_models):
              best_train_logit_correlation[ind] = train_logit_correlation[ind].result()
              best_val_logit_correlation[ind] = val_logit_correlation[ind].result()
              best_test_logit_correlation[ind] = test_logit_correlation[ind].result()
          if FLAGS.monitor_error_diversity:
            if FLAGS.task_id == 'DRO':
              subgroup_runs = 5
            else:
              subgroup_runs = 1
            for subg in range(subgroup_runs):
              for ind, prev_model in enumerate(prev_models):
                err_div_prev_model_ind = ind
                err1 = 0.0
                err2 = 0.0
                comm_err = 0.0
                for data in ds_val:
                  e1, e2, e3 = distributed_error_diversity(data, subg)
                  err1 += e1
                  err2 += e2
                  comm_err += e3
                min_err = tf.math.minimum(err1, err2)
                if min_err > 0:
                  if subg==0:
                    best_val_error_diversity[ind] = comm_err/min_err
                  else:
                    best_val_error_diversity_subgroup[subg-1][ind] = comm_err/min_err
                else:
                  if subg == 0:
                    best_val_error_diversity[ind] = 1.0
                  else:
                    best_val_error_diversity_subgroup[subg-1][ind] = 1.0
                err1 = 0.0
                err2 = 0.0
                comm_err = 0.0
                for data in ds_test:
                  e1, e2, e3 = distributed_error_diversity(data, subg)
                  err1 += e1
                  err2 += e2
                  comm_err += e3
                min_err = tf.math.minimum(err1, err2)
                if min_err > 0:
                  if subg==0:
                    best_test_error_diversity[ind] = comm_err/min_err
                  else:
                    best_test_error_diversity_subgroup[subg-1][ind] = comm_err/min_err
                else:
                  if subg==0:
                    best_test_error_diversity[ind] = 1.0
                  else:
                    best_test_error_diversity_subgroup[subg-1][ind] = 1.0

          if num_heads > 1:
            temp_best_val_acc_best_head = 0.0
            temp_best_test_acc_best_head = 0.0
            temp_best_extra_dset_acc_best_head = []
            for i in range(len(extra_dsets)):
              temp_best_extra_dset_acc_best_head.append(0.0)
            for i in range(num_heads):
              best_val_acc_heads[i] = val_accuracy_heads[i].result()
              best_test_acc_heads[i] = test_accuracy_heads[i].result()
              if best_val_acc_heads[i] > temp_best_val_acc_best_head:
                temp_best_val_acc_best_head = best_val_acc_heads[i]
              if best_test_acc_heads[i] > temp_best_test_acc_best_head:
                temp_best_test_acc_best_head = best_test_acc_heads[i]
              for j in range(len(extra_dsets)):
                best_extra_dset_acc_heads[i][j] = extra_dsets_accuracy_heads[i][j].result()
                if best_extra_dset_acc_heads[i][j] > temp_best_extra_dset_acc_best_head[j]:
                  temp_best_extra_dset_acc_best_head[j] = best_extra_dset_acc_heads[i][j]
            best_val_acc_best_head = temp_best_val_acc_best_head
            best_test_acc_best_head = temp_best_test_acc_best_head
            for i in range(len(extra_dsets)):
              best_extra_dset_acc_best_head[i] = temp_best_extra_dset_acc_best_head[i]

          if FLAGS.task_id == 'DRO':
            best_test_00_acc = test_00_accuracy.result()
            best_test_01_acc = test_01_accuracy.result()
            best_test_10_acc = test_10_accuracy.result()
            best_test_11_acc = test_11_accuracy.result()
            best_test_adj_acc = (
                best_test_00_acc * 3498 + best_test_01_acc * 184 +
                best_test_10_acc * 56 + best_test_11_acc * 1057) / 4795
            best_val_00_acc = val_00_accuracy.result()
            best_val_01_acc = val_01_accuracy.result()
            best_val_10_acc = val_10_accuracy.result()
            best_val_11_acc = val_11_accuracy.result()
        logging.info(
            '{{\'Epoch\': %d, \'Train_loss\': %f, \'Val_loss\': %f, \'Train_acc\': %f, \'Val_acc\': %f}}',
            epoch, train_loss, val_ce_loss.result(), train_accuracy.result(),
            val_accuracy.result())
        if FLAGS.use_complete_corr:
            test_adj_acc = (
                test_00_accuracy.result() * 3498 + test_11_accuracy.result() * 1057) / 4555
          else:
            test_adj_acc = (
                test_00_accuracy.result() * 3498 + test_01_accuracy.result() * 184 +
                test_10_accuracy.result() * 56 + test_11_accuracy.result() * 1057) / 4795


        test_ce_loss.reset_states()
        test_rr_loss.reset_states()
        test_HSIC_loss.reset_states()
        test_HSIC_th_ind_loss.reset_states()
        test_HSIC_XY_th_ind.reset_states()
        test_HSIC_XY_ind.reset_states()
        test_HSIC_YY.reset_states()
        test_HSIC_XX.reset_states()
        test_HSIC_XY.reset_states()
        test_HSIC_ind_loss.reset_states()
        test_accuracy.reset_states()
        test_accuracy.reset_states()
        val_ce_loss.reset_states()
        val_rr_loss.reset_states()
        val_HSIC_loss.reset_states()
        val_HSIC_th_ind_loss.reset_states()
        val_HSIC_XY_th_ind.reset_states()
        val_HSIC_XY_ind.reset_states()
        val_HSIC_YY.reset_states()
        val_HSIC_XX.reset_states()
        val_HSIC_XY.reset_states()
        val_HSIC_ind_loss.reset_states()
        val_accuracy.reset_states()
        for i in range(len(extra_dsets)):
          extra_dsets_accuracy[i].reset_states()
        if num_heads > 1:
          for i in range(num_heads):
            val_accuracy_heads[i].reset_states()
            test_accuracy_heads[i].reset_states()
            for j in range(len(extra_dsets)):
              extra_dsets_accuracy_heads[i][j].reset_states()
        if FLAGS.monitor_EG_overlap:
          for i in range(len(prev_models)):
            train_EG_overlap[i].reset_states()
            val_EG_overlap[i].reset_states()
            test_EG_overlap[i].reset_states()
        if FLAGS.task_id == 'DRO':
          test_00_accuracy.reset_states()
          test_01_accuracy.reset_states()
          test_10_accuracy.reset_states()
          test_11_accuracy.reset_states()
          val_00_accuracy.reset_states()
          val_01_accuracy.reset_states()
          val_10_accuracy.reset_states()
          val_11_accuracy.reset_states()
      else:
        train_accuracy.reset_states()

  vals_to_return = []
  vals_to_return.append(best_val_acc)
  vals_to_return.append(best_test_acc)
  if num_heads > 1:
    vals_to_return.append(best_val_acc_best_head)
    vals_to_return.append(best_test_acc_best_head)
    for i in range(len(extra_dsets)):
      vals_to_return.append(best_extra_dset_acc_best_head[i])
  for i in range(len(extra_dsets)):
    vals_to_return.append(best_extra_dset_acc[i])
  if FLAGS.task_id == 'DRO':
    vals_to_return.append(best_test_adj_acc)
    vals_to_return.append(best_test_00_acc)
    vals_to_return.append(best_test_01_acc)
    vals_to_return.append(best_test_10_acc)
    vals_to_return.append(best_test_11_acc)
    vals_to_return.append(best_val_00_acc)
    vals_to_return.append(best_val_01_acc)
    vals_to_return.append(best_val_10_acc)
    vals_to_return.append(best_val_11_acc)
  if FLAGS.monitor_EG_overlap:
    for i in range(len(prev_models)):
      vals_to_return.append(best_train_EG_overlap[i])
      vals_to_return.append(best_val_EG_overlap[i])
      vals_to_return.append(best_test_EG_overlap[i])
  if FLAGS.monitor_robustness_measures:
    for i in range(4):
      if i < 2:
        vals_to_return.append(best_train_gauss_robust[i])
      if i < 2:
        vals_to_return.append(best_val_gauss_robust[i])
      if i < 2:
        vals_to_return.append(best_test_gauss_robust[i])
      if i < 2:
        vals_to_return.append(best_train_mask_robust[i])
      if i < 2:
        vals_to_return.append(best_val_mask_robust[i])
      if i < 2:
        vals_to_return.append(best_test_mask_robust[i])
  if FLAGS.monitor_logit_correlation:
    for i in range(len(prev_models)):
      measurements_best_train_logit_correlation[i].create_measurement(
          objective_value=best_train_logit_correlation[i], step=0)
      vals_to_return.append(best_train_logit_correlation[i])
      measurements_best_val_logit_correlation[i].create_measurement(
          objective_value=best_val_logit_correlation[i], step=0)
      vals_to_return.append(best_val_logit_correlation[i])
      measurements_best_test_logit_correlation[i].create_measurement(
          objective_value=best_test_logit_correlation[i], step=0)
      vals_to_return.append(best_test_logit_correlation[i])
  if FLAGS.monitor_error_diversity:
    for i in range(len(prev_models)):
      vals_to_return.append(best_val_error_diversity[i])
      vals_to_return.append(best_test_error_diversity[i])
    
  return model, np.array(vals_to_return)
