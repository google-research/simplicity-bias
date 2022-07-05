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

"""Trains a classifier and evaluates finetuning performance. Specifically for sequential training of models"""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import numpy as np  # pylint: disable=unused-import
import resnet
import tensorflow as tf

from load_data import load_data
from train import train
from model import *

FLAGS = flags.FLAGS

flags.DEFINE_float('batch_norm_decay', 0.9, 'Batch norm decay parameter.')

flags.DEFINE_float('rr_weight', 0.0,
                   'Weight for the redundancy reduction term.')

flags.DEFINE_bool('use_rr_loss', False, 'Use redundancy reduction term.')

flags.DEFINE_bool('class_specific_rr_loss', True, 'Use class specific RR loss.')

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

flags.DEFINE_float('momentum', 0.9, 'momentum for SGD.')

flags.DEFINE_integer('train_batch_size', 64, 'Batch size for training.')

flags.DEFINE_integer('train_epochs', 100, 'Number of epochs to train for.')

flags.DEFINE_integer('train_epochs_finetune', 50,
                     'Number of epochs to finetune for.')

flags.DEFINE_integer('train_steps', 0,
                     'Number of train steps. If  > 0, overrides train epochs')

flags.DEFINE_integer(
    'train_steps_finetune', 0,
    'Number of train finetune steps. If  > 0, overrides train epochs finetune')

flags.DEFINE_integer(
    'eval_steps', 0,
    'Number of steps to eval for. If not provided, evals over entire dataset.')

flags.DEFINE_integer('lr_decay_gap', 50, 'Gap between decaying lr')

flags.DEFINE_float('lr_decay_factor', 0.5, 'Value to decay lr by')

flags.DEFINE_integer('val_batch_size', 256, 'Batch size for eval.')

flags.DEFINE_integer('test_batch_size', 256, 'Batch size for eval.')

flags.DEFINE_integer('num_train', -1, 'Num training examples.')

flags.DEFINE_integer('num_test', -1, 'Num test examples.')

flags.DEFINE_integer('num_train_finetune', 20000,
                     'Num training examples for finetuning.')

flags.DEFINE_integer('num_test_finetune', 2000,
                     'Num test examples for finetuning.')

flags.DEFINE_integer('buffer_size', 256, 'Buffer size for shuffling.')

flags.DEFINE_integer('checkpoint_epochs', 1,
                     'Number of epochs between checkpoints/summaries.')

flags.DEFINE_string('dataset', 'waterbirds', 'Name of a dataset.')

flags.DEFINE_bool(
    'cache_dataset', False,
    'Whether to cache the entire dataset in memory. If the dataset is '
    'ImageNet, this is a very bad idea, but for smaller datasets it can '
    'improve performance.')

flags.DEFINE_bool('use_dropout_pretrain', False,
                  'Whether to use dropout on the final layer for pretraining.')

flags.DEFINE_float('dropout_rate', 0.0,
                   'Dropout rate for final layer for pretraining.')

flags.DEFINE_bool(
    'finetune_from_random', False,
    'Whether to initialize the finetuning model using random weights.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_string('platform', 'GPU', 'To be run on GPU or TPU.')

flags.DEFINE_float('validation_split', 0.2,
                   'Validation split to use while training.')

flags.DEFINE_string(
    'checkpoint', None,
    'Loading from the given checkpoint for fine-tuning if a finetuning '
    'checkpoint does not already exist in model_dir.')

flags.DEFINE_string('optimizer', 'sgd', 'Optimizer to be used.')

flags.DEFINE_bool('project_out_prev_w', False, 'Project out previous w')

flags.DEFINE_bool(
    'zero_init_logits_layer', False,
    'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_bool('use_data_aug_with_DRO', False, 'True or False')

flags.DEFINE_integer(
    'fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linear head.')

flags.DEFINE_integer('keep_checkpoint_max', 5,
                     'Maximum number of checkpoints to keep.')

flags.DEFINE_enum('lr_decay_type', 'step_decay', ['step_decay', 'cosine_decay', 'warmup_cosine_decay'],
                  'Kind of decay to be used for learning rate')

flags.DEFINE_enum('learning_rate_scaling', 'linear', ['linear', 'sqrt'],
                  'Learning rate scaling to use')

flags.DEFINE_integer('warmup_steps', 0, 'Warmup steps for warmup and cosine decay')

flags.DEFINE_integer('warmup_epochs', 20, 'Warmup epochs for warmup and cosine decay')

flags.DEFINE_boolean(
    'global_bn', True,
    'Whether to aggregate BN statistics across distributed cores.')

flags.DEFINE_integer('proj_dim', 20, 'Output dimension of projection head')

flags.DEFINE_integer(
    'num_heads', 1,
    'Number of heads across which to decorrelate. One head is the standard config'
)

flags.DEFINE_integer('width_multiplier', 1,
                     'Multiplier to change width of network.')

flags.DEFINE_integer('resnet_depth', 50, 'Depth of ResNet.')

flags.DEFINE_float(
    'sk_ratio', 0.,
    'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')

flags.DEFINE_float('se_ratio', 0., 'If it is bigger than 0, it will enable SE.')

flags.DEFINE_float('weight_decay', 1e-4, 'weight decay to be used')

flags.DEFINE_float('logit_decay', 0.0, 'Decay to be used on logits')

flags.DEFINE_integer('image_size', 32, 'Input image size.')

flags.DEFINE_integer('val_epochs_gap', 10, 'Epoch gap to run validation after')

flags.DEFINE_integer('finetune_val_epochs_gap', 10,
                     'Epoch gap to run validation after')

flags.DEFINE_integer('val_steps_gap', 0, 'Steps gap to run validation after')

flags.DEFINE_integer('finetune_val_steps_gap', 0,
                     'Steps gap to run validation after')

flags.DEFINE_boolean('use_pretrained', True, 'whether to use pretrained model')

flags.DEFINE_string(
    'path', '/cns/sa-d/home/mloa/data/waterbirds_landbirds/waterbirds.pkl',
    'path for the dataset')

flags.DEFINE_bool('use_OOD_transform', False,
                  'Use data preprocessing specific to OOD dataset')

flags.DEFINE_float('clip_norm', None, 'global clip norm for the gradient')

flags.DEFINE_integer('num_runs', 1,
                     'Number of runs to average the evaluations over')

flags.DEFINE_integer('num_head_layers', 1, 'Number of layers to use in head')

flags.DEFINE_bool('use_early_stopping', False, 'Whether to use early stopping based on validation accuracy or not')

flags.DEFINE_integer('proj_layer', 0,
                     'Layer in head where applying projection for rr')

flags.DEFINE_integer('head_dim', 512, 'Dimension of head layers')

flags.DEFINE_string('model_dir', None, 'Path for loading/saving a model')

flags.DEFINE_string('model_finetune_dir', None,
                    'Path for loading/saving the model after finetuning')

flags.DEFINE_bool('use_seq_rr', True, 'Whether to use rr loss sequentially')

flags.DEFINE_integer('num_seq_models', 2, 'Number of sequential models to train')

flags.DEFINE_bool('load_model', False, 'whether to try to load model')

flags.DEFINE_bool('save_model', False, 'whether to save model')

flags.DEFINE_bool('lowerbound_rr', False, 'Whether to lowerbound rr loss')

flags.DEFINE_float('lowerbound_factor', 0.5, 'If lowerbound rr, by what factor of expected value')

flags.DEFINE_bool('use_exp_var_loss', False, 'Use explained away variance as loss')

flags.DEFINE_bool('use_MI_loss', False, 'Use MI based loss function')

flags.DEFINE_integer('CIFAR_label_1', 1, 'CIFAR class 1 for MNIST-CIFAR dataset')

flags.DEFINE_integer('CIFAR_label_2', 9, 'CIFAR class 2 for MNIST-CIFAR dataset')

flags.DEFINE_integer('MNIST_label_1', 0, 'MNIST class 1 for MNIST-CIFAR dataset')

flags.DEFINE_integer('MNIST_label_2', 1, 'MNIST class 2 for MNIST-CIFAR dataset')

flags.DEFINE_float('corr_frac', 1.0, 'Correlation factor of MNIST and CIFAR for MNIST-CIFAR dataset')

flags.DEFINE_bool('use_proj_head', True, 'Use a projection head in the model')

flags.DEFINE_bool('normalize_MI', False, 'Normalize MI based loss')

flags.DEFINE_bool('use_logit_decorr', False, 'Use logit decorrelation')

flags.DEFINE_bool('use_prob_decorr', False, 'Use probability decorrelation')

flags.DEFINE_bool('use_val_for_MI', False, 'Use validation set for MI based loss')

flags.DEFINE_bool('use_cifar_aug', False, 'Use CIFAR augmentation in MNIST-CIFAR dataset')

flags.DEFINE_bool('use_mnist_aug', False, 'Use MNIST augmentation in MNIST-CIFAR dataset')

flags.DEFINE_integer('num_classes', 10, 'Number of classes')

flags.DEFINE_bool('monitor_rr_grad_norms', False, 'Monitor rr gradient norms')

flags.DEFINE_float('use_rr_after_frac', 0.0, 'Use rr after a fraction of steps')

flags.DEFINE_bool('use_sq_MI', False, 'Use squared MI as loss instead of MI, for better gradients')

flags.DEFINE_bool('use_disagr_loss', False, 'use disagreement based loss')

flags.DEFINE_bool('normalize_MI_random', False, 'normalize MI by randomly shuffling probabilities')

flags.DEFINE_bool('use_num_sq_MI', False, 'Use the MI loss of the form (MI^2)/y so as to get properly scaled gradients')

flags.DEFINE_bool('use_stop_grad', False, 'Use stop gradient for MI normalization factor')

flags.DEFINE_bool('use_HSIC_loss', False, 'Use HSIC based independence test loss')

flags.DEFINE_bool('use_HSIC_diff', False, 'Use HSIC on logit difference')

flags.DEFINE_bool('lin_scale_rr_weight', False, 'Linearly scale down rr weight as number of sequential models goes up')

flags.DEFINE_integer('dataset_dim', 2, 'Dimension of LMS dataset')

flags.DEFINE_integer('num_lin', 1, 'Number of linear dimensions')

flags.DEFINE_integer('num_3_slabs', 1, 'Number of 3 slabs')

flags.DEFINE_integer('num_5_slabs', 0, 'Number of 5 slabs')

flags.DEFINE_integer('num_7_slabs', 0, 'Number of 7 slabs')

flags.DEFINE_bool('use_random_transform', False, 'Use random transformation of input coordinates')

flags.DEFINE_float('lin_margin', 0.1, 'Linear coordinate margin')

flags.DEFINE_float('slab_margin', 0.05, 'Slab coordinate margin')

flags.DEFINE_integer('fcn_layers', 3, 'Number of layers in FCN net for lms dataset')

flags.DEFINE_integer('hidden_dim', 512, 'Hidden dimension for FCN')

flags.DEFINE_bool('randomize_linear', False, 'Randomize linear coordinate in the dataset')

flags.DEFINE_bool('randomize_slabs', False, 'Randomize slab coordinates in the dataset')

flags.DEFINE_bool('turn_off_randomize_later', False, 'Turn off coordinate randomization later')

flags.DEFINE_bool('use_L4_reg', False, 'Use L4 instead of L2 regularization')

flags.DEFINE_bool('use_bn', True, 'Use BN in architecture')

flags.DEFINE_bool('use_HSIC_on_features', False, 'Use HSIC based loss on feature layers')

flags.DEFINE_integer('HSIC_feature_layer', 0, 'Feature layer to use HSIC loss on')

flags.DEFINE_multi_integer('HSIC_feature_layers', None, 'Feature layers to use HSIC loss on')

flags.DEFINE_bool('use_all_features_HSIC', False, 'Use features at all the layers for HSIC loss')

flags.DEFINE_bool('use_sq_HSIC', False, 'Square HSIC loss to manage gradients as loss goes down')

flags.DEFINE_bool('use_GAP_HSIC_features', True, 'Use GAP on HSIC features')

flags.DEFINE_bool('use_random_projections', False, 'Use random projections')

flags.DEFINE_integer('random_proj_dim', 1, 'Random projection dimension')

flags.DEFINE_bool('use_prev_logits_HSIC_features', False, 'Use logits of previous models for computing HSIC on features')

flags.DEFINE_bool('use_MNIST_labels', False, 'Use MNIST labels in MNIST-CIFAR')

flags.DEFINE_bool('switch_corr_later', False, 'Change correlation after first model')

flags.DEFINE_bool('switch_labels_later', False, 'Switch whether to use CIFAR or MNIST labels after first model')

flags.DEFINE_bool('monitor_EG_overlap', False, 'Monitor expected gradients overlap across models')

flags.DEFINE_bool('monitor_robustness_measures', False, 'Monitor Gaussian, mask and RDE based robustness measures')

flags.DEFINE_bool('monitor_error_diversity', False, 'Monitor error diversity')

flags.DEFINE_bool('monitor_logit_correlation', False, 'Monitor logit correlation')

flags.DEFINE_bool('sep_short_direct_branch', False, 'Separately make shortcut and direct branch independent of previous model')

flags.DEFINE_bool('use_pretrained_model_1', False, 'Utilise a pretrained first model.')

flags.DEFINE_string('pretrained_model_path', None, 'Path for first pretrained model')

flags.DEFINE_multi_string('pretrained_model_paths', None, 'Paths for pretrained models')

flags.DEFINE_multi_string('pretrained_checkpoint_paths', None, 'Paths for pretrained checkpoints')

flags.DEFINE_bool('use_indexed_checkpoints', False, 'Use particular checkpoint index')

flags.DEFINE_bool('check_tf_func', False, 'Check tf func')

flags.DEFINE_bool('use_FCN', False, 'Use FCN architecture')

flags.DEFINE_bool('monitor_EG_loss', False, 'Monitor EG loss')

flags.DEFINE_bool('use_equal_split', False, 'Use equal split for DRO setting')

flags.DEFINE_bool('use_EG_loss', False, 'Use expected gradients loss for avoiding collapse')

flags.DEFINE_integer('num_ref_EG_loss', 2, 'Number of referneces in EG loss')

flags.DEFINE_float('EG_loss_weight', 1e-3, 'Weight of EG loss')

flags.DEFINE_bool('binary_classification', False, 'Use binary classification and logistic loss')

flags.DEFINE_bool('use_color_labels', False, 'Use colors for label in color-MNIST or binary-color-MNIST')

flags.DEFINE_bool('use_CNN', False, 'Use custom CNN architecture')

flags.DEFINE_bool('use_HSIC_ratio', False, 'Use HSIC ratio as the loss')

flags.DEFINE_string(
    'master', 'local',
    "BNS name of the TensorFlow master to use. 'local' for GPU.")

flags.DEFINE_integer('project_out_rank', 0, 'Projecting certain dimensions out of input')

flags.DEFINE_float('project_out_factor', 0.0, 'Projecting out factor')

flags.DEFINE_float('eig_cutoff_factor', 0.0, 'Eigenvalue cutoff factor')

flags.DEFINE_integer('check_ranks_max', 10, 'Check rank of 1st hidden matrix')

flags.DEFINE_multi_integer('filters', [16, 32, 64], 'Filters to be used in a CNN')

flags.DEFINE_multi_integer('kernel_sizes', [3, 3, 3], 'kernel sizes to be used in a CNN')

flags.DEFINE_multi_integer('strides', [1, 2, 1], 'Strides to be used in a CNN')

flags.DEFINE_multi_integer('project_out_vecs', [1,2], 'Number of top SVD vectors to project out')

flags.DEFINE_bool('use_chizat_init', False, 'Whether to use chizat-bach initialization in head')

flags.DEFINE_bool('project_out_w', False, 'Project out w from representations')

flags.DEFINE_bool('use_complete_corr', False, 'Use complete correlation in DRO setting')

flags.DEFINE_bool('use_complete_corr_test', False, 'Use complete correlation in DRO setting')

flags.DEFINE_bool('flip_err_div_for_minority', False, 'Flip error diversity calc for minority classes')

flags.DEFINE_bool('measure_feat_robust', False, 'Measure robustness of features')

flags.DEFINE_float('max_gauss_noise_std', 5.0,  'Maximum gaussian noise std')

flags.DEFINE_boolean('use_tpu', True, 'Should we use TPU?')

flags.DEFINE_bool('check_torch_reps', False, 'Check torch reps')

flags.DEFINE_boolean(
    'train_split', 1,
    'Use train validation split while training, If set to false, use entire training dataset'
)

flags.DEFINE_bool('finetune_only_linear_head', False, 'Finetune only linear head')

_FRAC_POISON = flags.DEFINE_float('frac_poison', 0.,
                                  'Fraction of poisoned examples.')

_TASK_ID = flags.DEFINE_enum('task_id', 'DRO', [
    'Data-poisoning', 'DRO', 'Few-shot', 'CIFAR-10.2', 'CIFAR-10.2-finetune',
    'CINIC', 'CINIC-finetune'
], 'Specify the task that needs to be run.')

_FINETUNE_ONLY_HEAD = flags.DEFINE_bool(
    'finetune_only_head', False, 'whether to finetune head or the entire model')

_TRAIN_CLASSES = flags.DEFINE_multi_integer(
    'train_classes', [0, 1, 2, 3, 4], 'classes to train for few-shot learning')

_FINETUNE_CLASSES = flags.DEFINE_multi_integer(
    'finetune_classes', [5, 6, 7, 8, 9],
    'classes to finetune for few-shot learning')

TASK_STAGES = {
    'Data-poisoning': ['Train'],
    'DRO': ['Train'],
    'Few-shot': ['Train', 'Finetune'],
    'CIFAR-10.2': ['Train'],
    'CIFAR-10.2-finetune': ['Train'],
    'CINIC': ['Train'],
    'CINIC-finetune': ['Train'],
    'CIFAR-MNIST': ['Train'],
    'LMS': ['Train'],
    'MNIST': ['Train'],
    'color-MNIST': ['Train'],
    'Imagenette': ['Train'],
}


def createmodel(num_classes,
                head_dim,
                head_layers,
                proj_dim,
                proj_layer,
                use_proj=True,
                resnet_base=None,
                dropout_rate=0.0,
                num_heads=1):
  """Create a model with resnet base and a linear head."""
  if resnet_base is None:
    resnet_base = resnet.resnet(
        resnet_depth=FLAGS.resnet_depth,
        width_multiplier=FLAGS.width_multiplier,
        cifar_stem=FLAGS.image_size <= 32)
  # model = tf.keras.models.Sequential()
  # model.add(resnet_base)
  # if use_dropout:
  # model.add(tf.keras.layers.Dropout(rate=dropout_rate))
  # Removed sigmoid activation as need logits
  # model.add(tf.keras.layers.Dense(num_classes))
  if num_heads > 1:
    model = multihead_model(
        resnet_base,
        num_classes,
        proj_dim,
        proj_layer,
        head_dim + np.zeros(head_layers),
        num_heads,
        use_proj=use_proj,
        dropout_rate=dropout_rate)
  else:
    model = head_model(
        resnet_base,
        num_classes,
        proj_dim,
        proj_layer,
        head_dim + np.zeros(head_layers),
        use_proj = use_proj,
        dropout_rate=dropout_rate,
        use_bn=FLAGS.use_bn)

  return model


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info(tf.__version__)
  tf.config.set_soft_device_placement(True)
  logging.info('Successfully entered')
  # Setup the execution strategy
  if FLAGS.platform == 'GPU':
    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
  elif FLAGS.platform == 'TPU':
    # Setup and connect to TPU cluster
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.master)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
  else:
    raise ValueError('Unknown platform {}'.format(FLAGS.platform))

  logging.set_verbosity(logging.INFO)
  logging.info('Learning rate = %f, Batch size = %d', FLAGS.learning_rate,
               FLAGS.train_batch_size)

  adjust = 0
  if FLAGS.use_pretrained_model_1:
    adjust=FLAGS.num_seq_models - 1

  # Hack to make val epochs gap equal to train epochs
  if FLAGS.task_id in ['CIFAR-10.2', 'CIFAR-10.2-finetune', 'CINIC', 'CINIC-finetune', 'CIFAR-MNIST', 'MNIST', 'color-MNIST', 'LMS']:
    FLAGS.finetune_val_epochs_gap = FLAGS.train_epochs_finetune
    FLAGS.finetune_val_steps_gap = FLAGS.train_steps_finetune
    FLAGS.val_batch_size = FLAGS.train_batch_size
    FLAGS.test_batch_size = FLAGS.train_batch_size
  if FLAGS.binary_classification:
    FLAGS.num_classes = 1
  if FLAGS.proj_layer == -1:
    FLAGS.proj_layer  = FLAGS.num_head_layers
  if FLAGS.turn_off_randomize_later and FLAGS.use_random_transform:
    raise ValueError('Not supporting random transform and turn off randomize simultaneously as of now')
  for stage in TASK_STAGES[_TASK_ID.value]:
    if stage == 'Train':
      logging.info('Loading training dataset.')
      with strategy.scope():
        MI_ds = None
        W = None
        if FLAGS.task_id == 'CIFAR-10.2' or FLAGS.task_id == 'CIFAR-10.2-finetune':
          ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_test, train_len, OOD_train_len = load_data(
              dataset=FLAGS.dataset,
              desired_classes=_TRAIN_CLASSES.value,
              num_train=FLAGS.num_train,
              num_test=FLAGS.num_test,
              frac_poison=_FRAC_POISON.value,
              path=FLAGS.path)
        elif FLAGS.task_id in ['CINIC', 'CINIC-finetune', 'color-MNIST', 'CIFAR-MNIST']:
          ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_len, OOD_train_len = load_data(
              dataset=FLAGS.dataset,
              desired_classes=_TRAIN_CLASSES.value,
              num_train=FLAGS.num_train,
              num_test=FLAGS.num_test,
              frac_poison=_FRAC_POISON.value,
              path=FLAGS.path)
        elif FLAGS.task_id == 'LMS':
          ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_len, OOD_train_len, W = load_data(
              dataset=FLAGS.dataset,
              desired_classes=_TRAIN_CLASSES.value,
              num_train=FLAGS.num_train,
              num_test=FLAGS.num_test,
              frac_poison=_FRAC_POISON.value,
              path=FLAGS.path)
        else:
          ds_train, ds_val, ds_test, train_len = load_data(
              dataset=FLAGS.dataset,
              desired_classes=_TRAIN_CLASSES.value,
              num_train=FLAGS.num_train,
              num_test=FLAGS.num_test,
              frac_poison=_FRAC_POISON.value,
              path=FLAGS.path)

      logging.info('Successfully loaded training dataset')

      vals_avg = []
      vals_avg_2 = []
      vals_avg_3 = []
      vals_avg_4 = []
      vals_std = []
      vals_std_2 = []
      vals_std_3 = []
      vals_std_4 = []
      temp1 = FLAGS.randomize_linear
      temp2 = FLAGS.randomize_slabs
      temp3 = FLAGS.corr_frac
      temp4 = FLAGS.use_MNIST_labels
      for run in range(FLAGS.num_runs):
        if FLAGS.turn_off_randomize_later:
          FLAGS.randomize_linear = temp1
          FLAGS.randomize_slabs = temp2
          with strategy.scope():
            ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_len, OOD_train_len, W = load_data(
                dataset=FLAGS.dataset,
                desired_classes=_TRAIN_CLASSES.value,
                num_train=FLAGS.num_train,
                num_test=FLAGS.num_test,
                frac_poison=_FRAC_POISON.value,
                path=FLAGS.path)
        if FLAGS.switch_corr_later or FLAGS.switch_labels_later:
          if FLAGS.switch_corr_later:
            FLAGS.corr_frac = temp3
          if FLAGS.switch_labels_later:
            FLAGS.use_MNIST_labels = temp4
          with strategy.scope():
            ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_len, OOD_train_len = load_data(
              dataset=FLAGS.dataset,
              desired_classes=_TRAIN_CLASSES.value,
              num_train=FLAGS.num_train,
              num_test=FLAGS.num_test,
              frac_poison=_FRAC_POISON.value,
              path=FLAGS.path)
        models = []
        for model_seq in range(FLAGS.num_seq_models):
          with strategy.scope():
            if model_seq == 1 and FLAGS.turn_off_randomize_later:
              FLAGS.randomize_linear = False
              FLAGS.randomize_slabs = False
              ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_len, OOD_train_len, W = load_data(
                  dataset=FLAGS.dataset,
                  desired_classes=_TRAIN_CLASSES.value,
                  num_train=FLAGS.num_train,
                  num_test=FLAGS.num_test,
                  frac_poison=_FRAC_POISON.value,
                  path=FLAGS.path)
            if model_seq == 1 and (FLAGS.switch_corr_later or FLAGS.switch_labels_later):
              if FLAGS.switch_corr_later:
                FLAGS.corr_frac = 1.0
              if FLAGS.switch_labels_later:
                FLAGS.use_MNIST_labels = not(temp4)
              ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_len, OOD_train_len = load_data(
                dataset=FLAGS.dataset,
                desired_classes=_TRAIN_CLASSES.value,
                num_train=FLAGS.num_train,
                num_test=FLAGS.num_test,
                frac_poison=_FRAC_POISON.value,
                path=FLAGS.path)
            if FLAGS.task_id == 'DRO' or FLAGS.task_id == 'Imagenette':
              if FLAGS.use_pretrained:
                # tmp_filepath = '/tmp/weights_resnet50.h5'
                # tf.io.gfile.copy('/cns/sa-d/home/mloa/models/waterbirds/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', tmp_filepath, overwrite=True)
                if FLAGS.check_torch_reps:
                  base_model = tf.keras.Sequential()
                  base_model.add(tf.keras.layers.Layer())
                else:
                  if model_seq==0:
                    base_model = tf.keras.applications.resnet50.ResNet50(
                        include_top=False, weights='imagenet', pooling='avg')
                # tf.io.gfile.remove(tmp_filepath)
              else:
                if FLAGS.resnet_depth == 50:
                  base_model = tf.keras.applications.resnet50.ResNet50(
                      include_top=False, weights=None, pooling='avg')
                else:
                  base_model = resnet.resnet(
                      resnet_depth=FLAGS.resnet_depth,
                      width_multiplier=FLAGS.width_multiplier,
                      cifar_stem=False)
              #layer_names = ['input_1', 'pool1_pool', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'avg_pool']
              #layers = [base_model.get_layer(x).output for x in layer_names]
              #base_model = tf.keras.Model(base_model.input, outputs=layers)
              model = createmodel(
                  FLAGS.num_classes,
                  FLAGS.head_dim,
                  FLAGS.num_head_layers,
                  FLAGS.proj_dim,
                  FLAGS.proj_layer,
                  resnet_base=base_model,
                  dropout_rate=FLAGS.dropout_rate,
                  num_heads=FLAGS.num_heads,
                  use_proj=FLAGS.use_proj_head)
            elif FLAGS.task_id in ['CIFAR-10.2', 'CIFAR-10.2-finetune', 'CINIC', 'CINIC-finetune', 'CIFAR-MNIST', 'MNIST', 'color-MNIST']:
              if FLAGS.use_pretrained:
                if model_seq == 0:
                  resnet_base = tf.keras.applications.resnet50.ResNet50(
                      include_top=False, weights='imagenet', pooling='avg')
              elif FLAGS.use_FCN:
                hid_dims = FLAGS.hidden_dim + np.zeros(FLAGS.fcn_layers)
                resnet_base = FCN_backbone(hid_dims)
              elif FLAGS.use_CNN:
                resnet_base = CNN_backbone(filters = FLAGS.filters, kernel_sizes=FLAGS.kernel_sizes,
                                           strides=FLAGS.strides)
              else:
                resnet_base = None
              model = createmodel(
                  FLAGS.num_classes,
                  FLAGS.head_dim,
                  FLAGS.num_head_layers,
                  FLAGS.proj_dim,
                  FLAGS.proj_layer,
                  resnet_base=resnet_base,
                  dropout_rate=FLAGS.dropout_rate,
                  num_heads=FLAGS.num_heads,
                  use_proj=FLAGS.use_proj_head)
            elif FLAGS.task_id == 'LMS':
              if FLAGS.use_pretrained:
                resnet_base = tf.keras.Sequential()
                resnet_base.add(tf.keras.layers.Layer())
              else:
                hid_dims = FLAGS.hidden_dim + np.zeros(FLAGS.fcn_layers)
                resnet_base = FCN_backbone(hid_dims, use_bn=FLAGS.use_bn)
              model = createmodel(
                  FLAGS.num_classes,
                  FLAGS.head_dim,
                  FLAGS.num_head_layers,
                  FLAGS.proj_dim,
                  FLAGS.proj_layer,
                  resnet_base=resnet_base,
                  dropout_rate=FLAGS.dropout_rate,
                  num_heads=FLAGS.num_heads,
                  use_proj=FLAGS.use_proj_head)
            else:
              model = createmodel(
                  len(_TRAIN_CLASSES.value),
                  512,
                  0,
                  512,
                  0,
                  resnet_base=None,
                  use_proj=False,
                  dropout_rate=FLAGS.dropout_rate)
            if FLAGS.use_pretrained_model_1 and model_seq<FLAGS.num_seq_models-1:
              if FLAGS.finetune_only_head:
                save_vars = {}
                for i in range(len(model.layers)):
                  if i != 0:
                    save_vars['model{}'.format(str(i))] = model.layers[i]
                ckpt = tf.train.Checkpoint(**save_vars)
              else:
                ckpt = tf.train.Checkpoint(model=model)
              checkpoint_manager = tf.train.CheckpointManager(
                  ckpt, directory=FLAGS.pretrained_model_paths[model_seq], max_to_keep=1)
              logging.info('Loading model')
              logging.info(checkpoint_manager.checkpoints)
              if FLAGS.use_indexed_checkpoints:
                latest_ckpt = FLAGS.pretrained_checkpoint_paths[model_seq]
              else:
                latest_ckpt = checkpoint_manager.latest_checkpoint
              logging.info(latest_ckpt)
              if latest_ckpt:
                logging.info('Found checkpoint')
                ckpt.restore(latest_ckpt).expect_partial()
              models.append(model)
              continue
            steps_per_epoch = (train_len // FLAGS.train_batch_size) + 1
            if FLAGS.lr_decay_type == 'step_decay':
              decay_epochs = FLAGS.lr_decay_gap * (
                  np.arange(FLAGS.train_epochs // FLAGS.lr_decay_gap, dtype=int) +
                  1)
              decay_steps = list(decay_epochs * steps_per_epoch)
              lr_vals = []
              for i in range(len(decay_steps) + 1):
                lr_vals.append(FLAGS.learning_rate *
                               np.power(FLAGS.lr_decay_factor, i))
              lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                  decay_steps, lr_vals)
            elif FLAGS.lr_decay_type == 'cosine_decay':
              if FLAGS.train_steps > 0:
                decay_steps = FLAGS.train_steps
              else:
                decay_steps = FLAGS.train_epochs * steps_per_epoch
              lr_sched = tf.keras.optimizers.schedules.CosineDecay(
                  FLAGS.learning_rate, decay_steps)
            elif FLAGS.lr_decay_type == 'warmup_cosine_decay':
                lr_sched = WarmUpAndCosineDecay(FLAGS.learning_rate, train_len)
            else:
              raise ValueError('Unknown lr decay schedule {}'.format(
                  FLAGS.lr_decay_type))
            # Instantiate optimizer
            if FLAGS.optimizer == 'Adam':
              optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
              if FLAGS.project_out_rank > 0:
                optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr_sched)
              else:
                optimizer2 = None
            elif FLAGS.optimizer == 'sgd':
              optimizer = tf.keras.optimizers.SGD(
                  learning_rate=lr_sched, momentum=FLAGS.momentum)
              if FLAGS.project_out_rank > 0:
                optimizer2 = tf.keras.optimizers.SGD(learning_rate=lr_sched, momentum=FLAGS.momentum)
              else:
                optimizer2 = None
            else:
              raise ValueError('Unknown optimizer {}'.format(FLAGS.optimizer))

            logging.info('Starting training')
            epoch = tf.Variable(0)
            if FLAGS.finetune_only_head:
              save_vars = {}
              save_vars['optimizer'] = optimizer
              save_vars['epoch'] = epoch
              for i in range(len(model.layers)):
                if i != 0:
                  save_vars['model{}'.format(str(i))] = model.layers[i]
              ckpt = tf.train.Checkpoint(**save_vars)
            else:
              ckpt = tf.train.Checkpoint(
                  model=model, optimizer=optimizer, epoch=epoch)
            checkpoint_manager = tf.train.CheckpointManager(
                ckpt,
                directory='{}{}/{}/'.format(FLAGS.model_dir, str(run), str(model_seq)),
                max_to_keep=100)
            if FLAGS.load_model:
              logging.info('Loading model')
              latest_ckpt = checkpoint_manager.latest_checkpoint
              if latest_ckpt:
                logging.info('Found checkpoint')
                ckpt.restore(latest_ckpt).expect_partial()
            if FLAGS.task_id in ['CIFAR-10.2', 'CIFAR-10.2-finetune', 'CINIC', 'CINIC-finetune', 'CIFAR-MNIST', 'LMS', 'color-MNIST']:
              model, vals = train(
                  model,
                  ds_train,
                  ds_val,
                  ds_test,
                  train_len,
                  optimizer,
                  run=run,
                  extra_dsets=[ds_OOD_test],
                  num_heads=FLAGS.num_heads,
                  ckpt_manager=checkpoint_manager,
                  prev_models=models,
                  MI_ds = MI_ds,
                  optimizer2 = optimizer2,
                  W = W,
                  only_head=FLAGS.finetune_only_head,
                  only_linear_head=FLAGS.finetune_only_linear_head)
            elif FLAGS.task_id == 'DRO' or FLAGS.task_id == 'Imagenette':
              model, vals = train(
                  model,
                  ds_train,
                  ds_val,
                  ds_test,
                  train_len,
                  optimizer,
                  run=run,
                  only_head=FLAGS.finetune_only_head,
                  ckpt_manager=checkpoint_manager,
                  prev_models=models,
                  only_linear_head=FLAGS.finetune_only_linear_head)
            else:
              model, vals = train(
                  model,
                  ds_train,
                  ds_val,
                  ds_test,
                  train_len,
                  optimizer,
                  run=run,
                  ckpt_manager=checkpoint_manager,
                  prev_models=models)
          models.append(model)
          if run == 0:
            vals_avg.append(vals)
          else:
            vals_avg[model_seq-adjust] = (1.0 / (run + 1.0)) * vals + (run / (run + 1.0)) * vals_avg[model_seq-adjust]
          if run == 0:
            vals_std.append(vals**2)
          else:
            vals_std[model_seq-adjust] = (1.0 / (run + 1.0)) * (vals**2) + (run /
                                                          (run + 1.0)) * vals_std[model_seq-adjust]
          logging.info('Finished training')
          if FLAGS.task_id in ['CIFAR-10.2-finetune', 'CINIC-finetune', 'CIFAR-MNIST', 'color-MNIST', 'LMS']:
            with strategy.scope():
              if FLAGS.use_pretrained:
                base_model = models[0].layers[0]
                head_models = []
                for i in range(len(models)):
                  curr_model_layers = []
                  for layer_ind in range(len(models[i].layers)-1):
                    if layer_ind > 0:
                      curr_model_layers.append(models[i].layers[layer_ind])
                  curr_model = tf.keras.Sequential(layers=curr_model_layers)
                  head_models.append(curr_model)
                model1 = base_multi_head_model(base_model=base_model, head_models=head_models, num_classes=FLAGS.num_classes)
              else:
                temp_models = []
                for i in range(len(models)):
                  curr_model_layers = []
                  for layer_ind in range(len(models[i].layers)-1):
                    curr_model_layers.append(models[i].layers[layer_ind])
                  curr_model = tf.keras.Sequential(layers=curr_model_layers)
                  temp_models.append(curr_model)
                model1 = multi_base_model(temp_models, FLAGS.num_classes)
              steps_per_epoch = (train_len // FLAGS.train_batch_size) + 1
              if FLAGS.lr_decay_type == 'step_decay':
                decay_epochs = FLAGS.lr_decay_gap * (
                    np.arange(
                        FLAGS.train_epochs_finetune // FLAGS.lr_decay_gap,
                        dtype=int) + 1)
                decay_steps = list(decay_epochs * steps_per_epoch)
                lr_vals = []
                for i in range(len(decay_steps) + 1):
                  lr_vals.append(FLAGS.learning_rate *
                                 np.power(FLAGS.lr_decay_factor, i))
                lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    decay_steps, lr_vals)
              elif FLAGS.lr_decay_type == 'cosine_decay' or FLAGS.lr_decay_type == 'warmup_cosine_decay':
                if FLAGS.train_steps_finetune > 0:
                  decay_steps = FLAGS.train_steps_finetune
                else:
                  decay_steps = FLAGS.train_epochs_finetune * steps_per_epoch
                lr_sched = tf.keras.optimizers.schedules.CosineDecay(
                    FLAGS.learning_rate, decay_steps)
              else:
                raise ValueError('Unknown lr decay schedule {}'.format(
                    FLAGS.lr_decay_type))
              # Instantiate optimizer
              if FLAGS.optimizer == 'Adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
              elif FLAGS.optimizer == 'sgd':
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=lr_sched, momentum=FLAGS.momentum)
              else:
                raise ValueError('Unknown optimizer {}'.format(FLAGS.optimizer))
              model1, vals = train(
                  model1,
                  ds_OOD_train,
                  ds_OOD_val,
                  ds_OOD_test,
                  OOD_train_len,
                  optimizer,
                  run=run,
                  only_head=True,
                  only_linear_head=True,
                  finetune=True,
                  prefix='OOD_finetune_{}_'.format(model_seq))
          if run == 0:
            vals_avg_2.append(vals)
          else:
            vals_avg_2[model_seq-adjust] = (1.0 / (run + 1.0)) * vals + (run /
                                                       (run + 1.0)) * vals_avg_2[model_seq-adjust]
          if run == 0:
            vals_std_2.append(vals**2)
          else:
            vals_std_2[model_seq-adjust] = (1.0 /
                          (run + 1.0)) * (vals**2) + (run /
                                                      (run + 1.0)) * vals_std_2[model_seq-adjust]
          logging.info('Finished finetuning')
          if FLAGS.monitor_robustness_measures:
            with strategy.scope():
              if FLAGS.use_pretrained:
                base_model = models[0].layers[0]
                head_models = []
                for i in range(len(models)):
                  curr_model_layers = []
                  for layer_ind in range(len(models[i].layers)-1):
                    if layer_ind > 0:
                      curr_model_layers.append(models[i].layers[layer_ind])
                  curr_model = tf.keras.Sequential(layers=curr_model_layers)
                  head_models.append(curr_model)
                model1 = base_multi_head_model(base_model=base_model, head_models=head_models, num_classes=FLAGS.num_classes)
              else:
                temp_models = []
                for i in range(len(models)):
                  curr_model_layers = []
                  for layer_ind in range(len(models[i].layers)-1):
                    curr_model_layers.append(models[i].layers[layer_ind])
                  curr_model = tf.keras.Sequential(layers=curr_model_layers)
                  temp_models.append(curr_model)
                model1 = multi_base_model(temp_models, FLAGS.num_classes)
              steps_per_epoch = (train_len // FLAGS.train_batch_size) + 1
              if FLAGS.lr_decay_type == 'step_decay':
                decay_epochs = FLAGS.lr_decay_gap * (
                    np.arange(
                        FLAGS.train_epochs_finetune // FLAGS.lr_decay_gap,
                        dtype=int) + 1)
                decay_steps = list(decay_epochs * steps_per_epoch)
                lr_vals = []
                for i in range(len(decay_steps) + 1):
                  lr_vals.append(FLAGS.learning_rate *
                                 np.power(FLAGS.lr_decay_factor, i))
                lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    decay_steps, lr_vals)
              elif FLAGS.lr_decay_type == 'cosine_decay' or FLAGS.lr_decay_type == 'warmup_cosine_decay':
                if FLAGS.train_steps_finetune > 0:
                  decay_steps = FLAGS.train_steps_finetune
                else:
                  decay_steps = FLAGS.train_epochs_finetune * steps_per_epoch
                lr_sched = tf.keras.optimizers.schedules.CosineDecay(
                    FLAGS.learning_rate, decay_steps)
              else:
                raise ValueError('Unknown lr decay schedule {}'.format(
                    FLAGS.lr_decay_type))
              # Instantiate optimizer
              if FLAGS.optimizer == 'Adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
              elif FLAGS.optimizer == 'sgd':
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=lr_sched, momentum=FLAGS.momentum)
              else:
                raise ValueError('Unknown optimizer {}'.format(FLAGS.optimizer))
              model1, vals = train(
                  model1,
                  ds_train,
                  ds_val,
                  ds_test,
                  train_len,
                  optimizer,
                  run=run,
                  only_head=True,
                  only_linear_head=True,
                  finetune=True,
                  prefix='non_gauss_finetune_{}_'.format(model_seq))
            if run == 0:
              vals_avg_3.append(vals)
            else:
              vals_avg_3[model_seq-adjust] = (1.0 / (run + 1.0)) * vals + (run /
                                                         (run + 1.0)) * vals_avg_3[model_seq-adjust]
            if run == 0:
              vals_std_3.append(vals**2)
            else:
              vals_std_3[model_seq-adjust] = (1.0 /
                            (run + 1.0)) * (vals**2) + (run /
                                                        (run + 1.0)) * vals_std_3[model_seq-adjust]
          if FLAGS.monitor_robustness_measures:
            with strategy.scope():
              if FLAGS.use_pretrained:
                base_model = models[0].layers[0]
                head_models = []
                for i in range(len(models)):
                  curr_model_layers = []
                  for layer_ind in range(len(models[i].layers)-1):
                    if layer_ind > 0:
                      curr_model_layers.append(models[i].layers[layer_ind])
                  curr_model = tf.keras.Sequential(layers=curr_model_layers)
                  head_models.append(curr_model)
                model2 = base_multi_head_model(base_model=base_model, head_models=head_models, num_classes=FLAGS.num_classes)
              else:
                temp_models = []
                for i in range(len(models)):
                  curr_model_layers = []
                  for layer_ind in range(len(models[i].layers)-1):
                    curr_model_layers.append(models[i].layers[layer_ind])
                  curr_model = tf.keras.Sequential(layers=curr_model_layers)
                  temp_models.append(curr_model)
                model2 = multi_base_model(temp_models, FLAGS.num_classes)
              steps_per_epoch = (train_len // FLAGS.train_batch_size) + 1
              if FLAGS.lr_decay_type == 'step_decay':
                decay_epochs = FLAGS.lr_decay_gap * (
                    np.arange(
                        FLAGS.train_epochs_finetune // FLAGS.lr_decay_gap,
                        dtype=int) + 1)
                decay_steps = list(decay_epochs * steps_per_epoch)
                lr_vals = []
                for i in range(len(decay_steps) + 1):
                  lr_vals.append(FLAGS.learning_rate *
                                 np.power(FLAGS.lr_decay_factor, i))
                lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    decay_steps, lr_vals)
              elif FLAGS.lr_decay_type == 'cosine_decay' or FLAGS.lr_decay_type == 'warmup_cosine_decay':
                if FLAGS.train_steps_finetune > 0:
                  decay_steps = FLAGS.train_steps_finetune
                else:
                  decay_steps = FLAGS.train_epochs_finetune * steps_per_epoch
                lr_sched = tf.keras.optimizers.schedules.CosineDecay(
                    FLAGS.learning_rate, decay_steps)
              else:
                raise ValueError('Unknown lr decay schedule {}'.format(
                    FLAGS.lr_decay_type))
              # Instantiate optimizer
              if FLAGS.optimizer == 'Adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
              elif FLAGS.optimizer == 'sgd':
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=lr_sched, momentum=FLAGS.momentum)
              else:
                raise ValueError('Unknown optimizer {}'.format(FLAGS.optimizer))
              model2, vals = train(
                  model2,
                  ds_train,
                  ds_val,
                  ds_test,
                  train_len,
                  optimizer,
                  run=run,
                  only_head=True,
                  only_linear_head=True,
                  finetune=True,
                  prefix='gauss_finetune_{}_'.format(model_seq),
                  add_gauss_noise=True)
            if run == 0:
              vals_avg_4.append(vals)
            else:
              vals_avg_4[model_seq-adjust] = (1.0 / (run + 1.0)) * vals + (run /
                                                         (run + 1.0)) * vals_avg_4[model_seq-adjust]
            if run == 0:
              vals_std_4.append(vals**2)
            else:
              vals_std_4[model_seq-adjust] = (1.0 /
                            (run + 1.0)) * (vals**2) + (run /
                                                        (run + 1.0)) * vals_std_4[model_seq-adjust]

      for i in range(FLAGS.num_seq_models-adjust):
        vals_std[i] = np.sqrt(vals_std[i] - vals_avg[i]**2)
        if len(vals_std_2) > 0:
          vals_std_2[i] = np.sqrt(vals_std_2[i] - vals_avg_2[i]**2)
        if FLAGS.monitor_robustness_measures:
          vals_std_3[i] = np.sqrt(vals_std_3[i] - vals_avg_3[i]**2)
          vals_std_4[i] = np.sqrt(vals_std_4[i] - vals_avg_4[i]**2)
    else:
      ValueError('Unknown Stage {}'.format(stage))

if __name__ == '__main__':
  app.run(main)
