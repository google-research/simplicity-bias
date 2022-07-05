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

"""Data loading utils."""
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_addons import image as tfa_image
import math
import os
import pickle
from lms_dataset import get_lms_data

FLAGS = flags.FLAGS
_PATCH_SIZE = flags.DEFINE_integer('patch_size', 3,
                                   'Patch size for data poisoning.')
_SHUFFLE_BUFFER_SIZE = flags.DEFINE_integer('shuffle_buffer_size', 10000,
                                            'Buffer size for shuffling.')


def create_patch(
    x,
    y,
    patch_size: int,
    max_val: int = 1,
    min_val: int = -1,
    patch_val_range: float = 0.25,  # modifying patch_val_range
    flip: bool = True):
  """Adds a discriminative patch to given images that lie in [-1.,1.].

  Args:
    x: list of images
    y: list of labels
    patch_size: patch size
    max_val : max pixel value
    min_val : min pixel value
    patch_val_range: range of patch
    flip: whether to flip the patch

  Returns:
    a list of images with same dimensions as x
  """
  new_arr = []
  np.random.seed(0)  # used to fix the random value we generate everytime
  img_shape = x[0].shape
  for i in range(len(x)):
    if (y[i] == 1) is flip:
      # rand_val = np.random.randint(low=0, high=patch_val_range)
      rand_val = np.random.uniform(low=min_val, high=min_val + patch_val_range)
      new_img = np.zeros(img_shape)
      new_img[:patch_size, :patch_size, :] = rand_val
      new_img[patch_size:, :patch_size, :] = x[i][patch_size:, :patch_size, :]
      new_img[:patch_size, patch_size:, :] = x[i][:patch_size, patch_size:, :]
      new_img[patch_size:, patch_size:, :] = x[i][patch_size:, patch_size:, :]
    else:
      rand_val = np.random.randint(low=max_val - patch_val_range, high=max_val)
      new_img = np.zeros(img_shape)
      new_img[:patch_size, :patch_size, :] = rand_val
      new_img[patch_size:, :patch_size, :] = x[i][patch_size:, :patch_size, :]
      new_img[:patch_size, patch_size:, :] = x[i][:patch_size, patch_size:, :]
      new_img[patch_size:, patch_size:, :] = x[i][patch_size:, patch_size:, :]
    new_arr.append(new_img)
  return new_arr


def normalize_test_data(x, mean_arr, std_arr):
  """Standard Normalization transformation of data.

  Args:
    x: Array containing trainining data
    mean_arr: List of means
    std_arr: List of std

  Returns:
    Normalized data.
  """
  if (x.shape[3] != len(mean_arr)) or (x.shape[3] != len(std_arr)):
    raise ValueError(f'Invalid arguments num data channels {x.shape[3]}, length of mean array {len(mean_arr)} and length of std array {len(std_arr)}')  # pylint: disable=line-too-long
  for i in range(x.shape[3]):
    x[:, :, :, i] = (x[:, :, :, i] - mean_arr[i]) / std_arr[i]
  return x


def normalize_train_data(x):
  """Standard Normalization transformation of data.

  Args:
    x: Array containing trainining data

  Returns:
    Normalized data.
  """
  mean_arr = []
  std_arr = []
  for i in range(x.shape[3]):
    mean_arr.append(np.mean(x[:, :, :, i]))
    std_arr.append(np.std(x[:, :, :, i]))
  for i in range(x.shape[3]):
    x[:, :, :, i] = (x[:, :, :, i] - mean_arr[i]) / std_arr[i]
  return x, mean_arr, std_arr


def waterbirds_train_transform(x):
  MEAN = np.array([0.485, 0.456, 0.406])
  STD = np.array([0.229, 0.224, 0.225])
  # Convert CHW to HWC
  image = tf.transpose(x['inputs'], [1,2,0])
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.resize(image, [224, 224])
  image = tf.image.random_flip_left_right(image=image)
  image = tf.image.random_brightness(image, 0.25)
  image = tf.image.random_contrast(image, 0.75, 1.25)
  image = tf.image.random_saturation(image, 0.75, 1.25)
  image = (image - MEAN) / STD
  return {'image': image, 'label': x['labels'], 'metadata': x['metadata']}


def waterbirds_test_transform(x):
  MEAN = np.array([0.485, 0.456, 0.406])
  STD = np.array([0.229, 0.224, 0.225])
  if FLAGS.check_torch_reps:
    return {'image': x['inputs'], 'label': x['labels'], 'metadata': x['metadata']}
  else:
    # Convert CHW to HWC
    image = tf.transpose(x['inputs'], [1,2,0])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [224, 224])
    image = (image - MEAN) / STD
    return {'image': image, 'label': x['labels'], 'metadata': x['metadata']}


def cifar102_train_transform(x):
  MEAN = np.array([0.5133, 0.4973, 0.4619])
  STD = np.array([0.2110, 0.2100, 0.2119])
  image = tf.pad(x['image'], paddings=[[4, 4], [4, 4], [0, 0]])
  image = tf.image.random_crop(image, [32, 32, 3])
  image = tf.image.random_flip_left_right(image)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = (image - MEAN) / STD
  return {'image': image, 'label': x['label']}


def cinic_train_transform(x):
  MEAN = [0.47889522, 0.47227842, 0.43047404]
  STD = [0.24205776, 0.23828046, 0.25874835]
  image = tf.pad(x['image'], paddings=[[4, 4], [4, 4], [0, 0]])
  image = tf.image.random_crop(image, [32, 32, 3])
  image = tf.image.random_flip_left_right(image)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = (image - MEAN) / STD
  return {'image': image, 'label': x['label']}


def cifar10_train_transform(x):
  MEAN = np.array([0.4914, 0.4822, 0.4465])
  STD = np.array([0.2023, 0.1994, 0.2010])
  image = tf.pad(x['image'], paddings=[[4, 4], [4, 4], [0, 0]])
  image = tf.image.random_crop(image, [32, 32, 3])
  image = tf.image.random_flip_left_right(image)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = (image - MEAN) / STD
  return {'image': image, 'label': x['label']}


def cifar102_test_transform(x):
  MEAN = np.array([0.5133, 0.4973, 0.4619])
  STD = np.array([0.2110, 0.2100, 0.2119])
  image = tf.image.convert_image_dtype(x['image'], dtype=tf.float32)
  image = (image - MEAN) / STD
  return {'image': image, 'label': x['label']}


def cinic_test_transform(x):
  MEAN = [0.47889522, 0.47227842, 0.43047404]
  STD = [0.24205776, 0.23828046, 0.25874835]
  image = tf.image.convert_image_dtype(x['image'], dtype=tf.float32)
  image = (image - MEAN) / STD
  return {'image': image, 'label': x['label']}


def cifar10_test_transform(x):
  MEAN = np.array([0.4914, 0.4822, 0.4465])
  STD = np.array([0.2023, 0.1994, 0.2010])
  image = tf.image.convert_image_dtype(x['image'], dtype=tf.float32)
  image = (image - MEAN) / STD
  return {'image': image, 'label': x['label']}


def mnist_cifar_transform(x):
  MNIST_MEAN = np.array([0.5, 0.5, 0.5])
  MNIST_STD = np.array([0.5, 0.5, 0.5])
  mnist_image = x['MNIST_image']
  if FLAGS.use_mnist_aug:
    mnist_image = tfa_image.rotate(mnist_image, math.pi*tf.random.uniform([1])/6.0, interpolation='bilinear')
    mnist_image = tf.pad(mnist_image, paddings=[[4, 4], [4, 4], [0, 0]])
    mnist_image = tf.image.random_crop(mnist_image, [32, 32, 3])
  mnist_image = tf.image.convert_image_dtype(mnist_image, dtype=tf.float32)
  mnist_image = (mnist_image - MNIST_MEAN) / MNIST_STD

  CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465])
  CIFAR_STD = np.array([0.2023, 0.1994, 0.2010])
  cifar_image = x['CIFAR_image']
  if FLAGS.use_cifar_aug:
    cifar_image = tf.pad(cifar_image, paddings=[[4, 4], [4, 4], [0, 0]])
    cifar_image = tf.image.random_crop(cifar_image, [32, 32, 3])
    cifar_image = tf.image.random_flip_left_right(cifar_image)
  cifar_image = tf.image.convert_image_dtype(cifar_image, dtype=tf.float32)
  cifar_image = (cifar_image - CIFAR_MEAN) / CIFAR_STD

  final_image = tf.concat([mnist_image, cifar_image], axis=0)
  return {'image': final_image, 'label': x['label']}

def mnist_transform(x):
  MNIST_MEAN = np.array([0.5, 0.5, 0.5])
  MNIST_STD = np.array([0.5, 0.5, 0.5])
  mnist_image = tf.image.convert_image_dtype(x['image'], dtype=tf.float32)
  mnist_image = (mnist_image - MNIST_MEAN) / MNIST_STD
  return {'image': mnist_image, 'label': x['label']}


def perturb_train(x):
  image = tf.image.random_flip_left_right(image=x['image'])
  image = tf.image.random_brightness(image, 0.25)
  image = tf.image.random_contrast(image, 0.75, 1.25)
  image = tf.image.random_saturation(image, 0.75, 1.25)
  return {'image': image, 'label': x['label']}


def perturb_test(x):
  x = tf.image.central_crop(x, central_fraction=1)
  x = tf.image.random_flip_left_right(image=x)
  x = tf.image.random_brightness(x, 0.25)
  x = tf.image.random_contrast(x, 0.75, 1.25)
  x = tf.image.random_saturation(x, 0.75, 1.25)
  return x


def get_binary_datasets(X, Y, y1, y2):
  idx0 = (Y == y1).nonzero()[0]
  idx1 = (Y == y2).nonzero()[0]
  idx = np.concatenate((idx0, idx1))
  X_, Y_ = X[idx, :], (Y[idx] == y2).astype(int)
  P = np.random.permutation(len(X_))
  X_, Y_ = X_[P], Y_[P]
  return X_, Y_


def make_MNIST_CIFAR_compatible(X):
  X = np.stack([np.pad(X[i, :, :, 0], 2)[:, :, None] for i in range(len(X))])
  X = np.repeat(X, 3, axis=3)
  return X


def combine_datasets(X1, Y1, X2, Y2, corr_frac=1.0):
  # final Y is returned according to Y1
  X1_0 = X1[Y1 == 0]
  X1_1 = X1[Y1 == 1]
  X2_0 = X2[Y2 == 0]
  X2_1 = X2[Y2 == 1]

  num_class_0 = min(len(X1_0), len(X2_0))
  num_class_1 = min(len(X1_1), len(X2_1))
  per_class_examples = min(num_class_0, num_class_1)
  final_X1 = np.concatenate(
      [X1_0[:per_class_examples], X1_1[:per_class_examples]], axis=0)
  final_X2 = np.concatenate([
      X2_0[:int(corr_frac * per_class_examples)],
      X2_1[int(corr_frac * per_class_examples):per_class_examples],
      X2_1[:int(corr_frac * per_class_examples)],
      X2_0[int(corr_frac * per_class_examples):per_class_examples]
  ],
                            axis=0)
  final_Y = np.concatenate(
      [np.zeros((per_class_examples)),
       np.ones((per_class_examples))]).astype('int64')

  P = np.random.permutation(len(final_Y))

  return final_X1[P], final_X2[P], final_Y[P]


def load_mnistcifar_helper(data, key1, key2, corr_frac):
  CIFAR_train_image = data[key1]['images']
  CIFAR_train_label = data[key1]['labels']
  MNIST_train_image = data[key2]['images']
  MNIST_train_label = data[key2]['labels']

  CIFAR_train_image, CIFAR_train_label = get_binary_datasets(
      CIFAR_train_image, CIFAR_train_label, FLAGS.CIFAR_label_1,
      FLAGS.CIFAR_label_2)
  MNIST_train_image, MNIST_train_label = get_binary_datasets(
      MNIST_train_image, MNIST_train_label, FLAGS.MNIST_label_1,
      FLAGS.MNIST_label_2)
  MNIST_train_image = make_MNIST_CIFAR_compatible(MNIST_train_image)
  if FLAGS.use_MNIST_labels:
    MNIST_train_image, CIFAR_train_image, train_labels = combine_datasets(
        MNIST_train_image, MNIST_train_label, CIFAR_train_image,
        CIFAR_train_label, corr_frac)
  else:
    CIFAR_train_image, MNIST_train_image, train_labels = combine_datasets(
        CIFAR_train_image, CIFAR_train_label, MNIST_train_image,
        MNIST_train_label, corr_frac)
  train_ds = tf.data.Dataset.from_tensor_slices({
      'CIFAR_image': CIFAR_train_image,
      'MNIST_image': MNIST_train_image,
      'label': train_labels
  })
  return train_ds, len(train_labels)

def get_bg_color_image(images, color_lower, color_upper):
  fg = np.zeros_like(images)
  fg[images != 0] = 255
  fg[images == 0] = 0
  fg = np.repeat(fg, 3, axis=-1)
  if images.shape[0] == 0:
    return fg
  bg = np.zeros_like(images)
  bg[images != 0] = 0
  bg[images == 0] = 1
  bg = np.repeat(bg, 3, axis=-1)
  final_color = np.zeros(np.concatenate([np.reshape(images.shape[0], [-1]), color_lower.shape]), dtype=np.uint8)
  for i in range(len(color_lower)):
    final_color[:, i] = np.random.choice(np.arange(color_lower[i], color_upper[i]+1), size=(images.shape[0]))
  final_color = np.reshape(final_color, (images.shape[0], 1, 1, -1))
  bg = bg * final_color
  return fg + bg

def color_MNIST_helper(images, labels, corr_frac, color_map_lower, color_map_upper, use_color_labels=False):
  final_images = None
  final_labels = None
  num_labels = int(np.amax(labels) + 1)
  for label in range(num_labels):
    temp_images = images[labels == label]
    r = np.random.uniform(size=temp_images.shape[0])
    temp_images_1 = temp_images[r < corr_frac]
    temp_images_2 = temp_images[r >= corr_frac]
    curr_label_images = get_bg_color_image(temp_images_1, color_map_lower[label], color_map_upper[label])
    temp = np.array([label])
    temp = np.repeat(temp, temp_images_1.shape[0], axis=0)
    if final_images is None:
      final_images = curr_label_images
      final_labels = temp
    else:
      final_images = np.concatenate([final_images, curr_label_images], axis=0)
      final_labels = np.concatenate([final_labels, temp], axis=0)
    r1 = np.random.choice(num_labels - 1, temp_images_2.shape[0])
    for i in range(len(r1)):
      if r1[i] >= label:
        r1[i] = r1[i] + 1
    for label2 in range(num_labels):
      if label2 == label:
        continue
      curr_images = temp_images_2[r1 == label2]
      curr_images = get_bg_color_image(curr_images, color_map_lower[label2], color_map_upper[label2])
      if use_color_labels:
        temp = np.array([label2])
      else:
        temp = np.array([label])
      temp = np.repeat(temp, curr_images.shape[0], axis=0)
      if final_images is None:
        final_images = curr_images
        final_labels = temp
      else:
        final_images = np.concatenate([final_images, curr_images], axis=0)
        final_labels = np.concatenate([final_labels, temp], axis=0)

  p = np.random.permutation(final_images.shape[0])  # pytype: disable=attribute-error
  final_images = final_images[p]
  final_labels = final_labels[p]
  return final_images, final_labels

def load_color_MNIST(path, binary, batched=True):
  COLOUR_MAP_LOWER = np.array([[195, 0, 0], [0, 195, 0], [0, 0, 195], [195, 195, 0],
                         [195, 0, 195], [0, 195, 195], [195, 98, 0],
                         [195, 0, 98], [98, 0, 195], [98, 98, 98]],
                        dtype=np.uint8)
  COLOUR_MAP_UPPER = np.array([[255, 60, 60], [60, 255, 60], [60, 60, 255], [255, 255, 60],
                         [255, 60, 255], [60, 255, 255], [255, 158, 60],
                         [255, 60, 158], [158, 60, 255], [158, 158, 158]],
                        dtype=np.uint8)

  with tf.io.gfile.GFile(path, 'rb') as fobj:
    data = pickle.load(fobj)

  if FLAGS.train_split:
    train_image = data['MNIST_train_split']['images']
    train_label = data['MNIST_train_split']['labels']
  else:
    train_image = data['MNIST_train']['images']
    train_label = data['MNIST_train']['labels']
  if binary:
    train_image, train_label = get_binary_datasets(train_image, train_label,
                                                   FLAGS.MNIST_label_1,
                                                   FLAGS.MNIST_label_2)
    COLOUR_MAP_LOWER = COLOUR_MAP_LOWER[0:2]
    COLOUR_MAP_UPPER = COLOUR_MAP_UPPER[0:2]
  train_image_in, train_label_in = color_MNIST_helper(train_image, train_label,
                                                      FLAGS.corr_frac,
                                                      COLOUR_MAP_LOWER, COLOUR_MAP_UPPER, FLAGS.use_color_labels)
  train_ds = tf.data.Dataset.from_tensor_slices({
      'image': train_image_in,
      'label': train_label_in
  })
  train_ds = train_ds.repeat(-1).map(mnist_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    train_ds = train_ds.batch(FLAGS.train_batch_size)

  val_image = data['MNIST_val']['images']
  val_label = data['MNIST_val']['labels']
  if binary:
    val_image, val_label = get_binary_datasets(val_image, val_label,
                                               FLAGS.MNIST_label_1,
                                               FLAGS.MNIST_label_2)
  val_image_in, val_label_in = color_MNIST_helper(val_image, val_label,
                                                  FLAGS.corr_frac, COLOUR_MAP_LOWER, COLOUR_MAP_UPPER, FLAGS.use_color_labels)
  val_ds = tf.data.Dataset.from_tensor_slices({
      'image': val_image_in,
      'label': val_label_in
  })
  val_ds = val_ds.map(mnist_transform).shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    if FLAGS.platform == 'TPU':
      val_ds = val_ds.batch(FLAGS.val_batch_size, drop_remainder=True)
    else:
      val_ds = val_ds.batch(FLAGS.val_batch_size)

  test_image = data['MNIST_test']['images']
  test_label = data['MNIST_test']['labels']
  if binary:
    test_image, test_label = get_binary_datasets(test_image, test_label,
                                                 FLAGS.MNIST_label_1,
                                                 FLAGS.MNIST_label_2)
  test_image_in, test_label_in = color_MNIST_helper(test_image, test_label,
                                                    FLAGS.corr_frac, COLOUR_MAP_LOWER, COLOUR_MAP_UPPER, FLAGS.use_color_labels)
  test_ds = tf.data.Dataset.from_tensor_slices({
      'image': test_image_in,
      'label': test_label_in
  })
  test_ds = test_ds.map(mnist_transform).shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    if FLAGS.platform == 'TPU':
      test_ds = test_ds.batch(FLAGS.test_batch_size, drop_remainder=True)
    else:
      test_ds = test_ds.batch(FLAGS.test_batch_size)

  if binary:
    train_image_out, train_label_out = color_MNIST_helper(
        train_image, train_label, 0.5, COLOUR_MAP_LOWER, COLOUR_MAP_UPPER)
  else:
    train_image_out, train_label_out = color_MNIST_helper(
        train_image, train_label, 0.1, COLOUR_MAP_LOWER, COLOUR_MAP_UPPER)
  OOD_ds_train = tf.data.Dataset.from_tensor_slices({
      'image': train_image_out,
      'label': train_label_out
  })
  OOD_ds_train = OOD_ds_train.repeat(-1).map(mnist_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    OOD_ds_train = OOD_ds_train.batch(FLAGS.train_batch_size)

  if binary:
    val_image_out, val_label_out = color_MNIST_helper(val_image, val_label, 0.5,
                                                      COLOUR_MAP_LOWER, COLOUR_MAP_UPPER)
  else:
    val_image_out, val_label_out = color_MNIST_helper(val_image, val_label, 0.1,
                                                      COLOUR_MAP_LOWER, COLOUR_MAP_UPPER)
  OOD_ds_val = tf.data.Dataset.from_tensor_slices({
      'image': val_image_out,
      'label': val_label_out
  })
  OOD_ds_val = OOD_ds_val.map(mnist_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    if FLAGS.platform == 'TPU':
      OOD_ds_val = OOD_ds_val.batch(FLAGS.val_batch_size, drop_remainder=True)
    else:
      OOD_ds_val = OOD_ds_val.batch(FLAGS.val_batch_size)

  if binary:
    test_image_out, test_label_out = color_MNIST_helper(test_image, test_label,
                                                        0.5, COLOUR_MAP_LOWER, COLOUR_MAP_UPPER)
  else:
    test_image_out, test_label_out = color_MNIST_helper(test_image, test_label,
                                                        0.1, COLOUR_MAP_LOWER, COLOUR_MAP_UPPER)
  OOD_ds_test = tf.data.Dataset.from_tensor_slices({
      'image': test_image_out,
      'label': test_label_out
  })
  OOD_ds_test = OOD_ds_test.map(mnist_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    if FLAGS.platform == 'TPU':
      OOD_ds_test = OOD_ds_test.batch(FLAGS.test_batch_size, drop_remainder=True)
    else:
      OOD_ds_test = OOD_ds_test.batch(FLAGS.test_batch_size)

  return train_ds, val_ds, test_ds, OOD_ds_train, OOD_ds_val, OOD_ds_test, len(
      train_label_in), len(train_label_out)

def imagenette_train_transform(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = tf.image.resize(x['image'], size=[256, 256])
    image = tf.image.random_crop(image, size=(224, 224, 3))
    image = tf.image.random_flip_left_right(image)
    image = image/255.0
    image = (image - mean)/std

    return {'image': image, 'label': x['label']}

def imagenette_val_transform(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = tf.image.resize(x['image'], size=[256,256])
    image = tf.image.central_crop(image, central_fraction=224.0/256.0)
    image = image/255.0
    image = (image - mean) / std

    return {'image': image, 'label': x['label']}

def load_imagenette(batched=True):
  ds = tfds.load('imagenette/full-size-v2')
  train_ds = ds['train']
  val_ds = ds['validation'].filter(lambda x: x['label']==0 or x['label']==1)
  # Taking a subset of two classes out of 10
  train_len = int(len(train_ds)/5)
  train_ds = train_ds.filter(lambda x: x['label']==0 or x['label']==1)
  train_ds = train_ds.repeat(-1).map(imagenette_train_transform).shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    train_ds = train_ds.batch(FLAGS.train_batch_size)

  val_ds = val_ds.map(imagenette_val_transform).shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    if FLAGS.platform == 'TPU':
      val_ds = val_ds.batch(FLAGS.val_batch_size, drop_remainder=True)
    else:
      val_ds = val_ds.batch(FLAGS.val_batch_size)

  #use validation as test
  test_ds = val_ds

  return train_ds, val_ds, test_ds, train_len

def load_mnistcifar(path, batched=True):
  with tf.io.gfile.GFile(path, 'rb') as fobj:
    data = pickle.load(fobj)

  if FLAGS.train_split:
    train_ds, train_len = load_mnistcifar_helper(data, 'CIFAR_train_split',
                                                 'MNIST_train_split',
                                                 FLAGS.corr_frac)
  else:
    train_ds, train_len = load_mnistcifar_helper(data, 'CIFAR_train',
                                                 'MNIST_train', FLAGS.corr_frac)
  train_ds = train_ds.repeat(-1).map(mnist_cifar_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    train_ds = train_ds.batch(FLAGS.train_batch_size)

  val_ds, _ = load_mnistcifar_helper(data, 'CIFAR_val', 'MNIST_val',
                                     FLAGS.corr_frac)
  val_ds = val_ds.map(mnist_cifar_transform).shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    val_ds = val_ds.batch(FLAGS.val_batch_size)

  test_ds, _ = load_mnistcifar_helper(data, 'CIFAR_test', 'MNIST_test',
                                      FLAGS.corr_frac)
  test_ds = test_ds.map(mnist_cifar_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    test_ds = test_ds.batch(FLAGS.test_batch_size)

  if FLAGS.train_split:
    OOD_train_ds, OOD_train_len = load_mnistcifar_helper(
        data, 'CIFAR_train_split', 'MNIST_train_split', 0.5)
  else:
    OOD_train_ds, OOD_train_len = load_mnistcifar_helper(
        data, 'CIFAR_train', 'MNIST_train', 0.5)
  OOD_train_ds = OOD_train_ds.repeat(-1).map(mnist_cifar_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    OOD_train_ds = OOD_train_ds.batch(FLAGS.train_batch_size)

  OOD_val_ds, _ = load_mnistcifar_helper(data, 'CIFAR_val', 'MNIST_val', 0.5)
  OOD_val_ds = OOD_val_ds.map(mnist_cifar_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    OOD_val_ds = OOD_val_ds.batch(FLAGS.val_batch_size)

  OOD_test_ds, _ = load_mnistcifar_helper(data, 'CIFAR_test', 'MNIST_test', 0.5)
  OOD_test_ds = OOD_test_ds.map(mnist_cifar_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    OOD_test_ds = OOD_test_ds.batch(FLAGS.test_batch_size)

  return train_ds, val_ds, test_ds, OOD_train_ds, OOD_val_ds, OOD_test_ds, train_len, OOD_train_len

def load_LMS(batched=True):
  NUM_TRAIN = 100000
  NUM_VAL = 10000
  NUM_TEST = 10000
  config = {
      'num_train': NUM_TRAIN,
      'dim': FLAGS.dataset_dim,
      'lin_margin': FLAGS.lin_margin,
      'slab_margin': FLAGS.slab_margin,
      'same_margin': False,
      'random_transform': FLAGS.use_random_transform,
      'width': 1,  # data width
      'bs': 256,
      'corrupt_lin': 0.0,
      'corrupt_lin_margin': False,
      'corrupt_slab': 0.0,
      'corrupt_slab7': 0.0,
      'num_test': NUM_VAL + NUM_TEST,
      'num_lin': FLAGS.num_lin,
      'num_slabs': FLAGS.num_5_slabs,
      'num_slabs7': FLAGS.num_7_slabs,
      'num_slabs3': FLAGS.num_3_slabs,
  }
  X, Y, W = get_lms_data(**config)
  tf.print('W:', W)
  if FLAGS.randomize_linear:
    X = X.dot(W.T)
    p = np.random.permutation(X.shape[0])
    X[:, 0] = X[p, 0]
    X = X.dot(W)
  elif FLAGS.randomize_slabs:
    X = X.dot(W.T)
    p = np.random.permutation(X.shape[0])
    X[:, 1:] = X[p, 1:]
    X = X.dot(W)
  save_obj = {'X': X, 'Y': Y, 'W': W}
  if not tf.io.gfile.exists(FLAGS.model_dir):
    tf.io.gfile.makedirs(FLAGS.model_dir)
  file_name = '{}{}'.format(FLAGS.model_dir, 'w.pkl')
  with tf.io.gfile.GFile(file_name, 'wb') as f:
    pickle.dump(save_obj, f)
  X_train = X[:NUM_TRAIN]
  Y_train = Y[:NUM_TRAIN]
  X_val = X[NUM_TRAIN:NUM_TRAIN + NUM_VAL]
  Y_val = Y[NUM_TRAIN:NUM_TRAIN + NUM_VAL]
  X_test = X[NUM_TRAIN + NUM_VAL:NUM_TRAIN + NUM_VAL + NUM_TEST]
  Y_test = Y[NUM_TRAIN + NUM_VAL:NUM_TRAIN + NUM_VAL + NUM_TEST]

  X_rand_train = X_train.dot(W.T)
  p = np.random.permutation(X_rand_train.shape[0])
  X_rand_train[:, 0] = X_rand_train[p, 0]
  X_rand_train = X_rand_train.dot(W)

  X_rand_val = X_val.dot(W.T)
  p = np.random.permutation(X_rand_val.shape[0])
  X_rand_val[:, 0] = X_rand_val[p, 0]
  X_rand_val = X_rand_val.dot(W)

  X_rand_test = X_test.dot(W.T)
  p = np.random.permutation(X_rand_test.shape[0])
  X_rand_test[:, 0] = X_rand_test[p, 0]
  X_rand_test = X_rand_test.dot(W)

  train_ds = tf.data.Dataset.from_tensor_slices({
      'image': X_train,
      'label': Y_train
  })
  train_ds = train_ds.repeat(-1).shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    train_ds = train_ds.batch(FLAGS.train_batch_size)

  val_ds = tf.data.Dataset.from_tensor_slices({'image': X_val, 'label': Y_val})
  val_ds = val_ds.shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    val_ds = val_ds.batch(FLAGS.val_batch_size)

  test_ds = tf.data.Dataset.from_tensor_slices({
      'image': X_test,
      'label': Y_test
  })
  test_ds = test_ds.shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    test_ds = test_ds.batch(FLAGS.test_batch_size)

  OOD_train_ds = tf.data.Dataset.from_tensor_slices({
      'image': X_rand_train,
      'label': Y_train
  })
  OOD_train_ds = OOD_train_ds.repeat(-1).shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    OOD_train_ds = OOD_train_ds.batch(FLAGS.train_batch_size)

  OOD_val_ds = tf.data.Dataset.from_tensor_slices({'image': X_rand_val, 'label': Y_val})
  OOD_val_ds = OOD_val_ds.shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    OOD_val_ds = OOD_val_ds.batch(FLAGS.val_batch_size)

  OOD_test_ds = tf.data.Dataset.from_tensor_slices({
      'image': X_rand_test,
      'label': Y_test
  })
  OOD_test_ds = OOD_test_ds.shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    OOD_test_ds = OOD_test_ds.batch(FLAGS.test_batch_size)

  return train_ds, val_ds, test_ds, OOD_train_ds, OOD_val_ds, OOD_test_ds, NUM_TRAIN, NUM_TRAIN, tf.constant(W, dtype=tf.float64)

# def load_2D_dataset(batched=True):
#   NUM_TRAIN = 100000
#   NUM_VAL = 10000
#   NUM_TEST = 10000

def load_mnist(path, batched=True):
  with tf.io.gfile.GFile(path, 'rb') as fobj:
    data = pickle.load(fobj)

  if FLAGS.train_split:
    train_image = data['MNIST_train_split']['images']
    train_label = data['MNIST_train_split']['labels']
  else:
    train_image = data['MNIST_train']['images']
    train_label = data['MNIST_train']['labels']

  train_ds = tf.data.Dataset.from_tensor_slices({
      'image': train_image,
      'label': train_label
  })
  train_ds = train_ds.repeat(-1).map(mnist_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    train_ds = train_ds.batch(FLAGS.train_batch_size)

  val_image = data['MNIST_val']['images']
  val_label = data['MNIST_val']['labels']
  val_ds = tf.data.Dataset.from_tensor_slices({
      'image': val_image,
      'label': val_label
  })
  val_ds = val_ds.map(mnist_transform).shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    val_ds = val_ds.batch(FLAGS.val_batch_size)

  test_image = data['MNIST_test']['images']
  test_label = data['MNIST_test']['labels']
  test_ds = tf.data.Dataset.from_tensor_slices({
      'image': test_image,
      'label': test_label
  })
  test_ds = test_ds.map(mnist_transform).shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    test_ds = test_ds.batch(FLAGS.test_batch_size)

  return train_ds, val_ds, test_ds, len(train_label)


def load_binary_mnist(path, batched=True):
  with tf.io.gfile.GFile(path, 'rb') as fobj:
    data = pickle.load(fobj)

  if FLAGS.train_split:
    train_image = data['MNIST_train_split']['images']
    train_label = data['MNIST_train_split']['labels']
  else:
    train_image = data['MNIST_train']['images']
    train_label = data['MNIST_train']['labels']
  train_image, train_label = get_binary_datasets(train_image, train_label,
                                                 FLAGS.MNIST_label_1,
                                                 FLAGS.MNIST_label_2)
  train_ds = tf.data.Dataset.from_tensor_slices({
      'image': train_image,
      'label': train_label
  })
  train_ds = train_ds.repeat(-1).map(mnist_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    train_ds = train_ds.batch(FLAGS.train_batch_size)

  val_image = data['MNIST_val']['images']
  val_label = data['MNIST_val']['labels']
  val_image, val_label = get_binary_datasets(val_image, val_label,
                                             FLAGS.MNIST_label_1,
                                             FLAGS.MNIST_label_2)
  val_ds = tf.data.Dataset.from_tensor_slices({
      'image': val_image,
      'label': val_label
  })
  val_ds = val_ds.map(mnist_transform).shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    val_ds = val_ds.batch(FLAGS.val_batch_size)

  test_image = data['MNIST_test']['images']
  test_label = data['MNIST_test']['labels']
  test_image, test_label = get_binary_datasets(test_image, test_label,
                                               FLAGS.MNIST_label_1,
                                               FLAGS.MNIST_label_2)
  test_ds = tf.data.Dataset.from_tensor_slices({
      'image': test_image,
      'label': test_label
  })
  test_ds = test_ds.map(mnist_transform).shuffle(_SHUFFLE_BUFFER_SIZE.value)
  if batched:
    test_ds = test_ds.batch(FLAGS.test_batch_size)

  return train_ds, val_ds, test_ds, len(train_label)


def load_cinic10(path, batched=True):
  with tf.io.gfile.GFile(path, 'rb') as fobj:
    data = pickle.load(fobj)

  if FLAGS.train_split:
    train_image = data['train_split']['images']
    train_label = data['train_split']['labels']
  else:
    train_image = data['train']['images']
    train_label = data['train']['labels']

  train_ds = tf.data.Dataset.from_tensor_slices({
      'image': train_image,
      'label': train_label
  })
  train_ds = train_ds.repeat(-1).map(cifar10_train_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    train_ds = train_ds.batch(FLAGS.train_batch_size)

  val_image = data['val']['images']
  val_label = data['val']['labels']
  val_ds = tf.data.Dataset.from_tensor_slices({
      'image': val_image,
      'label': val_label
  })
  val_ds = val_ds.map(cifar10_test_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    val_ds = val_ds.batch(FLAGS.val_batch_size)

  test_image = data['test']['images']
  test_label = data['test']['labels']
  test_ds = tf.data.Dataset.from_tensor_slices({
      'image': test_image,
      'label': test_label
  })
  test_ds = test_ds.map(cifar10_test_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    test_ds = test_ds.batch(FLAGS.test_batch_size)

  OOD_train_image = data['OOD_train']['images']
  OOD_train_label = data['OOD_train']['labels']
  OOD_ds_train = tf.data.Dataset.from_tensor_slices({
      'image': OOD_train_image,
      'label': OOD_train_label
  })
  if FLAGS.use_OOD_transform:
    OOD_ds_train = OOD_ds_train.repeat(-1).map(cinic_train_transform).shuffle(
        _SHUFFLE_BUFFER_SIZE.value)
  else:
    OOD_ds_train = OOD_ds_train.repeat(-1).map(cifar10_train_transform).shuffle(
        _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    OOD_ds_train = OOD_ds_train.batch(FLAGS.train_batch_size)

  OOD_val_image = data['OOD_val']['images']
  OOD_val_label = data['OOD_val']['labels']
  OOD_ds_val = tf.data.Dataset.from_tensor_slices({
      'image': OOD_val_image,
      'label': OOD_val_label
  })
  if FLAGS.use_OOD_transform:
    OOD_ds_val = OOD_ds_val.map(cinic_test_transform).shuffle(
        _SHUFFLE_BUFFER_SIZE.value)
  else:
    OOD_ds_val = OOD_ds_val.map(cifar10_test_transform).shuffle(
        _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    OOD_ds_val = OOD_ds_val.batch(FLAGS.test_batch_size)

  OOD_test_image = data['OOD_test']['images']
  OOD_test_label = data['OOD_test']['labels']
  OOD_ds_test = tf.data.Dataset.from_tensor_slices({
      'image': OOD_test_image,
      'label': OOD_test_label
  })
  if FLAGS.use_OOD_transform:
    OOD_ds_test = OOD_ds_test.map(cinic_test_transform).shuffle(
        _SHUFFLE_BUFFER_SIZE.value)
  else:
    OOD_ds_test = OOD_ds_test.map(cifar10_test_transform).shuffle(
        _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    OOD_ds_test = OOD_ds_test.batch(FLAGS.test_batch_size)

  return train_ds, val_ds, test_ds, OOD_ds_train, OOD_ds_val, OOD_ds_test, len(
      train_label), len(OOD_train_label)


def load_cifar10(path, batched=True):
  with tf.io.gfile.GFile(path, 'rb') as fobj:
    data = pickle.load(fobj)

  if FLAGS.train_split:
    train_image = data['train_split']['images']
    train_label = data['train_split']['labels']
  else:
    train_image = data['train']['images']
    train_label = data['train']['labels']

  train_ds = tf.data.Dataset.from_tensor_slices({
      'image': train_image,
      'label': train_label
  })
  train_ds = train_ds.repeat(-1).map(cifar10_train_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    train_ds = train_ds.batch(FLAGS.train_batch_size)

  val_image = data['val']['images']
  val_label = data['val']['labels']
  val_ds = tf.data.Dataset.from_tensor_slices({
      'image': val_image,
      'label': val_label
  })
  val_ds = val_ds.map(cifar10_test_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    val_ds = val_ds.batch(FLAGS.val_batch_size)

  test_image = data['test']['images']
  test_label = data['test']['labels']
  test_ds = tf.data.Dataset.from_tensor_slices({
      'image': test_image,
      'label': test_label
  })
  test_ds = test_ds.map(cifar10_test_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    test_ds = test_ds.batch(FLAGS.test_batch_size)

  OOD_train_image = data['OOD_train']['images']
  OOD_train_label = data['OOD_train']['labels']
  OOD_ds_train = tf.data.Dataset.from_tensor_slices({
      'image': OOD_train_image,
      'label': OOD_train_label
  })
  if FLAGS.use_OOD_transform:
    OOD_ds_train = OOD_ds_train.repeat(-1).map(
        cifar102_train_transform).shuffle(_SHUFFLE_BUFFER_SIZE.value)
  else:
    OOD_ds_train = OOD_ds_train.repeat(-1).map(cifar10_train_transform).shuffle(
        _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    OOD_ds_train = OOD_ds_train.batch(FLAGS.train_batch_size)

  OOD_test_image = data['OOD_test']['images']
  OOD_test_label = data['OOD_test']['labels']
  OOD_ds_test = tf.data.Dataset.from_tensor_slices({
      'image': OOD_test_image,
      'label': OOD_test_label
  })
  if FLAGS.use_OOD_transform:
    OOD_ds_test = OOD_ds_test.map(cifar102_test_transform).shuffle(
        _SHUFFLE_BUFFER_SIZE.value)
  else:
    OOD_ds_test = OOD_ds_test.map(cifar10_test_transform).shuffle(
        _SHUFFLE_BUFFER_SIZE.value)
  if batched:
    OOD_ds_test = OOD_ds_test.batch(FLAGS.test_batch_size)

  return train_ds, val_ds, test_ds, OOD_ds_train, OOD_ds_test, len(
      train_label), len(OOD_train_label)

def load_waterbirds_reps(path):
  def _parse_function(example_proto):
    feature_desc = {
        'inputs': tf.io.FixedLenFeature([2048], tf.float32),
        'labels': tf.io.FixedLenFeature([], tf.int64),
        'metadata': tf.io.FixedLenFeature([2], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_desc)
    return parsed_features

  filenames = []
  if FLAGS.use_complete_corr:
    filepath = os.path.join(path, 'train.tfrecords')
  else:
    filepath = os.path.join(path, 'train-all.tfrecords')
  filenames.append(filepath)
  raw_train_ds = tf.data.TFRecordDataset(filenames)
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
  raw_train_ds = raw_train_ds.with_options(options)
  train_ds = raw_train_ds.map(_parse_function)
  train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
  train_ds = train_ds.repeat(-1).map(waterbirds_test_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value).batch(FLAGS.train_batch_size)

  filenames = []
  filepath = os.path.join(path, 'val-all.tfrecords')
  filenames.append(filepath)
  raw_val_ds = tf.data.TFRecordDataset(filenames)
  raw_val_ds = raw_val_ds.with_options(options)
  val_ds = raw_val_ds.map(_parse_function)
  val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
  val_ds = val_ds.map(waterbirds_test_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if FLAGS.platform == 'TPU':
    val_ds = val_ds.batch(FLAGS.val_batch_size, drop_remainder=True)
  else:
    val_ds = val_ds.batch(FLAGS.val_batch_size)

  filenames = []
  filepath = os.path.join(path, 'test-all.tfrecords')
  filenames.append(filepath)
  raw_test_ds = tf.data.TFRecordDataset(filenames)
  raw_test_ds = raw_test_ds.with_options(options)
  test_ds = raw_test_ds.map(_parse_function)
  test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
  test_ds = test_ds.map(waterbirds_test_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if FLAGS.platform == 'TPU':
    test_ds = test_ds.batch(FLAGS.test_batch_size, drop_remainder=True)
  else:
    test_ds = test_ds.batch(FLAGS.test_batch_size)

  if FLAGS.use_complete_corr:
    return train_ds, val_ds, test_ds, 4555
  else:
    return train_ds, val_ds, test_ds, 4795

def load_waterbirds(path):

  def _parse_function(example_proto):
    feature_desc = {
        'inputs': tf.io.FixedLenFeature([3,224,224], tf.float32),
        'labels': tf.io.FixedLenFeature([], tf.int64),
        'metadata': tf.io.FixedLenFeature([2], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_desc)
    return parsed_features

  filenames = []
  for i in range(1):
    if FLAGS.use_equal_split:
      filepath = os.path.join(path, 'waterbirds_equal_split.tfrecords')
    elif FLAGS.use_complete_corr:
      filepath = os.path.join(path, 'waterbirds_train_{}_complete_corr.tfrecords'.format(str(i)))
    else:
      filepath = os.path.join(path, 'waterbirds_train_{}.tfrecords'.format(str(i)))
    filenames.append(filepath)
  # filepath = os.path.join(path, 'waterbirds_train.tfrecords')
  # filenames = [filepath]
  raw_train_ds = tf.data.TFRecordDataset(filenames)
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
  raw_train_ds = raw_train_ds.with_options(options)
  train_ds = raw_train_ds.map(_parse_function)
  # filepath = os.path.join(path, '{}.pkl'.format(split))
#   with tf.io.gfile.GFile(path, 'rb') as fobj:
#     data = pickle.load(fobj)

#   train_image = data['train']['inputs'].transpose(0, 2, 3,
#                                                   1)  #Convert NCHW to NHWC
#   train_label = data['train']['labels']
#   train_metadata = data['train']['metadata']
#   if FLAGS.use_complete_corr:
#     train_image = train_image[train_metadata[:,0]==train_metadata[:,1]]
#     train_label = train_label[train_metadata[:,0]==train_metadata[:,1]]
#     train_metadata = train_metadata[train_metadata[:,0]==train_metadata[:,1]]
#   def gen_train():
#     for i in range(len(train_image)):
#       l = {}
#       l['image'] = train_image[i]
#       l['label'] = train_label[i]
#       l['metadata'] = train_metadata[i]
#       yield l
#   train_ds = tf.data.Dataset.from_generator(gen_train,
#                                             output_signature={'image': tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
#                                                               'label': tf.TensorSpec(shape=(), dtype=tf.int64),
#                                                               'metadata': tf.TensorSpec(shape=(None), dtype=tf.int64)})
  # train_ds = tf.data.Dataset.from_tensor_slices({
  #     'image': train_image,
  #     'label': train_label,
  #     'metadata': train_metadata
  # })
  train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
  if FLAGS.use_data_aug_with_DRO:
    train_ds = train_ds.repeat(-1).map(waterbirds_train_transform).shuffle(
        _SHUFFLE_BUFFER_SIZE.value).batch(FLAGS.train_batch_size)
  else:
    train_ds = train_ds.repeat(-1).map(waterbirds_test_transform).shuffle(
        _SHUFFLE_BUFFER_SIZE.value).batch(FLAGS.train_batch_size)

  # val_image = data['val']['inputs'].transpose(0, 2, 3, 1)  #Convert NCHW to NHWC
  # val_label = data['val']['labels']
  # val_metadata = data['val']['metadata']
  # def gen_val():
  #   for i in range(len(val_image)):
  #     l = {}
  #     l['image'] = val_image[i]
  #     l['label'] = val_label[i]
  #     l['metadata'] = val_metadata[i]
  #     yield l
  # val_ds = tf.data.Dataset.from_generator(gen_val,
  #                                           output_signature={'image': tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
  #                                                             'label': tf.TensorSpec(shape=(), dtype=tf.int64),
  #                                                             'metadata': tf.TensorSpec(shape=(None), dtype=tf.int64)})
  # filepath = os.path.join(path, 'waterbirds_val.tfrecords')
  # filenames = [filepath]
  filenames = []
  for i in range(1):
    if FLAGS.use_complete_corr_test:
      filepath = os.path.join(path, 'waterbirds_val_{}_complete_corr.tfrecords'.format(str(i)))
    else:
      filepath = os.path.join(path, 'waterbirds_val_{}.tfrecords'.format(str(i)))
    filenames.append(filepath)
  raw_val_ds = tf.data.TFRecordDataset(filenames)
  raw_val_ds = raw_val_ds.with_options(options)
  val_ds = raw_val_ds.map(_parse_function)
  val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
  # val_ds = tf.data.Dataset.from_tensor_slices({
  #     'image': val_image,
  #     'label': val_label,
  #     'metadata': val_metadata
  # })
  val_ds = val_ds.map(waterbirds_test_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if FLAGS.platform == 'TPU':
    val_ds = val_ds.batch(FLAGS.val_batch_size, drop_remainder=True)
  else:
    val_ds = val_ds.batch(FLAGS.val_batch_size)

  # test_image = data['test']['inputs'].transpose(0, 2, 3,
  #                                               1)  #Convert NCHW to NHWC
  # test_label = data['test']['labels']
  # test_metadata = data['test']['metadata']
  # def gen_test():
  #   for i in range(len(test_image)):
  #     l = {}
  #     l['image'] = test_image[i]
  #     l['label'] = test_label[i]
  #     l['metadata'] = test_metadata[i]
  #     yield l
  # test_ds = tf.data.Dataset.from_generator(gen_test,
  #                                           output_signature={'image': tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
  #                                                             'label': tf.TensorSpec(shape=(), dtype=tf.int64),
  #                                                             'metadata': tf.TensorSpec(shape=(None), dtype=tf.int64)})
  # filepath = os.path.join(path, 'waterbirds_test.tfrecords')
  # filenames = [filepath]
  filenames = []
  for i in range(1):
    if FLAGS.use_complete_corr_test:
      filepath = os.path.join(path, 'waterbirds_test_{}_complete_corr.tfrecords'.format(str(i)))
    else:
      filepath = os.path.join(path, 'waterbirds_test_{}.tfrecords'.format(str(i)))
    filenames.append(filepath)
  raw_test_ds = tf.data.TFRecordDataset(filenames)
  raw_test_ds = raw_test_ds.with_options(options)
  test_ds = raw_test_ds.map(_parse_function)
  test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
  # test_ds = tf.data.Dataset.from_tensor_slices({
  #     'image': test_image,
  #     'label': test_label,
  #     'metadata': test_metadata
  # })
  test_ds = test_ds.map(waterbirds_test_transform).shuffle(
      _SHUFFLE_BUFFER_SIZE.value)
  if FLAGS.platform == 'TPU':
    test_ds = test_ds.batch(FLAGS.test_batch_size, drop_remainder=True)
  else:
    test_ds = test_ds.batch(FLAGS.test_batch_size)

  if FLAGS.use_complete_corr:
    return train_ds, val_ds, test_ds, 4555
  else:
    return train_ds, val_ds, test_ds, 4795


def load_data(dataset: str,
              desired_classes,
              num_train: int = -1,
              num_test: int = -1,
              frac_poison: float = 0.,
              testset_type: int = 1,
              path: str = '',
              batched=True):
  """Takes dataset and other options as inputs and returns the dataset loader.

  Args:
    dataset : string of dataset name
    desired_classes : list of classes that need to be included
    num_train : number of training examples
    num_test : number of test examples
    frac_poison : fraction of examples that are poisoned in train data
    testset_type : Type of test set to be returned

  Returns:
    train_dataset and test_dataset objects.
  """
  if dataset == 'cifar10':
    ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_test, train_length, OOD_train_length = load_cifar10(
        path, batched=batched)
    if batched:
      strategy = tf.distribute.get_strategy()
      ds_train = strategy.experimental_distribute_dataset(ds_train)
      ds_val = strategy.experimental_distribute_dataset(ds_val)
      ds_test = strategy.experimental_distribute_dataset(ds_test)
      ds_OOD_train = strategy.experimental_distribute_dataset(ds_OOD_train)
      ds_OOD_test = strategy.experimental_distribute_dataset(ds_OOD_test)
      return ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_test, train_length, OOD_train_length
    else:
      return ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_test, train_length, OOD_train_length
  elif dataset == 'cinic10':
    ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_length, OOD_train_length = load_cinic10(
        path, batched=batched)
    if batched:
      strategy = tf.distribute.get_strategy()
      ds_train = strategy.experimental_distribute_dataset(ds_train)
      ds_val = strategy.experimental_distribute_dataset(ds_val)
      ds_test = strategy.experimental_distribute_dataset(ds_test)
      ds_OOD_train = strategy.experimental_distribute_dataset(ds_OOD_train)
      ds_OOD_val = strategy.experimental_distribute_dataset(ds_OOD_val)
      ds_OOD_test = strategy.experimental_distribute_dataset(ds_OOD_test)
      return ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_length, OOD_train_length
    else:
      return ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_length, OOD_train_length
  elif dataset == 'binary-color-mnist' or dataset == 'color-mnist':
    if dataset == 'color-mnist':
      ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_length, OOD_train_length = load_color_MNIST(
          path, binary=False, batched=batched)
    else:
      ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_length, OOD_train_length = load_color_MNIST(
          path, binary=True, batched=batched)
    if batched:
      strategy = tf.distribute.get_strategy()
      ds_train = strategy.experimental_distribute_dataset(ds_train)
      ds_val = strategy.experimental_distribute_dataset(ds_val)
      ds_test = strategy.experimental_distribute_dataset(ds_test)
      ds_OOD_train = strategy.experimental_distribute_dataset(ds_OOD_train)
      ds_OOD_val = strategy.experimental_distribute_dataset(ds_OOD_val)
      ds_OOD_test = strategy.experimental_distribute_dataset(ds_OOD_test)
      return ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_length, OOD_train_length
    else:
      return ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_length, OOD_train_length
  elif dataset == 'mnist-cifar':
    ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_length, OOD_train_length = load_mnistcifar(
        path, batched=batched)
    if batched:
      strategy = tf.distribute.get_strategy()
      ds_train = strategy.experimental_distribute_dataset(ds_train)
      ds_val = strategy.experimental_distribute_dataset(ds_val)
      ds_test = strategy.experimental_distribute_dataset(ds_test)
      ds_OOD_train = strategy.experimental_distribute_dataset(ds_OOD_train)
      ds_OOD_val = strategy.experimental_distribute_dataset(ds_OOD_val)
      ds_OOD_test = strategy.experimental_distribute_dataset(ds_OOD_test)
      return ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_length, OOD_train_length
    else:
      return ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_length, OOD_train_length
  elif dataset == 'mnist':
    ds_train, ds_val, ds_test, train_length = load_mnist(path, batched=batched)
    if batched:
      strategy = tf.distribute.get_strategy()
      ds_train = strategy.experimental_distribute_dataset(ds_train)
      ds_val = strategy.experimental_distribute_dataset(ds_val)
      ds_test = strategy.experimental_distribute_dataset(ds_test)
      return ds_train, ds_val, ds_test, train_length
    else:
      return ds_train, ds_val, ds_test, train_length
  elif dataset == 'binary-mnist':
    ds_train, ds_val, ds_test, train_length = load_binary_mnist(
        path, batched=batched)
    if batched:
      strategy = tf.distribute.get_strategy()
      ds_train = strategy.experimental_distribute_dataset(ds_train)
      ds_val = strategy.experimental_distribute_dataset(ds_val)
      ds_test = strategy.experimental_distribute_dataset(ds_test)
      return ds_train, ds_val, ds_test, train_length
    else:
      return ds_train, ds_val, ds_test, train_length
  elif dataset == 'waterbirds':
    if FLAGS.check_torch_reps:
      ds_train, ds_val, ds_test, train_length = load_waterbirds_reps(path)
    else:
      ds_train, ds_val, ds_test, train_length = load_waterbirds(path)
    strategy = tf.distribute.get_strategy()
    ds_train = strategy.experimental_distribute_dataset(ds_train)
    ds_val = strategy.experimental_distribute_dataset(ds_val)
    ds_test = strategy.experimental_distribute_dataset(ds_test)
    return ds_train, ds_val, ds_test, train_length
  elif dataset == 'lms':
    ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_length, OOD_train_length, W = load_LMS(
        batched=batched)
    if batched:
      strategy = tf.distribute.get_strategy()
      ds_train = strategy.experimental_distribute_dataset(ds_train)
      ds_val = strategy.experimental_distribute_dataset(ds_val)
      ds_test = strategy.experimental_distribute_dataset(ds_test)
      ds_OOD_train = strategy.experimental_distribute_dataset(ds_OOD_train)
      ds_OOD_val = strategy.experimental_distribute_dataset(ds_OOD_val)
      ds_OOD_test = strategy.experimental_distribute_dataset(ds_OOD_test)
      return ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_length, OOD_train_length, W
    else:
      return ds_train, ds_val, ds_test, ds_OOD_train, ds_OOD_val, ds_OOD_test, train_length, OOD_train_length, W
  elif dataset == 'imagenette':
    ds_train, ds_val, ds_test, train_length = load_imagenette(batched=batched)
    if batched:
      strategy = tf.distribute.get_strategy()
      ds_train = strategy.experimental_distribute_dataset(ds_train)
      ds_val = strategy.experimental_distribute_dataset(ds_val)
      ds_test = strategy.experimental_distribute_dataset(ds_test)
      return ds_train, ds_val, ds_test, train_length
    else:
      return ds_train, ds_val, ds_test, train_length
  else:
    raise ValueError(f'Invalid dataset {dataset}')
  if (frac_poison > 0.) and (len(desired_classes) != 2):
    raise ValueError('Poisoning supported only on binary classification.')
  # Ensure desired_classes does not have any repeated elements
  desired_classes_set = set(desired_classes)
  if len(desired_classes_set) != len(desired_classes):
    raise ValueError(f'Invalid desired classes {desired_classes}')
  (ds_train, ds_test), ds_info = tfds.load(  # pylint: disable=unused-variable
      dataset,
      split=['train', 'test'],
      batch_size=-1,
      shuffle_files=True,
      with_info=True)
  ds_train = tfds.as_numpy(ds_train)
  ds_test = tfds.as_numpy(ds_test)
  x_train = ds_train['image']
  y_train = ds_train['label']
  x_test = ds_test['image']
  y_test = ds_test['label']
  # Map desired_classes to begin from 0
  desired_classes_mapping = {}
  for ind, item in enumerate(desired_classes_set):
    desired_classes_mapping[item] = ind
  new_x_train = []
  new_y_train = []
  for ind, label in enumerate(y_train):
    if label in desired_classes_set:
      new_x_train.append(x_train[ind])
      new_y_train.append(desired_classes_mapping[label])
  if num_train == -1:
    train_length = len(new_x_train)
  else:
    train_length = num_train
  new_x_test = []
  new_y_test = []
  for ind, label in enumerate(y_test):
    if label in desired_classes_set:
      new_x_test.append(x_test[ind])
      new_y_test.append(desired_classes_mapping[label])
  # Normalize train and test dataset
  new_x_train, mean_arr, std_arr = normalize_train_data(
      np.asarray(new_x_train) / max_val)
  new_x_test = normalize_test_data(
      np.asarray(new_x_test) / max_val, mean_arr, std_arr)
  if frac_poison > 0.:
    new_x_train_poisoned = create_patch(
        new_x_train, new_y_train, patch_size=_PATCH_SIZE.value, flip=False)
    if testset_type == 2:
      new_x_test_poisoned = create_patch(
          new_x_test, new_y_test, patch_size=_PATCH_SIZE.value, flip=True)
    elif testset_type == 1:
      new_x_test_poisoned = new_x_test
    else:
      raise ValueError('Invalid arguments testset_type: Testset_type takes only values 1 or 2 ')  # pylint: disable=line-too-long
    # perturbed Train data
    p = np.random.permutation(len(new_x_train))
    new_img_arr = []
    new_label_arr = []
    for i, ind in enumerate(p):
      if i < frac_poison * len(new_x_train):
        new_img_arr.append(new_x_train[ind])
      else:
        new_img_arr.append(new_x_train_poisoned[ind])
      new_label_arr.append(new_y_train[ind])
    new_x_train = new_img_arr
    new_y_train = new_label_arr
    # perturbed Test data
    new_x_test = new_x_test_poisoned
  # new_x_train = normalize_data(
  #     np.asarray(new_x_train) / max_val, mean_arr, std_arr)
  # new_x_test = normalize_data(
  #     np.asarray(new_x_test) / max_val, mean_arr, std_arr)
  new_x_train = tf.convert_to_tensor(new_x_train)
  new_y_train = tf.convert_to_tensor(np.asarray(new_y_train))
  ds_train = tf.data.Dataset.from_tensor_slices({
      'image': new_x_train,
      'label': new_y_train
  }).shuffle(buffer_size=_SHUFFLE_BUFFER_SIZE.value).take(num_train)
  if dataset == 'cifar10':
    ds_train = ds_train.repeat(-1).map(perturb_train).shuffle(
        buffer_size=_SHUFFLE_BUFFER_SIZE.value).batch(
            batch_size=FLAGS.train_batch_size)
  else:
    ds_train = ds_train.repeat(-1).shuffle(
        buffer_size=_SHUFFLE_BUFFER_SIZE.value).batch(
            batch_size=FLAGS.train_batch_size)
  new_x_test = tf.convert_to_tensor(new_x_test)
  new_y_test = tf.convert_to_tensor(np.asarray(new_y_test))
  ds_test = tf.data.Dataset.from_tensor_slices({
      'image': new_x_test,
      'label': new_y_test
  }).shuffle(buffer_size=_SHUFFLE_BUFFER_SIZE.value).take(num_test).batch(
      batch_size=FLAGS.eval_batch_size)

  # Don't think we should perturb test
  # ds_test = perturb_test(ds_test)
  strategy = tf.distribute.get_strategy()
  ds_train = strategy.experimental_distribute_dataset(ds_train)
  ds_test = strategy.experimental_distribute_dataset(ds_test)
  return ds_train, ds_test, train_length
