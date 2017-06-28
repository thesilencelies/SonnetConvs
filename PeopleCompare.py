# module to load a series of 16 bit grayscale images of people 

# based on mnist.py
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for loading and reading image data then using it for comparisons.
   Returned dataset are the flattened images concatenated, with labels true (1) or false (0) 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import random


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class DataSet(object):

  def __init__(self,
               images,
               labels,
               dtype=dtypes.float32,
               reshape=False):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
   
    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
    if dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def generate_pairs_foreach(images, labels, positive_ratio = 0.5):
   """generates a partner for each image in the input and returns
      the pairs and whether they have the same label or not
   """
   #this will be computationally slow - maybe serialise the results?
   size = images.shape[0]
   npos = positive_ratio * size
   nneg = size - npos
   perm = numpy.arange(size)
   numpy.random.shuffle(perm)
   pairimgs = []
   slabels = []
   it = 0
   for i in range(0,size):
      # decide if it's positive or not
      positive = (npos/(npos + nneg)) < random.random()
      if positive :
         npos-=1
      else :
         nneg-=1
      # iterate through the random set until you hit an example of the correct type
      while True :
         if (labels[it] == labels[i]) == positive :
            pairimgs.append(it)
            if positive :
               slabels.append([1,0])
            else :
               slabels.append([0,1])
            it = (it + 1) % size
            break
         it = (it + 1) % size
   resimgs = numpy.concatenate([images, images[pairimgs]],1)
   reslabels = numpy.array(slabels)
   return resimgs,reslabels



def read_data_sets(imagesfile,
                   labelsfile,
                   fake_data=False,
                   dtype=dtypes.float32,
                   reshape=False,
                   validation_size=5000):
  if fake_data:

    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  images = numpy.load(imagesfile).astype(float)
  labels = numpy.load(labelsfile).astype(float)
  

  #the training set is the first 70%, the test set is a random subsample of the whole
  train_images = images[0:int(images.shape[0]*0.7)]
  train_labels= labels[0:(images.shape[0]*0.7)]
  perm = numpy.arange(images.shape[0])
  numpy.random.shuffle(perm)
  perm = perm[0:int(perm.shape[0]*0.4)]
  test_images = images[perm]
  test_labels = labels[perm]


  #flatten the inputs
  test_images = test_images.reshape(test_images.shape[0],
                                test_images.shape[1] * test_images.shape[2])
  train_images = train_images.reshape(train_images.shape[0],
                                train_images.shape[1] * train_images.shape[2])

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  #generate the pairs
  train_pairs, train_pair_labels = generate_pairs_foreach(train_images, train_labels)
  validation_pairs, validation_pair_labels = generate_pairs_foreach(validation_images, validation_labels)
  test_pairs, test_pair_labels = generate_pairs_foreach(test_images, test_labels)
   

  train = DataSet(train_pairs, train_pair_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_pairs,
                       validation_pair_labels,
                       dtype=dtype,
                       reshape=reshape)
  test = DataSet(test_pairs, test_pair_labels, dtype=dtype, reshape=reshape)

  return base.Datasets(train=train, validation=validation, test=test)


def load_people_compare(imagefile="spinppl.dat", labelfile="spinlables.dat"):
  return read_data_sets(imagefile,labelfile)
