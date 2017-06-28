# module to load a series of 16 bit grayscale images of people 

import numpy as np
import random
import tensorflow as tf


def read_pair(filename_queue):
  reader = tf.TFRecordReader()
  # One can read a single serialized example from a filename
  # serialized_example is a Tensor of type string.
  _, serialized_example = reader.read(filename_queue)
  # The serialized example is converted back to actual values.
  # One needs to describe the format of the objects to be returned
  features = tf.parse_single_example(
      serialized_example,
      features={
          # We know the length of both fields. If not the
          # tf.VarLenFeature could be used
          'label': tf.FixedLenFeature([2], tf.int64),
          'image': tf.VarLenFeature( tf.float),
          'rows' : tf.FixedLenFeature([], tf.int64),
          'cols' : tf.FixedLenFeature([], tf.int64)
      })
  label = features['label']
  image_1 = features['image_1']
  image_2 = features['image_2']


#from the top, let's make it a FIFO queue loading files from a folder

def input_pipeline(filenames, batch_size, read_threads, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example_list = [read_pair(filename_queue)
                  for _ in range(read_threads)]
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example1_batch, example2_batch, label_batch = tf.train.shuffle_batch_join(
      example_list, batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example1_batch, example2_batch, label_batch
