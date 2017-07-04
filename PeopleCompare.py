# module to load a series of 16 bit grayscale images of people 

import numpy as np
import random
import scipy.ndimage
import tensorflow as tf


def read_pair(filename_queue, width = 200, height = 290):
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
          'image_1': tf.FixedLenFeature([1,height,width,1], tf.float32),
          'image_2': tf.FixedLenFeature([1,height,width,1], tf.float32),
          'rows' : tf.FixedLenFeature([], tf.int64),
          'cols' : tf.FixedLenFeature([], tf.int64)
      })
  label = features['label']
  image_1 = features['image_1']
  image_2 = features['image_2']
  return image_1, image_2, label


#from the top, let's make it a FIFO queue loading files from a folder

def input_pipeline(filenames, batch_size, read_threads, num_epochs=None, imgwidth = 200, imgheight = 290):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example_list = [read_pair(filename_queue, imgwidth, imgheight)
                  for _ in range(read_threads)]
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example1_batch, example2_batch, label_batch = tf.train.shuffle_batch_join(
      example_list, batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example1_batch, example2_batch, label_batch

def _distort(img):
  if random.random() < 0.5:
    img = np.flip(img,2)
  #randomly scale it
  if random.random() < 0.5:
    zeros = np.zeros(img.shape,img1.dtype)
    
    zeros[0][img.shape[1]/4:img.shape[1]/4+img.shape[1]/2][img.shape[2]/4:img.shape[2]/2+img.shape[2]/4] = scipy.ndimage.zoom(img, 2, order=1)
    img = zeros
  #randomly add noise
  if random.random() < 0.5:
    img = img + np.random.normal(0,0x2000,img.shape)

  return img

def distorted_input_pipeline(filenames, batch_size, read_threads, num_epochs=None, imgwidth = 200, imgheight = 290):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example_list = [read_pair(filename_queue, imgwidth, imgheight)
                  for _ in range(read_threads)]
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example1_batch, example2_batch, label_batch = tf.train.shuffle_batch_join(
      example_list, batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example1_batch, example2_batch, label_batch
