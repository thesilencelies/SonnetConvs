#using the people comparing system instead
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import sonnet as snt
import argparse

import InceptionModule
import CrossInputNeighborhoodDifferences as CIND
import MultiScale as ms

import PeopleCompare as pcs

#actual training
def reshapeMod1(inp):
   return tf.reshape(inp,[-1,28,28,1])

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def main(_):
   # Import data
   people = pc.read_data_sets()

   imgsize = 784*2

   # Create the model
   x = tf.placeholder(tf.float32, [None, imgsize])

   # Define loss and optimizer
   y_ = tf.placeholder(tf.float32, [None,2])

   keep_prob = tf.placeholder(tf.float32)

   # break the input into it's two tensors first
   split = tf.split(x,2,1)

   kernel_shape = 3
   #per-input convolutions then max pooling
   conv = snt.Conv2D(output_channels=5,kernel_shape=kernel_shape,
                        stride=1,name="conv1")
   conv2 = snt.Conv2D(output_channels=5,kernel_shape=kernel_shape,
                        stride=1,name="conv2")
   conv5 = snt.Conv2D(output_channels=3,kernel_shape=5,
                        stride=1,name="summaryFeatures")
   maxpool = lambda x : tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
   perinput_net = ms.MultiScaleModule([conv,tf.nn.relu, conv2,maxpool], 3, name="per_input")
   #cross input then two  linear layers
   linh = snt.Linear(1000, name="hidden")
   lino = snt.Linear(2, name="linear")
   both_net = snt.Sequential([CIND.CrossInputNeighborhoodDifferences(),conv5,tf.nn.relu, tf.contrib.layers.flatten, linh, tf.nn.relu, lino], name="combined") 


   y_res = both_net( (perinput_net(reshapeMod1(split[0])), perinput_net(reshapeMod1(split[1]))) )

   cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_res))
   train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
   correct_prediction = tf.equal(tf.argmax(y_res, 1), tf.argmax(y_, 1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

   reglosses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

   train_step = tf.train.RMSPropOptimizer(1e-4, 0.9,
                                    momentum=0.9,
                                    epsilon=1.0).minimize(cross_entropy)


   #variable_summaries(y_res)
   saver = tf.train.Saver()

   #training
   with tf.Session() as sess:
       #summary sutff
       merged = tf.summary.merge_all()
       train_writer = tf.summary.FileWriter('/tmp/tensorflow' + '/train',
                                               sess.graph)
       test_writer = tf.summary.FileWriter('/tmp/tensorflow' + '/test')
       #training
       sess.run(tf.global_variables_initializer())
       for i in range(20000):
         batch = people.train.next_batch(50)
         if i % 100 == 0:
           train_accuracy = accuracy.eval(feed_dict={
               x: batch[0], y_: batch[1], keep_prob: 1.0})
           print('step %d, training accuracy %g' % (i, train_accuracy))
         train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
         if i % 1000 == 999:
           saver.save(sess,'peopleModel', global_step=i)
       accuracies = []
       for _ in range(people.test.num_examples):
         test_batch = people.test.next_batch(500)
         accuracies.append(accuracy.eval(feed_dict={
               x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))
       print('test accuracy %g' % sum(accuracies)/len(accuracies))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

