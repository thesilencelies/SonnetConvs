#using the people comparing system instead
"""
Makes a graph that can be used to compare people and decide if they are the same or not

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import sonnet as snt
import argparse
import sys

import InceptionModule
import CrossInputNeighborhoodDifferences as CIND
import MultiScale as ms

import PeopleCompare as pc

#scales and reshapes
def reshapeMod1(inp):
   return tf.reshape(tf.scalar_mul(1./0xffff,inp),[-1,290,200,1])

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
  parser = argparse.ArgumentParser()
  parser.add_argument("train_data")
  parser.add_argument("test_data")
  parser.add_argument("--batch_size", type=int, default=50)
  parser.add_argument("--n_read_threads", type=int, default = 3)
  parser.add_argument("--num_epochs", type=int, default=None)
  args = parser.parse_args()

  # Import data
  filenames = [args.train_data]
  # Create the model
  with tf.device('/cpu:0'): #we need to load using the CPU or it allocated a stupid amount of memory
    x1, x2, y_ = pc.input_pipeline(filenames, args.batch_size, args.n_read_threads, num_epochs=args.num_epochs)

  keep_prob = tf.placeholder(tf.float32)

  kernel_shape = 3
  batch_norm = snt.BatchNorm()
  #per-input convolutions then max pooling
  conv = snt.Conv2D(output_channels=5,kernel_shape=kernel_shape,
                      stride=1,name="conv1")
  conv2 = snt.Conv2D(output_channels=5,kernel_shape=kernel_shape,
                      stride=1,name="conv2")
  conv3 = snt.Conv2D(output_channels=5,kernel_shape=kernel_shape,
                      stride=1,name="conv2")
  conv5 = snt.Conv2D(output_channels=3,kernel_shape=5,
                      stride=1,name="summaryFeatures")
  maxpool = lambda x : tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
  perinput_net = ms.MultiScaleModule([conv,tf.nn.relu, conv2,maxpool,conv3,maxpool], 3, name="per_input")
  #cross input then two  linear layers
  linh = snt.Linear(1000, name="hidden")
  lino = snt.Linear(2, name="linear")
  both_net = snt.Sequential([CIND.CrossInputNeighborhoodDifferences(),conv5,tf.nn.relu, tf.contrib.layers.flatten, linh, tf.nn.relu, lino], name="combined") 


  y_res = both_net( (perinput_net(batch_norm(reshapeMod1(x1))), perinput_net(batch_norm(reshapeMod1(x2)))) )

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


  with tf.Session() as sess:
    #summary sutff
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/tensorflow' + '/train',
                                             sess.graph)
    sess.run(tf.global_variables_initializer())


    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        i = 0
        while not coord.should_stop():
            # Run training steps
            sess.run(train_step, feed_dict = {keep_prob: 0.5})
            if i % 1000 == 999:
              saver.save(sess,'peopleModel', global_step=i)
            #talk to us about what's happening
            if i% 100 == 99:
              train_accuracy = accuracy.eval(feed_dict={keep_prob: 1.0})
              print('step %d, test accuracy %g' % (i, train_accuracy))
            i += 1
    except tf.errors.OutOfRangeError:
        print('Done testing -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    saver.save(sess,'FinalPeopleModel', global_step=i)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='.',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

