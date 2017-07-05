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
import random

import InceptionModule
import CrossInputNeighborhoodDifferences as CIND
import MultiScale as ms

import PeopleCompare as pc

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

IMAGE_HEIGHT = 290
IMAGE_WIDTH = 200

#scales and reshapes
def reshapeMod1(inp):
   return tf.reshape(tf.scalar_mul(1./0xffff,inp),[-1,IMAGE_HEIGHT,IMAGE_WIDTH,1])

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

def inputs(filename, batch_size, n_read_threads = 3, num_epochs = None):
  """
  reads the paired images for comparison
  input: name of the file to load from, parameters of the loading process
  output: the two images and the label (a logit classifier for 2 class - yes or no)
  """
  with tf.device('/cpu:0'): #we need to load using the CPU or it allocated a stupid amount of memory
    x1, x2, y_ = pc.input_pipeline([filename], batch_size, n_read_threads, num_epochs=num_epochs)
  return x1, x2, y_ 

def distorted_inputs(filename, batch_size, n_read_threads = 3, num_epochs = None):
  """
  reads the paired images for comparison
  input: name of the file to load from, parameters of the loading process
  output: the two images and the label (a logit classifier for 2 class - yes or no)
  """
  with tf.device('/cpu:0'): #we need to load using the CPU or it allocated a stupid amount of memory
    x1, x2, y_ = pc.distorted_input_pipeline([filename], batch_size, n_read_threads, num_epochs=num_epochs, imgwidth = IMAGE_WIDTH, imgheight = IMAGE_HEIGHT)
  return x1, x2, y_ 


def model(image1, image2):

  kernel_shape = 3
  batch_norm = snt.BatchNorm()
  #per-input convolutions then max pooling
  conv = snt.Conv2D(output_channels=5,kernel_shape=kernel_shape,
                      stride=1,name="conv1")
  conv2 = snt.Conv2D(output_channels=5,kernel_shape=kernel_shape,
                      stride=1,name="conv2")
  conv3 = snt.Conv2D(output_channels=5,kernel_shape=kernel_shape,
                      stride=1,name="conv3")
  conv5 = snt.Conv2D(output_channels=3,kernel_shape=5,
                      stride=1,name="summaryFeatures")
  maxpool = lambda x : tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
  perinput_net = ms.MultiScaleModule([conv,tf.nn.relu, conv2,maxpool,conv3,maxpool], 3, name="per_input")
  #cross input then two  linear layers
  linh = snt.Linear(1000, name="hidden")
  lino = snt.Linear(2, name="linear")
  both_net = snt.Sequential([CIND.CrossInputNeighborhoodDifferences(),conv5,tf.nn.relu,maxpool, tf.contrib.layers.flatten, linh, tf.nn.relu, lino], name="combined") 

  y_res = both_net( (perinput_net(batch_norm(reshapeMod1(image1))), perinput_net(batch_norm(reshapeMod1(image2)))) )
  return y_res



def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)


  return cross_entropy_mean


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op



def train(total_loss, global_step, batch_size, nex_per_epoch):
  """Train people Re-ID model
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
 # Variables that affect learning rate.
  num_batches_per_epoch = nex_per_epoch / batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  #loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  # with tf.control_dependencies([loss_averages_op]):
  opt = tf.train.RMSPropOptimizer(lr, 0.9,
                                  momentum=0.9,
                                  epsilon=1.0)

  # Apply gradients.
  apply_gradient_op = opt.minimize(total_loss)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Track the moving averages of all trainable variables.
  #variable_averages = tf.train.ExponentialMovingAverage(
  #    MOVING_AVERAGE_DECAY, global_step)
  #variables_averages_op = variable_averages.apply(tf.trainable_variables())

 # with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
 #   train_op = tf.no_op(name='train')

  return apply_gradient_op




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
                      stride=1,name="conv3")
  conv5 = snt.Conv2D(output_channels=3,kernel_shape=5,
                      stride=1,name="summaryFeatures")
  conv1 = snt.Conv2D(output_channels=1, kernel_shape=1,stride=1,name="1dconv")
  maxpool = lambda x : tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
  perinput_net = snt.Sequential([ms.MultiScaleModule([conv,tf.nn.relu, conv2,maxpool], 3, name="per_input"),tf.nn.relu,conv1, conv3,maxpool])
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
              print('step %d, train accuracy %g' % (i, train_accuracy))
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

