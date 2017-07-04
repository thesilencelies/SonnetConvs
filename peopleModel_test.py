#using the people comparing system instead
"""
tests the people comparison model
"""
import tensorflow as tf
import peopleModel as pm

import argparse

def eval_once(saver, summary_writer, correct_prediction, summary_op, args):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    correct_prediction: correct_prediction op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(args.num_examples / args.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * args.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([correct_prediction])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(args):
  """Eval people comparison for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.

    image1, image2, labels = pm.inputs(args.test_data, args.batch_size, args.n_read_threads, args.num_epochs)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = pm.model(image1, image2)

    # Calculate predictions.
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))

    # Restore the moving average version of the learned variables for eval.

    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(args.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, correct_prediction, summary_op, args)
      time.sleep(1)



def main(_):
  parser = argparse.ArgumentParser()
  parser.add_argument("test_data")
  parser.add_argument("checkpoint_dir")
  parser.add_argument("--batch_size", type=int, default=50)
  parser.add_argument("--n_read_threads", type=int, default = 3)
  parser.add_argument("--num_epochs", type=int, default=None)
  parser.add_argument("--num_examples", type=int, default = 5000)
  parser.add_argument("--eval_dir", default = ".")
  args = parser.parse_args()
  evaluate(args)



if __name__ == '__main__':
  tf.app.run()

