#using the people comparing system instead
"""
trains the people comparison model
"""
import tensorflow as tf
import peopleModel as pm

import argparse

def main(_):
  parser = argparse.ArgumentParser()
  parser.add_argument("train_data")
  parser.add_argument("save_dir")
  parser.add_argument("--batch_size", type=int, default=50)
  parser.add_argument("--n_read_threads", type=int, default = 3)
  parser.add_argument("--num_epochs", type=int, default=None)
  args = parser.parse_args()

  global_step = tf.contrib.framework.get_or_create_global_step()

  img1, img2, label = pm.distorted_inputs(args.train_data, args.batch_size, args.n_read_threads, args.num_epochs)
 
  logits = pm.model(img1,img2)
  
  loss = pm.loss(logits, label)

  train_op = pm.train(loss, global_step, args.batch_size, 5000)

  correct_prediction = tf.equal(tf.argmax(label, 1), tf.argmax(logits, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
            sess.run(train_op)
            if i % 1000 == 999:
              saver.save(sess,args.save_dir, global_step=global_step)
            #talk to us about what's happening
            if i% 100 == 99:
              train_accuracy = accuracy.eval()
              print('step %d, train accuracy %g' % (i, train_accuracy))
            i += 1
    except tf.errors.OutOfRangeError:
        print('Done testing -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    saver.save(sess,args.save_dir, global_step=global_step)



if __name__ == '__main__':
  tf.app.run()

