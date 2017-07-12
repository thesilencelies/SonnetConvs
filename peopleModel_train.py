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
  parser.add_argument("--batch_size", type=int, default=100)
  parser.add_argument("--n_read_threads", type=int, default = 3)
  parser.add_argument("--num_epochs", type=int, default=None)
  parser.add_argument("--image_width", type=int, default=200)
  parser.add_argument("--image_height", type=int, default=290)
  parser.add_argument("--imagescale", type=int,default=0xffff)
  args = parser.parse_args()

  global_step = tf.contrib.framework.get_or_create_global_step()

  inc_global = tf.assign_add(global_step, 1, name="incriment")

  img1, img2, label = pm.distorted_inputs(args.train_data, args.batch_size, args.n_read_threads, args.num_epochs, image_width=args.image_width, image_height=args.image_height )
 
  logits = pm.model(img1,img2, image_width=args.image_width, image_height=args.image_height, scaling_term=1.0/args.imagescale)
  
  loss = pm.loss(logits, label)

  train_op = pm.train(loss, global_step, args.batch_size, 1000)

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
    sess.graph.finalize()
    try:
        while not coord.should_stop():
            # Run training steps
            sess.run(train_op)
            if (tf.train.global_step(sess, global_step) % 1000) == 999:
              saver.save(sess,args.save_dir, global_step=global_step)
            #talk to us about what's happening
            if (tf.train.global_step(sess, global_step) % 100) == 99:
              train_accuracy = accuracy.eval()
              print('step %d, train accuracy %g' % (tf.train.global_step(sess, global_step), train_accuracy))
            sess.run(inc_global)
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

