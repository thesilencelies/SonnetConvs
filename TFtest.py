import tensorflow as tf
import numpy as np

def model(features, labels, mode) :
   #build a linear model
   W1 = tf.get_variable("W1", [2,2], dtype=tf.float64)
   b1 = tf.get_variable("b1", [2], dtype=tf.float64)
   h1 = tf.nn.sigmoid(tf.matmul(features['x'],W1) + b1)
   #two layer
   W2 = tf.get_variable("W2", [2,1], dtype=tf.float64)
   b2 = tf.get_variable("b2", [1], dtype=tf.float64)
   y = tf.nn.sigmoid(tf.matmul(h1,W2) + b2)
   #loss function
   loss = tf.reduce_sum(tf.square(y - labels))
   #training sub graph
   global_step = tf.train.get_global_step()
   optimiser = tf.train.GradientDescentOptimizer(0.01)
   train = tf.group(optimiser.minimize(loss),
                     tf.assign_add(global_step, 1))
   #connect it up
   return tf.contrib.learn.ModelFnOps(
            mode=mode, predictions=y,
            loss = loss,
            train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)

#dataset
x = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = np.array([0.,1.,1.,0.])

input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=10000)

#train
estimator.fit(input_fn=input_fn, steps=1000)

#evaluate
print(estimator.evaluate(input_fn = input_fn, steps = 10))
