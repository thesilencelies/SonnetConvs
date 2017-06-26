#multiple scales input/output module written in sonnet

import tensorflow as tf
import sonnet as snt


class MultiScaleModule(snt.AbstractModule):
  def __init__(self, layers, nscales, name="Multiple_Scales"):
    super(MultiScaleModule, self).__init__(name=name)
    # Store a copy of the iterable in a tuple to ensure users cannot modify the
    # iterable later, and protect against iterables which can only be read once.
    self._layers = tuple(layers)

    is_not_callable = [(i, mod) for i, mod in enumerate(self._layers)
                       if not callable(mod)]

    if is_not_callable:
      raise TypeError("Items {} not callable with types: {}".format(
          ", ".join(str(i) for i, _ in is_not_callable),
          ", ".join(type(layer).__name__ for _, layer in is_not_callable)))

    self._nscales = nscales

  def _build(self, inputs):
    #scales down the inputs using max pooling
    results = []
    scaled = inputs
    for l in self._layers:
      scaled = l(scaled)
    results.append(scaled)
    finalsize = [results[0].shape[1].value, results[0].shape[2].value]
    for i in range(1,self._nscales):
      scaled = tf.nn.max_pool(inputs, ksize=[1,2*i,2*i,1], strides=[1, 2*i,2*i, 1], padding='SAME')
      #apply the layers to each scale independantly
      for l in self._layers:
        scaled = l(scaled)
      #upscale the results to the same scale
      scaled = tf.image.resize_bilinear(scaled, finalsize)
      results.append(scaled)

    #concatenate them
    return tf.concat(results,3)


#test

if __name__ == "__main__":
  a = tf.reshape(tf.constant([1.0,2.0,3.0,4.0]),[1,2,2,1])
  conv= snt.Conv2D(output_channels=1,kernel_shape=1,stride=1,name="conv")
  msm = MultiScaleModule([conv],2)
  b = msm(a)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(b.eval())
