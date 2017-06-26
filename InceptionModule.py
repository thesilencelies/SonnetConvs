#implimentation of the standard inceptionnet v3 inception module in sonnet
import tensorflow as tf
import sonnet as snt


class InceptionModule(snt.AbstractModule):
  def __init__(self, output_channels, name="inception_module"):
    super(InceptionModule, self).__init__(name=name)
    self._output_channels = output_channels

  def _build(self, inputs):
    reshapeFlat = lambda x : tf.contrib.layers.flatten(x)

    conv1d5 = snt.Conv2D(output_channels=self._output_channels, kernel_shape=1,
                        stride=1,name="inception5input")

    conv1d3 = snt.Conv2D(output_channels=self._output_channels, kernel_shape=1,
                        stride=1,name="inception3input")

    conv1dm = snt.Conv2D(output_channels=self._output_channels, kernel_shape=1,
                        stride=1,name="inceptionpoolinput")

    conv1d1 = snt.Conv2D(output_channels=self._output_channels, kernel_shape=1,
                        stride=1,name="inception1channel")

    conv3d5a = snt.Conv2D(output_channels=self._output_channels, kernel_shape=3,
                        stride=1,name="inception5stage1")

    conv3d5b = snt.Conv2D(output_channels=self._output_channels, kernel_shape=3,
                        stride=1,name="inception5stage2")

    conv3d3 = snt.Conv2D(output_channels=self._output_channels, kernel_shape=3,
                        stride=1,name="inception3channel")

    maxpool = lambda x : tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    return tf.concat([reshapeFlat(conv3d5b(conv3d5a(conv1d5(inputs)))),
                     reshapeFlat(conv3d3(conv1d3(inputs))),
                     reshapeFlat(maxpool(conv1dm(inputs))),
                     reshapeFlat(conv1d1(inputs))],1)          # then connect it.
