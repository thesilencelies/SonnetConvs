#an implimentation of Cross-Input Neighborhood Differences from http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ahmed_An_Improved_Deep_2015_CVPR_paper.pdf in sonnet

import tensorflow as tf
import sonnet as snt

class CrossInputNeighborhoodDifferences(snt.AbstractModule):
  """ docstring """
  def __init__(self, name="Cross_Input_Neighborhood_Differences"):
    super(CrossInputNeighborhoodDifferences, self).__init__(name=name)
  
  def _build(self, inputs):
     """takes an input tuple of exactly two tensors of identical dimension, and computes the neighbourhood difference for each"""
     #compute the map in both directions
     #padding is above, below, left, right
     list1 = []
     list2 = []
     for i in range(0,5) :
       for j in range(0,5) : 
         list1.append(tf.pad(inputs[0], [[0,0],[i,4-i],[j,4-j],[0,0]]))
         list2.append(tf.pad(inputs[1], [[0,0],[i,4-i],[j,4-j],[0,0]]))

     pad1 = tf.concat(list1,3)
     pad2 = tf.concat(list2,3)
     tile1 = tf.tile(tf.pad(inputs[0],[[0,0],[2,2],[2,2],[0,0]]), [1,1,1,25])
     tile2 = tf.tile(tf.pad(inputs[1],[[0,0],[2,2],[2,2],[0,0]]), [1,1,1,25])

     return tf.concat([tile1 - pad2, tile2 - pad1], 3)


##for testing
if __name__ == '__main__':
   a = tf.reshape(tf.constant([1.0,2.0,3.0,4.0]),[1,2,2,1])
   b = tf.reshape(tf.constant([2.0,2.0,4.0,4.0]),[1,2,2,1])
   mdl = CrossInputNeighborhoodDifferences()
   res = mdl((a,b))
   with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      print(sess.run(res))
