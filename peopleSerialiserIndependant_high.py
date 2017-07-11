"""
turns the images of people in the given folder into a single numpy array, scaling and padding as appropriate, then serialises it so that it can be loaded easily
"""

import cv2
import numpy as np
import dill as pickle
import argparse
import csv
import random
import tensorflow as tf

def scaleAndPadToSize(img, width, height):
  #scale such that the height is correct
  res = cv2.resize(img, (img.shape[1]*height/img.shape[0],height))
  padwidth = width - res.shape[1]
  halfpad = int(padwidth/2)
  if padwidth < 0 :    
    res = res[:, -padwidth + halfpad : halfpad]    
  else:
    res = np.pad(res, [[0, 0],[halfpad,padwidth - halfpad]],'edge')
  return res


def makeAndSavePair(size, positive_ratio, usedLabels, usedRatio, height, width):
   # load the full set of images to a memory mapped file (this prevents us running out of memory here)
   imgarr = np.load("numpyarr.npy", mmap_mode='r')
   npos = positive_ratio * size
   nneg = size - npos
   perm = np.arange(int(len(usedLabels)*usedRatio))
   np.random.shuffle(perm)
   it = 0
   for _i in range(0,size):
      i = _i % len(perm)        
      # decide if it's positive or not
      positive = (npos/(npos + nneg)) < random.random()
      if positive :
         npos-=1
      else :
         nneg-=1
      # iterate through the random set until you hit an example of the correct type
      while True :
         it = (it + 1) % len(perm)
         if (usedLabels[it] == usedLabels[i]) == positive :
            break
      if positive :
         label = [1,0]
      else :
         label = [0,1]

     #construct the example
      example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
              # Features contains a map of string to Feature proto objects
              feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=label)),
                'image_1': tf.train.Feature(
                    float_list=tf.train.FloatList(value=np.nditer(imgarr[i].astype("float")))),
                'image_2': tf.train.Feature(
                    float_list=tf.train.FloatList(value=np.nditer(imgarr[it].astype("float")))),
                'rows': tf.train.Feature(
                       int64_list=tf.train.Int64List(value=[height])),
                'cols': tf.train.Feature(
                       int64_list=tf.train.Int64List(value=[width])),
       }))
      writer.write(example.SerializeToString())


if __name__ == "__main__":
  # parse the command line parameters
  parser = argparse.ArgumentParser()
  parser.add_argument("image_prefix")
  parser.add_argument("labels_file", help="labels as a csv file")
  parser.add_argument("labels_outfile", help="output file for the array")
  parser.add_argument("single_output")
  parser.add_argument("paired_output_train")
  parser.add_argument("paired_output_test")
  parser.add_argument("--image_width", type=int, default = 300)
  parser.add_argument("--image_height", type=int, default = 420)
  parser.add_argument("--positive_ratio", type=float, default = 0.5)
  parser.add_argument("--train_ratio", type=float, default = 0.8, help="the ration of the total used for training")
  parser.add_argument("--ntest_pairs", type=int, default = 5000, help="number of testing pairs to generate")
  parser.add_argument("--ntrain_pairs", type=int, default = 20000, help="number of training pairs to generate")

  args = parser.parse_args()

  labels = []
  # load the labels
  with open(args.labels_file, 'rb') as labfile:
    csvReader = csv.reader(labfile)
    for row in csvReader:
      labels.append(row)
  

  usedLabels = []
  #initially write the images to a non-paired file
  with tf.python_io.TFRecordWriter(args.single_output) as writer:
    #we need a numpy array of the images as well to get our desired result
    imglst = []
    # load the images
    for i in range(0,len(labels)):
      img = cv2.imread(args.image_prefix + str(i) + ".png", cv2.IMREAD_UNCHANGED)
      if img is not None:        
        img = (img/256).astype(np.uint8)
        # pad the images
        img = scaleAndPadToSize(img, int(args.image_width), int(args.image_height))
        img = np.reshape(img,[1,int(args.image_width), int(args.image_height),1])
        usedLabels.append(int(labels[i][0]))
        #construct the example
        example = tf.train.Example(
              # Example contains a Features proto object
              features=tf.train.Features(
                # Features contains a map of string to Feature proto objects
                feature={
                  # A Feature contains one of either a int64_list,
                  # float_list, or bytes_list
                  'label': tf.train.Feature(
                      int64_list=tf.train.Int64List(value=[int(labels[i][0])])),
                  'image': tf.train.Feature(
                      float_list=tf.train.FloatList(value=np.nditer(img.astype(float)))),
                  'rows': tf.train.Feature(
                         int64_list=tf.train.Int64List(value=[int(args.image_height)])),
                  'cols': tf.train.Feature(
                         int64_list=tf.train.Int64List(value=[int(args.image_width)])),
                  }))
        writer.write(example.SerializeToString())
        #also store it in a single numpy file so we can read it memory mapped
        imglst.append(img)
    np.save("numpyarr", np.concatenate(imglst))


  #now form pairs and save those to a different file
  with tf.python_io.TFRecordWriter(args.paired_output_train) as writer:

    makeAndSavePair(args.ntrain_pairs, args.positive_ratio, usedLabels, args.train_ratio, int(args.image_width), int(args.image_height))
    writer.close()
     

  with tf.python_io.TFRecordWriter(args.paired_output_test) as writer:
    makeAndSavePair(args.ntest_pairs, args.positive_ratio, usedLabels, 1, int(args.image_width), int(args.image_height))
    writer.close()
