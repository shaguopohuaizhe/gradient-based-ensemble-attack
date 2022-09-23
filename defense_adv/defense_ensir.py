from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv

import numpy as np
import tensorflow as tf
from scipy.misc import imread
from nets import inception_resnet_v2

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
    
tf.flags.DEFINE_string(
    'test_path', "../output", 'Input directory with images.')

tf.flags.DEFINE_string(
    'checkpoint_path', './ckpt/ens_adv_inception_resnet_v2.ckpt', 'Path to checkpoint for inception network.')
    
tf.flags.DEFINE_string(
    'csv_path', '../small_imagenet.csv', 'Path to csv for dataset.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 128, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS

def load_target_class():
  """Loads target classes."""
  with tf.gfile.Open(FLAGS.csv_path) as f:
      next(f)
      target_class = {os.path.split(row[0])[-1][:-5]+'.png': int(row[2])+1 for row in csv.reader(f) if len(row) >= 2}
  with tf.gfile.Open(FLAGS.csv_path) as f:
      next(f)
      true_label = {os.path.split(row[0])[-1][:-5]+'.png': int(row[1])+1 for row in csv.reader(f) if len(row) >= 2}
  return target_class, true_label
  
def load_images(batch_shape):
    """Read png images from input directory in batches.

    Args:
      test_path: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    
    with tf.gfile.Open(FLAGS.csv_path) as f:
        next(f)
        allfile =[row[0] for row in csv.reader(f) if len(row) >= 2]
    for filepath in allfile:
        cleandir = os.path.split(filepath)[-2]
        
        cleanfile = os.path.split(filepath)[-1][:-5]+'.png'
        
        dirname = os.path.split(cleandir)[-1]
        newdir = os.path.join(FLAGS.test_path,dirname)
        filepath = os.path.join(newdir,cleanfile)
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
         
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def main(_):

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    origin_preds = {}
    flip_preds = {}
    all_filenames = []
    
    all_images_taget_class, all_images_true_label = load_target_class()

    # Origin
    with tf.Graph().as_default():
        # Prepare graph
        # x_input is a binary image (converted in the load_images phase)
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False)

        predictions = end_points['Predictions']

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.checkpoint_path,
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images in load_images(batch_shape):
                pred = sess.run(predictions, feed_dict={x_input: images})
                target_class_for_batch = ([all_images_taget_class[n] for n in filenames])
                true_label_for_batch = ([all_images_true_label[n] for n in filenames])
                prediction = []
                for filename, p in zip(filenames, list(pred)):
                    origin_preds[filename] = p
                    
                    all_filenames.append(filename)
                    prediction.append(np.argmax(p))

                


    target = 0
    nontarget = 0
    for name in all_filenames:
        pre_label = np.argmax(origin_preds[name])
        
        if pre_label != all_images_true_label[name]:
            nontarget+=1
            
    
    fo = open("results_ensir.txt", "a")
    fo.write(str(nontarget/len(all_filenames)))
    fo.write('\n')
    fo.close()

if __name__ == '__main__':
    tf.app.run()