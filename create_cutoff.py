r"""Convert cutoff dataset to TFRecord for object_detection.
Attention Please!!!

1)For easy use of this script, Your coco dataset directory struture should like this :
    +Your  dataset root
        +class1
        +class2
        +class3
            -pic1.jpg
            -pic2.jpg

Example usage:
    python create_cutoff.py --data_dir=/path/to/your/root/directory \
        --set=train \
        --output_path=/where/you/want/to/save/pascal.record
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pycocotools.coco import COCO
from PIL import Image
from random import shuffle
import os, sys
import numpy as np
import tensorflow as tf
import logging
from label_to_id import label_get_id
import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to jpeg dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set or validation set')
flags.DEFINE_string('output_filepath', '', 'Path to output TFRecord')
flags.DEFINE_bool('resize_flag',0,'whether to resize images')
FLAGS = flags.FLAGS

def getsize(infile):
    try:
       with Image.open(infile) as im:
         target=im.size
         return target
    except IOError:
       print("cannot create thumbnail for '%s'" % infile)

def resize_ssd(infile,outfile, target_x = 300, target_y = 300):
    try:
       with Image.open(infile) as im:
         if im.size[0] >= im.size[1] and im.size[0] > target_x:
             rate=target_x/float(im.size[0])
             target=(target_x, round(im.size[1]*rate))
         elif im.size[1] > im.size[0] and im.size[1] > target_y:
             rate=target_y/float(im.size[1])
             target=(round(im.size[0]*rate),target_y)
         else:
             target=im.size
         im = im.resize(target)
         im=im.crop((0,0,target_x,target_y))
         im.save(outfile, "JPEG")
         return target,[target_x,target_y]
    except IOError:
       print("cannot create thumbnail for '%s'" % infile)

def load_cutoff_dection_dataset(imgs_dir, label_file, resize_flag = 1):
    """Load data from dataset by pycocotools. This tools can be download from "http://mscoco.org/dataset/#download"
    Args:
        imgs_dir: directories of images
        label_file: file path of pbtxt file
    Return:
        cutoff_data: list of dictionary format information of each image
    """
    cutoff_data = []

    print("load_cutoff_dection_dataset")
    for dirname in os.listdir(imgs_dir):
      path = os.path.join(imgs_dir, dirname)
      if not os.path.isdir(path):
        continue
      print("Readling images: %s "%(dirname))
      dirid=label_get_id(label_file,dirname)
      if dirid < 0:
        print("Cannot find label: %s "%(dirname))
        continue
      outpath=os.path.join(path,"out")
      if not os.path.exists(outpath):
        os.makedirs(outpath)
      for filename in os.listdir(path):
        if not filename.endswith(".jpg"):
          continue
        img_path = os.path.join(path, filename)
        imgout_path = os.path.join(path,"out", filename)
        if resize_flag == 1:
          target,newsize= resize_ssd(img_path,imgout_path)
          img_path = imgout_path
          bboxes_data = [0.0,0.0,target[0]/float(newsize[0]),target[1]/float(newsize[1])]
        else:
          newsize=getsize(img_path)
          bboxes_data = [0.0,0.0,1.0,1.0]
        # the format of coco bounding boxs is [Xmin, Ymin, width, height]
        bboxes = []
        labels = []
        img_info = {}
        bboxes.append(bboxes_data)
        labels.append(dirid)
        img_bytes = tf.gfile.FastGFile(img_path,'rb').read()

        img_info['pixel_data'] = img_bytes
        img_info['height'] = newsize[1]
        img_info['width'] = newsize[0]
        img_info['bboxes'] = bboxes
        img_info['labels'] = labels
        cutoff_data.append(img_info)

    return cutoff_data


def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    bboxes = img_data['bboxes']
    xmin, xmax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(img_data['height']),
        'image/width': dataset_util.int64_feature(img_data['width']),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label': dataset_util.int64_list_feature(img_data['labels']),
        'image/encoded': dataset_util.bytes_feature(img_data['pixel_data']),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf-8')),
    }))
    return example

def main(_):
    if FLAGS.set == "train":
        imgs_dir = os.path.join(FLAGS.data_dir, 'train')
        pbtxt_filepath = os.path.join(FLAGS.data_dir,'data.pbtxt')
        print("Convert cutoff train file to tf record")
    elif FLAGS.set == "val":
        imgs_dir = os.path.join(FLAGS.data_dir, 'val')
        pbtxt_filepath = os.path.join(FLAGS.data_dir,'data.pbtxt')
        print("Convert cutoff val file to tf record")
    else:
        raise ValueError("you must either convert train data or val data")
    # load total coco data
    cutoff_data = load_cutoff_dection_dataset(imgs_dir,pbtxt_filepath,resize_flag=FLAGS.resize_flag)
    total_imgs = len(cutoff_data)
    # write coco data to tf record
    with tf.python_io.TFRecordWriter(FLAGS.output_filepath) as tfrecord_writer:
        for index, img_data in enumerate(cutoff_data):
            if index % 100 == 0:
                print("Converting images: %d / %d" % (index, total_imgs))
            example = dict_to_coco_example(img_data)
            tfrecord_writer.write(example.SerializeToString())
    print("Converting images : %d " % (total_imgs))


if __name__ == "__main__":
    tf.app.run()
