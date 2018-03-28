r"""Convert raw Microsoft COCO dataset to TFRecord for object_detection.
Attention Please!!!

1)For easy use of this script, Your coco dataset directory struture should like this :
    +Your coco dataset root
        +train2014
        +val2014
        +annotations
            -instances_train2014.json
            -instances_val2014.json
2)To use this script, you should download python coco tools from "http://mscoco.org/dataset/#download" and make it.
After make, copy the pycocotools directory to the directory of this "create_coco_tf_record.py"
or add the pycocotools path to  PYTHONPATH of ~/.bashrc file.

Example usage:
    python create_coco_tf_record.py --data_dir=/path/to/your/coco/root/directory \
        --set=train \
        --output_path=/where/you/want/to/save/pascal.record
        --shuffle_imgs=True
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

import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw Microsoft COCO dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set or validation set')
flags.DEFINE_string('output_filepath', '', 'Path to output TFRecord')
flags.DEFINE_bool('shuffle_imgs',True,'whether to shuffle images of coco')
flags.DEFINE_bool('resize_flag',1,'whether to resize images of coco')
FLAGS = flags.FLAGS

def resize_frcnn(infile,outfile, target_max = 1024, target_min = 600):
    try:
       with Image.open(infile) as im:
         if im.size[0] >= im.size[1]:
             rx=target_max/float(im.size[0])
             ry=target_min/float(im.size[1])
             if abs(rx-1) >= abs(ry-1):
                  target=(target_max, round(im.size[1]*rx))
                  ratio=rx
             else:
                  target=(round(im.size[0]*ry), target_min)
                  ratio=ry
             crop=((target[0]-target_max)/2, (target[1]-target_min)/2, (target[0]+target_max)/2, (target[1]+target_min)/2)
             org=[(target[0]-target_max)/2, (target[1]-target_min)/2]
             newsize=[target_max,target_min]
         else:
             rx=target_min/float(im.size[0])
             ry=target_max/float(im.size[1])
             if abs(rx-1) >= abs(ry-1):
                  target=(target_min, round(im.size[1]*rx))
                  ratio=rx
             else:
                  target=(round(im.size[0]*ry), target_max)
                  ratio=ry
             crop=((target[0]-target_min)/2, (target[1]-target_max)/2, (target[0]+target_min)/2, (target[1]+target_max)/2)
             org=[(target[0]-target_min)/2, (target[1]-target_max)/2]
             newsize=[target_min,target_max]
         # print(ratio)
         # print(str(target))
         im = im.resize(target)
         # print(str(crop))
         im = im.crop(crop)
         im.save(outfile, "JPEG")
         return ratio,org,newsize
    except IOError:
       print("cannot create thumbnail for '%s'" % infile)

def load_coco_dection_dataset(imgs_dir, annotations_filepath, shuffle_img = True, resize_flag = 1):
    """Load data from dataset by pycocotools. This tools can be download from "http://mscoco.org/dataset/#download"
    Args:
        imgs_dir: directories of coco images
        annotations_filepath: file path of coco annotations file
        shuffle_img: wheter to shuffle images order
    Return:
        coco_data: list of dictionary format information of each image
    """
    coco = COCO(annotations_filepath)
    img_ids = coco.getImgIds() # totally 82783 images
    cat_ids = coco.getCatIds() # totally 90 catagories, however, the number of categories is not continuous, \
                               # [0,12,26,29,30,45,66,68,69,71,83] are missing, this is the problem of coco dataset.

    if shuffle_img:
        shuffle(img_ids)

    coco_data_H = []
    coco_data_V = []

    nb_imgs = len(img_ids)
    labeld ={}
    print("load_coco_dection_dataset")
    for index, img_id in enumerate(img_ids):
        if index % 100 == 0:
            print("Readling images: %d / %d "%(index, nb_imgs))
        # if index >=2000:
        #    break
        img_info = {}
        bboxes = []
        labels = []

        img_detail = coco.loadImgs(img_id)[0]
        pic_height = img_detail['height']
        pic_width = img_detail['width']
        # print("("+str(pic_height)+","+str(pic_width)+")")
        ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)

        img_path = os.path.join(imgs_dir, img_detail['file_name'])
        imgout_path = os.path.join(imgs_dir+"/out/", img_detail['file_name'])
        ratio=1.0
        org=[0,0]
        newsize= [pic_width,pic_height]
        if resize_flag == 1:
            print("resize_frcnn"+str(index))
            ratio,org,newsize= resize_frcnn(img_path,imgout_path)
            # print(imgout_path)
            img_path=imgout_path
            print("resize_frcnn done"+str(index))
        for ann in anns:
            bboxes_data = ann['bbox']
            bboxes_data = [(ratio*bboxes_data[0]-org[0])/float(newsize[0]), (ratio*bboxes_data[1]-org[1])/float(newsize[1]),\
                                  (ratio*bboxes_data[2])/float(newsize[0]), (ratio*bboxes_data[3])/float(newsize[1])]
                         # the format of coco bounding boxs is [Xmin, Ymin, width, height]
            if bboxes_data[0] <0:
                bboxes_data[2]+=bboxes_data[0]
                bboxes_data[0]=0
            if bboxes_data[1] <0:
                bboxes_data[3]+=bboxes_data[1]
                bboxes_data[1]=0
            if bboxes_data[2] > 1-bboxes_data[0]:
                bboxes_data[2]=1-bboxes_data[0]
            if bboxes_data[3] > 1-bboxes_data[1]:
                bboxes_data[3]=1-bboxes_data[1]
            if bboxes_data[2] <= 0 or bboxes_data[3] <=0 or bboxes_data[0] > 1 or bboxes_data[1] > 1:
                continue
            # print(str(bboxes_data))
            bboxes.append(bboxes_data)
            labels.append(ann['category_id'])
            if ann['category_id'] in labeld:
                labeld[ann['category_id']] += 1
            else:
                labeld[ann['category_id']]=1

        img_bytes = tf.gfile.FastGFile(img_path,'rb').read()

        img_info['pixel_data'] = img_bytes
        img_info['height'] = newsize[1]
        img_info['width'] = newsize[0]
        img_info['bboxes'] = bboxes
        img_info['labels'] = labels

        if resize_flag != 1:
                coco_data_H.append(img_info)
        else:
                if newsize[1] >= newsize[0]:
                    coco_data_H.append(img_info)
                else:
                    coco_data_V.append(img_info)
    print(labeld)
    return coco_data_H, coco_data_V


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
        imgs_dir = os.path.join(FLAGS.data_dir, 'train2017')
        annotations_filepath = os.path.join(FLAGS.data_dir,'annotations','instances_train2017.json')
        print("Convert coco train file to tf record")
    elif FLAGS.set == "val":
        imgs_dir = os.path.join(FLAGS.data_dir, 'val2017')
        annotations_filepath = os.path.join(FLAGS.data_dir,'annotations','instances_val2017.json')
        print("Convert coco val file to tf record")
    else:
        raise ValueError("you must either convert train data or val data")
    # load total coco data
    coco_data_H, coco_data_V = load_coco_dection_dataset(imgs_dir,annotations_filepath,shuffle_img=FLAGS.shuffle_imgs,resize_flag=FLAGS.resize_flag)
    total_imgs_H = len(coco_data_H)
    total_imgs_V = len(coco_data_V)
    # write coco data to tf record
    if FLAGS.resize_flag == 1:
        with tf.python_io.TFRecordWriter(FLAGS.output_filepath+"H") as tfrecord_writer:
            for index, img_data in enumerate(coco_data_H):
                if index % 100 == 0:
                    print("Converting images: %d / %d" % (index, total_imgs_H))
                example = dict_to_coco_example(img_data)
                tfrecord_writer.write(example.SerializeToString())
        with tf.python_io.TFRecordWriter(FLAGS.output_filepath+"V") as tfrecord_writer:
            for index, img_data in enumerate(coco_data_V):
                if index % 100 == 0:
                    print("Converting images: %d / %d" % (index, total_imgs_V))
                example = dict_to_coco_example(img_data)
                tfrecord_writer.write(example.SerializeToString())
        print("Converting images H, V: %d , %d" % (total_imgs_H, total_imgs_V))
    else:
        with tf.python_io.TFRecordWriter(FLAGS.output_filepath) as tfrecord_writer:
            for index, img_data in enumerate(coco_data_H):
                if index % 100 == 0:
                    print("Converting images: %d / %d" % (index, total_imgs_H))
                example = dict_to_coco_example(img_data)
                tfrecord_writer.write(example.SerializeToString())
        print("Converting images : %d " % (total_imgs_H))


if __name__ == "__main__":
    tf.app.run()
