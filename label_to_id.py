import os, sys
import tensorflow as tf
from object_detection.utils import label_map_util

def label_get_id(label_file,label_name):
    try:
        get_dict = label_map_util.get_label_map_dict(label_file,label_name)
        get_label_id = get_dict[label_name]
    except Exception as e:
        print ("No find label name {}".format(label_name))
        return -1
    #print "get_label_id:{}".format(get_label_id)
    return get_label_id


if __name__ == "__main__":
	label_file = sys.argv[1]
	label_name = sys.argv[2]	
	get_label_id = label_get_id(label_file,label_name)
	print ("get_label_id:{}".format(get_label_id))
