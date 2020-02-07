import sys
import os
import glob
import random
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb

def make_datum(img, label):
    return caffe_pb2.Datum(
        channels=1,
        width=224,
        height=224,
        label=label,
        data=img.tostring())


class GenerateLmdb(object):

    def __init__(self, img_path):
        self.img_lst = glob.glob(os.path.join(img_path, '*', '*.png'))
        print ('input_img list num is %s' % len(self.img_lst))
        random.shuffle(self.img_lst)



    def generate_lmdb(self, label_lst, percentage, train_path, validation_path):
        print ('now generate train lmdb')
        self._generate_lmdb(label_lst, percentage, True, train_path)
        print ('now generate validation lmdb')
        self._generate_lmdb(label_lst, percentage, False, validation_path)
        print ('\n generate all images')

    def _generate_lmdb(self, label_lst, percentage, b_train, input_path):
        output_db = lmdb.open(input_path, map_size=int(1e11))
        split_ratio =  (percentage * len(self.img_lst))/100
        
        with output_db.begin(write=True) as in_txn:
            for idx, img_path in enumerate(self.img_lst):
                if b_train:
                    if idx < split_ratio:
                        img =  cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (224,224))
                        label = label_lst.index(img_path.split('/')[-1].split('\\')[-2])
                        print(idx, label, img_path)
                        datum = make_datum(img, label)
                        str_id = '{:0>5d}'.format(idx)
                        in_txn.put(str_id.encode('ascii'), datum.SerializeToString())
                else:
                    if idx >= split_ratio:
                        img =  cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (224,224))
                        label = label_lst.index(img_path.split('/')[-1].split('\\')[-2])
                        print(idx, label, img_path)
                        datum = make_datum(img, label)
                        str_id = '{:0>5d}'.format(idx)
                        in_txn.put(str_id.encode('ascii'), datum.SerializeToString())
        output_db.close()
        

def get_label_lst_by_dir(f_dir):
    return os.listdir(f_dir)


img_path = '/Data/...'
cl = GenerateLmdb(img_path)
train_lmdb = '/train_224_lmdb'
validation_lmdb = '/valid_224_lmdb'

input_path = '/Data/.....'
label_lst = get_label_lst_by_dir(input_path)
print ('label_lst is: %s' % ', '.join(label_lst))
percentage = 80  # percentage of training data
cl.generate_lmdb(label_lst, percentage, train_lmdb, validation_lmdb)

