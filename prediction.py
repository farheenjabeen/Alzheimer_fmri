import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import matplotlib.pyplot as plt
import random

caffe.set_mode_cpu() 

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img

mean_blob = caffe_pb2.BlobProto()
with open('/20_LMDB_224/mean_224.binaryproto','rb') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


net = caffe.Net('deploy.prototxt', 'resnet_18_iter_100000.caffemodel', caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
idx = 0
idad = 0
idcn = 0
idemci = 0
idlmci = 0
idmci = 0
idsmc = 0

n_acc = 0
ad_acc = 0
cn_acc = 0
emci_acc = 0
lmci_acc = 0
mci_acc = 0
smc_acc = 0

#Loading images
lmdb_file = "/20_LMDB_224/valid_224_lmdb"
lmdb_env = lmdb.open(lmdb_file)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    test_label = datum.label
    data = caffe.io.datum_to_array(datum)
    img = data.astype(np.uint8)
    img = np.transpose(img, (1,2,0))

    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']
    print("Predicted:", pred_probas.argmax(), "|Actual:", test_label)
    
    if(pred_probas.argmax() == test_label):
        n_acc = n_acc + 1
        if(pred_probas.argmax() == 0):
            ad_acc = ad_acc + 1
        elif(pred_probas.argmax() == 1):
            cn_acc = cn_acc + 1
        elif(pred_probas.argmax() == 2):
            emci_acc = emci_acc + 1
        elif(pred_probas.argmax() == 3):
            lmci_acc = lmci_acc + 1
        elif(pred_probas.argmax() == 4):
            mci_acc = mci_acc + 1
        elif(pred_probas.argmax() == 5):
            smc_acc = smc_acc + 1

    idx = idx + 1
    if(test_label == 0):
        idad = idad + 1
    if(test_label == 1):
        idcn = idcn + 1
    if(test_label == 2):
        idemci = idemci + 1
    if(test_label == 3):
        idlmci = idlmci + 1
    if(test_label == 4):
        idmci = idmci + 1
    if(test_label == 5):
        idsmc = idsmc + 1

    print(idx, "|", idad,"|",idcn , "|", idemci ,"|", idlmci , "|", idmci , "|", idsmc)

accuracy = float((n_acc*100)/idx)
accuracy0 = float((ad_acc*100)/idad)
accuracy1 = float((cn_acc*100)/idcn)
accuracy2 = float((emci_acc*100)/idemci)
accuracy3 = float((lmci_acc*100)/idlmci)
accuracy4 = float((mci_acc*100)/idmci)
accuracy5 = float((smc_acc*100)/idsmc)


print("Test Accuracy:", accuracy, "%")
print("Test Accuracy of AD:", accuracy0, "%")
print("Test Accuracy of CN:", accuracy1, "%")
print("Test Accuracy of EMCI:", accuracy2, "%")
print("Test Accuracy of LMCI:", accuracy3, "%")
print("Test Accuracy of MCI:", accuracy4, "%")
print("Test Accuracy of SMC:", accuracy5, "%")
