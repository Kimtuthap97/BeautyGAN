# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2
import argparse
import time
from PIL import Image
from model.BiSeNet import BiSeNet
import torch
import torchvision.transforms as transforms
from mtcnn import MTCNN
from tqdm import tqdm
from utils import preprocess, evaluate, deprocess, crop_face
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='./celeba/Img/img_align_celeba', help='path to the no_makeup image')
parser.add_argument('--bbox_file', type=str, default='bbox.pkl')
args = parser.parse_args()

# batch_size = 1
img_size = 256

no_makeups = glob.glob(os.path.join(args.img_path, '*.*'))
no_makeups = [x for x in no_makeups if os.path.isfile(x)==True]
makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
with open('bbox.pkl', 'rb') as f:
    bbox_dict = pickle.load(f)
    
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session(config=tf_config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph(os.path.join('model', 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint('model'))
graph = ops.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

# INFERENCE
for i in tqdm(range(len(no_makeups))):
#     print(no_makeups[i])
    no_makeup = imread(no_makeups[i])
#     no_makeup = crop_face(no_makeup)
    bbox=bbox_dict[no_makeups[i].split('/')[-1]]
    if bbox == [0, 0, 0, 0]:
        pass
    else:
#     cv2.imwrite('crop_face.png', no_makeup)
#     print(no_makeup.shape)
#     print(no_makeup[i])
#     try:
        x, y, ext, ext = bbox[0], bbox[1], bbox[2], bbox[3]
        no_makeup = no_makeup[y:y+ext, x:x+ext, :]
        no_makeup = cv2.resize(no_makeup, (img_size, img_size))
        face = evaluate(no_makeup, img_size)
        X_img = np.expand_dims(preprocess(no_makeup), 0)
    #     print('X_img', X_img.shape)
        for j in range(len(makeups)):
            makeup = cv2.resize(imread(makeups[j]), (img_size, img_size))
            Y_img = np.expand_dims(preprocess(makeup), 0)
            start_time = time.time()
            Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
            duration = time.time() - start_time
            Xs_ = deprocess(Xs_)
            X_temp = no_makeup/255.
            Xs_[0][np.where(face==0)]=X_temp[np.where(face==0)]
            imsave('./celeba/makeup/{}_{}.png'.format(no_makeups[i].split('/')[-1], j), Xs_[0])