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

def evaluate(img_path, img_size, cp='/home/thaontp79/makeup/BeautyGAN/model/cp/79999_iter.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(cp))
    net.eval()
    
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        img = Image.open(img_path)
#         print(img, type(img), img.shape)
#         print(img)
        img = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
#         print(type(img), img.shape)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0).astype('float32')
        return cv2.resize(parsing, (img_size, img_size))
#     cv2.resize(parsing, (img_size, img_size))

parser = argparse.ArgumentParser()
parser.add_argument('--no_makeup', type=str, default='/home/thaontp79/makeup/BeautyGAN/imgs/a2/24.png', help='path to the no_makeup image')
args = parser.parse_args()

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

batch_size = 1
img_size = 256
no_makeup = cv2.resize(imread(args.no_makeup), (img_size, img_size))
# print(args.no_makeup)
face = evaluate(args.no_makeup, img_size)

X_img = np.expand_dims(preprocess(no_makeup), 0)

# Get BACKGROUND

makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
result[img_size: 2 *  img_size, :img_size] = no_makeup / 255.

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join('model', 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint('model'))

graph = ops.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

for i in range(len(makeups)):
    makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
    Y_img = np.expand_dims(preprocess(makeup), 0)
    start_time = time.time()
#     print(X_img.shape, Y_img.shape)
    Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
    
    duration = time.time() - start_time
    Xs_ = deprocess(Xs_)
    ### HI ###
    #   17 hair
    #   12: upper lip
    #   13: lower lip
    #   1: face
    #   2, 3: eyes brow
    #   10: nose
    #   14: neck
    
    X_temp = no_makeup/255.
#     X_temp[]=Xs_[0][np.where(face==12)]
#     print(X_temp.shape, Xs_[0].shape, len(face), len(face[0]))
#     print(face)
#     Xs_[0][np.where(face==17)]=X_temp[np.where(face==17)]
    Xs_[0][np.where(face==0)]=X_temp[np.where(face==0)]
#     Xs_[0][np.where(face==17)]=X_temp[np.where(face==17)]
#     X_temp[np.where(face==1)]=Xs_[0][np.where(face==1)]
#     Xs_[0][np.where(face==17)]=X_temp[np.where(face==17)]
    imsave(makeups[i].split('/')[-1], Xs_[0])
    
#     print(X_img.shape, X_img[0].shape, X_img, X_temp[0])
#     print('Save image', makeups[i].split('/')[-1], ' Infer time: ', duration)
    
    result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup/255.
    result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = Xs_[0]
    
imsave('result.jpg', result)
