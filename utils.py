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

def evaluate(img, img_size, cp='/home/thaontp79/makeup/BeautyGAN/model/cp/79999_iter.pth'):
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
#         img = Image.open(img_path)
        img = cv2.resize(img.astype('uint8'), (512, 512))
        img = to_tensor(img).cuda()
        img = torch.unsqueeze(img, 0)
#         print(type(img), img.shape)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        
        return cv2.resize(parsing.astype('uint8'), (img_size, img_size))
    
def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

def crop_face(img):
#     print(file_name)
    detector = MTCNN()
#     img = cv2.imread(file_name)
#     print(temp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     print('img.shape', img.shape)
    faces=detector.detect_faces(img)
#     print(faces)
    if len(faces)==0:
        print('{0}: No face found')
        return img[0:1, 0:1, 0:1]
    else:
        faces=faces[0]['box']
        x, y, z, k = faces[0], faces[1], faces[2], faces[3]
#         print('x, y, z, k', x, y, z, k)
        ext = [z, k][np.argmax([z, k])]
        ext=int(ext*1.2)
        x=int(x-0.5*int(ext-z))
        if x < 0:
            x =0
        if y < 0:
            y=0
#         plt.imshow(temp[y:y+ext, x:x+ext, :])
    
        return cv2.cvtColor(img[y:y+ext, x:x+ext, :], cv2.COLOR_RGB2BGR)