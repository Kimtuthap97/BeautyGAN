import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from mtcnn import MTCNN
from tqdm import tqdm as tqdm
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', type=str, default = '/home/thaontp79/makeup/BeautyGAN/makeup',
                    help='path to folder of input images to CROP FACES')
parser.add_argument('--out_path', type=str, default= '/home/thaontp79/makeup/BeautyGAN/angles/sample/',
                    help='OUTPUT directory')

args = parser.parse_args()

images_list = glob.glob(os.path.join(args.dir_path, '*.*'))
detector = MTCNN()
for file_name in tqdm(images_list):
#     print(file_name)
    temp = cv2.imread(file_name)
#     print(temp)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
#     temp=img[int(h*i):int(h*(i+1)), int(w*j):int(w*(j+1)), :]
#     gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    faces=detector.detect_faces(temp)
#     print(faces)
    if len(faces)==0:
        print('{0}: No face found'.format(file_name))
    else:
        faces=faces[0]['box']
        x, y, z, k = faces[0], faces[1], faces[2], faces[3]
        ext = [z, k][np.argmax([z, k])]
        ext=int(ext*1.2)
        x=int(x-0.5*int(ext-z))
        if x < 0: x =0
        plt.imshow(temp[y:y+ext, x:x+ext, :])

        cv2.imwrite(os.path.join(args.out_path, file_name.split('/')[-1]),
                    cv2.cvtColor(temp[y:y+ext, x:x+ext, :], cv2.COLOR_RGB2BGR))
#         plt.show()