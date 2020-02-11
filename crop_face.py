import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from mtcnn import MTCNN
from tqdm import tqdm as tqdm
import glob
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', type=str, default = '/home/thaontp79/makeup/BeautyGAN/celeba/Img/img_align_celeba',
                    help='path to folder of input images to CROP FACES')
parser.add_argument('--out_path', type=str, default= '/home/thaontp79/makeup/BeautyGAN/',
                    help='OUTPUT directory')
parser.add_argument('--ext', type=str, default='png')

args = parser.parse_args()
# print(args)
images_list = glob.glob(os.path.join(args.dir_path, '*.*'))
detector = MTCNN()
# print(os.path.join(args.dir_path, '*.*'))
# print(images_list)
bbox={}
# print('len', len(images_list), images_list[0])
for file_name in tqdm(images_list):
    temp = cv2.imread(file_name)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    faces=detector.detect_faces(temp)
    if len(faces)==0:
#         print('{0}: No face found'.format(file_name))
        bbox[file_name.split('/')[-1]]=[0, 0, 0, 0]
    else:
        faces=faces[0]['box']
        x, y, z, k = faces[0], faces[1], faces[2], faces[3]
        ext = [z, k][np.argmax([z, k])]
        ext=int(ext*1.2)
        x=int(x-0.5*int(ext-z))
        
        if x < 0:
            x =0
        if y < 0:
            y=0
        bbox[file_name.split('/')[-1]]=[x, y, ext, ext]
        
        if args.ext == 'png':
            cv2.imwrite(os.path.join(args.out_path, file_name.split('/')[-1]),
                        cv2.cvtColor(temp[y:y+ext, x:x+ext, :], cv2.COLOR_RGB2BGR))
            
print('Saving file')
with open(os.path.join(args.out_path, 'bbox.pkl'), 'wb') as handle:
    pickle.dump(bbox, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

# print a == b
            
#         plt.show()