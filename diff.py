import cv2
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from tqdm import tqdm
def evaluate(result, a):
    (h, w, _) = result.shape
    
    original = result[256:, 0:256, :]
    print(original.dtype)
    diff = original
    for i in range(1, 10):
        output = result[256:, 256*i:256*(i+1), :]
        temp = original.astype('float') + a*(output.astype('float') - original.astype('float'))
        temp = np.clip(temp, 0, 255).astype('uint8')
        diff = np.concatenate([diff, temp], axis = 1)
#     plt.figure(figsize=(15, 15*5))
#     plt.imshow(diff)
#     plt.show()
    return diff
list_img = glob.glob(os.path.join('/home/thaontp79/makeup/BeautyGAN/results', '*.*'))
for i in tqdm(range(0, len(list_img))):
    img = cv2.cvtColor(cv2.imread(list_img[i]), cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=(15, 15*5))
#     plt.imshow(img)
#     plt.title(list_img[i])
#     plt.show()
    compare_img=img
    for a in np.arange(0,1.2, 0.1):
#         print(a)
        result=evaluate(img, a)
        compare_img=np.concatenate([compare_img, result], axis=0)
#     print(img)
#     print(compare_img)
    plt.figure(figsize=(10, 11))
    plt.imshow(compare_img)
    plt.savefig('./diff/{0}'.format(list_img[i].split('/')[-1]))
#     plt.show()
    plt.close()