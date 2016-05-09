import os
base_path = os.getcwd()+'/'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.cm as cm
import time
from PIL import Image
import sys
import math
from skimage.util import img_as_float
from skimage import io
import pdb

def get_local_weights(image, fg_indexs, bg_indexs, output_image_path, mask_size=49, sigma_3_square=0.1, a=0.6, b=0.2, num_step=10, fg_bias=0.3, bg_bias=-0.3, threshold=0.75, ignored_indexs=None):
    '''
    Get the updating weights given image
    '''
    if ignored_indexs is None:
        ignored_indexs = []
    
    if mask_size % 2 != 1:
        raise Exception("mask size has too be odd")

    start = time.time()

    padding_size = mask_size/2

    height, width = image.shape[0], image.shape[1]
    N = height*width
    F = np.asmatrix(np.zeros((N, mask_size**2)))
    
    F_star = np.asmatrix(np.zeros((N, mask_size**2)))
    C = np.zeros(N)
    f = lambda c1, c2: np.exp(-LA.norm(c1-c2)/(sigma_3_square))
    scale_to_1 = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
    other_indexs = [i for i in range(N) if (i not in fg_indexs and i not in bg_indexs)]
    
    # Test
    # i=j=0
    # k=l=10
    # print f(image[i,j,:3], image[k,l,:3])
    
    mask = [[1]*mask_size]*mask_size

    print "Calculate F..."
    count = 0
    for i in xrange(height):
        for j in xrange(width):
            for k in xrange(-(len(mask)/2), len(mask)/2+1):
                for l in xrange(-(len(mask[0])/2), len(mask[0])/2+1):
                    if (0 == k and 0 == l) or i+k<0 or i+k>height-1 or j+l<0 or j+l>width-1:
                        continue
                    F[i*width+j, (k+mask_size/2)*mask_size+l+mask_size/2] = f(image[i,j,:3], image[i+k,j+l,:3])
            count += 1

            F_star[i*width+j, :] = np.divide(F[i*width+j, :], np.sum(F[i*width+j,:]))
            C[i*width+j] = 1./np.max(F[i*width+j,:])

            print 100*float(count)/(height*width),"% completed          \r",

    print ""

    C_star = a*(C-np.min(C))/(np.max(C)-np.min(C))+b

    done = time.time()
    print "Done. ", done - start, "seconds."
    start = done

    return 

if __name__ == '__main__':
    '''
    args:
        argv[1]: Image file
        argv[2]: Saliency map file
        argv[3]: Result image(after cut) file
        argv[4]: Refined saliency map file
        argv[5]: New height
        argv[6]: New width
    '''
    if len(sys.argv) < 3:
        raise ValueError("Please input the image file and saliency file")

    image_file, saliency_image_file = sys.argv[1], sys.argv[2]
    after_cut_name = sys.argv[3]
    output_image_name = sys.argv[4]

    input_image_path = base_path+image_file
    saliency_image_path = base_path+saliency_image_file
    # output_image_path = base_path+output_directory+'/'+saliency_image_file.split('/')[-1][:-4]+'.bmp'
    output_image_path = base_path+output_image_name

    image = img_as_float(io.imread(input_image_path))
    saliency_image = img_as_float(io.imread(saliency_image_path))

    # resize the image

    new_height = int(sys.argv[5])
    new_width = int(sys.argv[6])

    if new_height > 0 and new_width > 0:
        old_image = mpimg.imread(input_image_path)

        image = Image.open(input_image_path)
        saliency_image = Image.open(saliency_image_path)
        
        old_width = image.width
        old_height = image.height

        image = image.resize((new_width, new_height), Image.ANTIALIAS)
        saliency_image =saliency_image.resize((new_width, new_height), Image.ANTIALIAS)

        input_image_path = input_image_path[:-4]+'-'+str(new_height)+'-'+str(new_width)+input_image_path[-4:]
        image.save(input_image_path)
        saliency_image_path = saliency_image_path[:-4]+'-'+str(new_height)+'-'+str(new_width)+saliency_image_path[-4:]
        saliency_image.save(saliency_image_path)

        # output_image_path = base_path+output_directory+'/'+saliency_image_file.split('/')[-1][:-4]+'-'+str(new_height)+'-'+str(new_width)+'.bmp'

    foreground_indexs = get_foreground_indexs(saliency_image, output_image_path) 
    background_indexs = get_background_indexs(saliency_image, output_image_path) 

    get_local_weights(image, foreground_indexs, background_indexs, output_image_path)
