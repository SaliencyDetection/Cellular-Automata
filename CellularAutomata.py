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
import pdb

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_background_indexs(image, output_image_path, quantile=0.15, ignored_indexs=None):
    # not a gray scale
    if ignored_indexs is None:
        ignored_indexs = []
    if len(image.shape)>2 and image.shape[2] > 1:
        image = rgb2gray(image) 
    
    image_flat = image.flatten()
    # set as maximum to be ignored
    image_flat[ignored_indexs] = 2.0
    indexs = np.argsort(image_flat)[:quantile*image_flat.size]
    
    background_image_flat = np.ones((image.shape[0]*image.shape[1]))
    background_image_flat[indexs] = 0
    background_image = background_image_flat.reshape((image.shape[0], image.shape[1]))
    output_image_PIL = Image.fromarray((background_image*255.).astype(np.uint8))
    output_image_path = output_image_path[:-4]+'-background'+output_image_path[-4:]
    output_image_PIL.save(output_image_path)
    
    return indexs
    
def get_foreground_indexs(image, output_image_path, quantile=0.01, ignored_indexs=None):
    if ignored_indexs is None:
        ignored_indexs = []
    # not a gray scale
    if len(image.shape)>2 and image.shape[2] > 1:
        image = rgb2gray(image) 
    
    image_flat = image.flatten()
    image_flat[ignored_indexs] = -1.0
    indexs = np.argsort(image_flat,)[-quantile*image_flat.size:]

    foreground_image_flat = np.ones((image.shape[0]*image.shape[1]))
    foreground_image_flat[indexs] = 0
    foreground_image = foreground_image_flat.reshape((image.shape[0], image.shape[1]))
    output_image_PIL = Image.fromarray((foreground_image*255.).astype(np.uint8))
    output_image_path = output_image_path[:-4]+'-foreground'+output_image_path[-4:]
    output_image_PIL.save(output_image_path)
    
    return indexs

def cellular_automata(image, fg_indexs, bg_indexs, output_image_path, mask_size=5, sigma_3_square=0.1, a=0.6, b=0.2, num_step=10, fg_bias=0.3, bg_bias=-0.3, threshold=0.5, ignored_indexs=None):
    if ignored_indexs is None:
        ignored_indexs = []
    
    start = time.time()

    height, width = image.shape[0], image.shape[1]
    N = height*width
    F = np.asmatrix(np.zeros((N, N)))
    f = lambda c1, c2: np.exp(-LA.norm(c1-c2)/(sigma_3_square))
    scale_to_1 = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
    other_indexs = [i for i in range(N) if (i not in fg_indexs and i not in bg_indexs)]
    
    # Test
    # i=j=0
    # k=l=10
    # print f(image[i,j,:3], image[k,l,:3])
    
    mask = [[1]*mask_size]*mask_size

    print "Calculate F..."
    for i in xrange(height):
        for j in xrange(width):
            for k in xrange(i-len(mask)/2, i+len(mask)/2+1):
                for l in xrange(j-len(mask[0])/2, j+len(mask[0])/2+1):
                    if (i == k and j == l) or k<0 or k>height-1 or l<0 or l>width-1:
                        continue
                    F[i*width+j, k*width+l] = f(image[i,j,:3], image[k,l,:3])
    done = time.time()
    print "Done. ", done - start, "seconds."
    start = done

    D = np.asmatrix(np.diag(np.asarray(np.sum(F, axis=1)).flatten()))

    F_star = inv(D)*F

    C_diag = np.asarray(np.divide(1, np.max(F, axis=1))).flatten()
    C_max, C_min = np.max(C_diag), np.min(C_diag)
    C_star_diag = a*(C_diag-C_min)/(C_max-C_min)+b
    C_star = np.asmatrix(np.diag(C_star_diag))
    
    # S_0 = np.asmatrix(0.5*np.ones((N, 1)))
    # S_0 = np.np.asarray(S_0).flatten()[ignored_indexs] = 0.1

    S_0 = 0.5*np.ones((N, 1))
    S_0[ignored_indexs] = 0.1
    S_0 = np.asmatrix(S_0)
    S = S_0

    func = lambda x: 1. if x > 1. else (0. if x < 0. else x)
        
    for i in xrange(num_step):
        S[fg_indexs] += fg_bias
        S[bg_indexs] += bg_bias
    
        S_new = C_star*S + (np.identity(N)-C_star)*F_star*S
        S_new[other_indexs] = scale_to_1(S_new[other_indexs])
        S_new = np.vectorize(func)(S_new)

        print i, "iteration"
        print "Norm of difference:", LA.norm(S_new-S)
        S = S_new

    for i in xrange(num_step):    
        S_new = C_star*S + (np.identity(N)-C_star)*F_star*S
        S_new = scale_to_1(S_new)
        print i, "iteration"
        print "Norm of difference:", LA.norm(S_new-S)
        S = S_new

    output_image = S.reshape(height, width, 1)
    # plt.imshow(output_image, cmap = cm.Greys_r, vmin = 0, vmax = 1)
    # plt.show()
    output_image_PIL = Image.fromarray((output_image*255.).astype(np.uint8))
    output_image_path = output_image_path[:-4]+'-'+str(mask_size)+output_image_path[-4:]
    print "Image saved at", output_image_path
    output_image_PIL.save(output_image_path)

    output_image_flat = np.asarray(output_image).flatten()
    saliency_indexs = [i for i in xrange(len(output_image_flat)) if output_image_flat[i] > threshold]
    return saliency_indexs

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError("Please input the image file and saliency file")

    image_file, saliency_image_file = sys.argv[1], sys.argv[2]
    output_directory = sys.argv[5]
    
    input_image_path = base_path+image_file
    saliency_image_path = base_path+saliency_image_file
    output_image_path = base_path+output_directory+'/'+saliency_image_file.split('/')[-1][:-4]+'.bmp'

    # convert jpg to png
    if input_image_path[-4:] == '.jpg':
        image = Image.open(input_image_path)
        input_image_path = input_image_path[:-4]+'.png'
        image.save(input_image_path)
    if saliency_image_path[-4:] == '.jpg':
        image = Image.open(saliency_image_path)
        saliency_image_path = saliency_image_path[:-4]+'.png'
        image.save(saliency_image_path)

    # resize the image

    new_height = int(sys.argv[3])
    new_width = int(sys.argv[4])

    if new_height > 0 and new_width > 0:
        
        image = Image.open(input_image_path)
        saliency_image = Image.open(saliency_image_path)

        image = image.resize((new_width, new_height), Image.ANTIALIAS)
        saliency_image =saliency_image.resize((new_width, new_height), Image.ANTIALIAS)

        input_image_path = input_image_path[:-4]+'-'+str(new_height)+'-'+str(new_width)+input_image_path[-4:]
        image.save(input_image_path)
        saliency_image_path = saliency_image_path[:-4]+'-'+str(new_height)+'-'+str(new_width)+saliency_image_path[-4:]
        saliency_image.save(saliency_image_path)

        output_image_path = base_path+output_directory+'/'+saliency_image_file.split('/')[-1][:-4]+'-'+str(new_height)+'-'+str(new_width)+'.bmp'

    image = mpimg.imread(input_image_path)
    saliency_image = mpimg.imread(saliency_image_path)

    foreground_indexs = get_foreground_indexs(saliency_image, output_image_path) 
    background_indexs = get_background_indexs(saliency_image, output_image_path) 

    cellular_automata(image, foreground_indexs, background_indexs, output_image_path)
