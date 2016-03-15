base_path = '/Users/Kuhn/Dropbox/Study/UCSD 2015 Fall/CSE 253 Neural Network/project/cellar_automata/ca_python/'

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

from CellularAutomata import rgb2gray, get_background_indexs, get_foreground_indexs, cellular_automata

def multi_label(image_list, saliency_image_list, output_image_path_list):
    ignored_indexs = None

    for i in xrange(len(image_list)):
        foreground_indexs = get_foreground_indexs(saliency_image_list[i], output_image_path, ignored_indexs=ignored_indexs) 
        background_indexs = get_background_indexs(saliency_image_list[i], output_image_path, ignored_indexs=ignored_indexs)
        if ignored_indexs == None:
            ignored_indexs = cellular_automata(image[i], foreground_indexs, background_indexs, output_image_path[i], ignored_indexs=ignored_indexs)
        else:
            ignored_indexs += cellular_automata(image[i], foreground_indexs, background_indexs, output_image_path[i], ignored_indexs=ignored_indexs)

if __name__ == '__main__':
    N = 2

    image_file_list = ['test/origin_label16_img_0041.jpg',
                       'test/origin_label17_img_0041.jpg']
    saliency_image_file_list = ['test/saliency_label16_img_0041.jpg',
                                'test/saliency_label17_img_0041.jpg']

    input_image_path_list = [base_path+image_file for image_file in image_file_list]
    saliency_image_path_list = [base_path+saliency_image_file for saliency_image_file in saliency_image_file_list]
    output_image_path_list = [base_path+'saliencymap/'+saliency_image_file.split('/')[-1][:-4]+'.bmp' for saliency_image_file in saliency_image_file_list]

    # convert jpg to png
    for i in xrange(N):
        if input_image_path_list[i][-4:] == '.jpg':
            image = Image.open(input_image_path_list[i])
            input_image_path_list[i] = input_image_path_list[i][:-4]+'.png'
            image.save(input_image_path_list[i])
    for i in xrange(N):
        if saliency_image_path_list[i][-4:] == '.jpg':
            image = Image.open(saliency_image_path_list[i])
            saliency_image_path_list[i] = saliency_image_path_list[i][:-4]+'.png'
            image.save(saliency_image_path_list[i])

    # resize the image
    image_list = [None]*N
    saliency_image_list = [None]*N
    for i in xrange(N):
        # if len(sys.argv) > 3:
        if True:
            # new_height = int(sys.argv[3])
            # new_width = int(sys.argv[4])
            new_height = 50
            new_width = 50

            image_list[i] = Image.open(input_image_path_list[i])
            saliency_image_list[i] = Image.open(saliency_image_path_list[i])

            image_list[i] = image_list[i].resize((new_width, new_height), Image.ANTIALIAS)
            saliency_image_list[i] =saliency_image_list[i].resize((new_width, new_height), Image.ANTIALIAS)

            input_image_path_list[i] = input_image_path_list[i][:-4]+'-'+str(new_height)+'-'+str(new_width)+input_image_path_list[i][-4:]
            image_list[i].save(input_image_path_list[i])
            saliency_image_path_list[i] = saliency_image_path_list[i][:-4]+'-'+str(new_height)+'-'+str(new_width)+saliency_image_path_list[i][-4:]
            saliency_image_list[i].save(saliency_image_path_list[i])
            output_image_path_list[i] = base_path+'saliencymap/'+saliency_image_file_list[i].split('/')[-1][:-4]+'-'+str(new_height)+'-'+str(new_width)+'.bmp'

        image_list[i] = mpimg.imread(input_image_path_list[i])
        saliency_image_list[i] = mpimg.imread(saliency_image_path_list[i])

    ignored_indexs = None
    for i in xrange(N):
        foreground_indexs = get_foreground_indexs(saliency_image_list[i], output_image_path_list[i], ignored_indexs=ignored_indexs) 
        background_indexs = get_background_indexs(saliency_image_list[i], output_image_path_list[i], ignored_indexs=ignored_indexs)
        if ignored_indexs == None:
            ignored_indexs = cellular_automata(image_list[i], foreground_indexs, background_indexs, output_image_path_list[i], ignored_indexs=ignored_indexs)
        else:
            ignored_indexs += cellular_automata(image_list[i], foreground_indexs, background_indexs, output_image_path_list[i], ignored_indexs=ignored_indexs)

    # foreground_indexs = get_foreground_indexs(saliency_image_list[i], output_image_path) 
    # background_indexs = get_background_indexs(saliency_image_list[i], output_image_path) 

    # cellular_automata(image, foreground_indexs, background_indexs, output_image_path)