import pdb
import os
from os.path import isfile, join

image_path = os.getcwd() + '/source'
files = [f for f in os.listdir(image_path) if isfile(join(image_path, f))]

origins = [f for f in files if f[:6]=='origin']
saliencys = [f for f in files if f[:8]=='saliency']

origins.sort()
saliencys.sort()

for i in xrange(len(origins)):
    if origins[i][6:] == saliencys[i][8:]:
        os.system('python CellularAutomata.py '+ 'source/'+origins[i]+ ' source/'+saliencys[i]+' 50 50 output')
