# Cellular-Automata

http://cseweb.ucsd.edu/~yaq007/0873.pdf


### Usage

#### example
python ca_slic.py test/cat_origin.jpg test/cat_saliency.jpg test/cat_origin_after_cut.png saliencymap/refined_saliency_cat_origin.png 500 0.05 0.25 0

#### args
        argv[1](str):   Image file
        argv[2](str):   Saliency map file
        argv[3](str):   Result image(after cut) file
        argv[4](str):   Refined saliency map file
        argv[5](int):   Number of segments of superpixel
        argv[6](float): Foreground quantile
        argv[7](float): Background quantile
        argv[8](bool):  Show intermediate image for debugging
    
