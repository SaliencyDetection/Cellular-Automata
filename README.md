# Cellular-Automata

http://cseweb.ucsd.edu/~yaq007/0873.pdf


### Usage

#### example

python ca_slic.py -i source/origin_0241.jpg -sl source/saliency_label3_0241.jpg -rsl saliencymap/refined_saliency_label3_0241.png -ns 500 -fq 0.05 -bq 0.25 -fb 0.5 -bb -0.3 -d

#### args

    usage: python ca_slic.py [-h] [-i IMAGE] [-sl SALIENCY_LIST [SALIENCY_LIST ...]]
               [-rsl OUTPUT_SALIENCY_LIST [OUTPUT_SALIENCY_LIST ...]]
               [-ns NUM_SEGMENTS] [-fq FG_QUANTILE] [-bq BG_QUANTILE]
               [-fb FG_BIAS] [-bb BG_BIAS] [-d]

    optional arguments:
      -h, --help            show this help message and exit
      -i IMAGE, --image IMAGE
                            Image file
      -sl SALIENCY_LIST [SALIENCY_LIST ...], --saliency_list SALIENCY_LIST [SALIENCY_LIST ...]
                            Saliency map file list
      -rsl OUTPUT_SALIENCY_LIST [OUTPUT_SALIENCY_LIST ...], --output_saliency_list OUTPUT_SALIENCY_LIST [OUTPUT_SALIENCY_LIST ...]
                            Refined saliency map file (output)
      -ns NUM_SEGMENTS, --num_segments NUM_SEGMENTS
                            Number of segments of superpixel
      -fq FG_QUANTILE, --fg_quantile FG_QUANTILE
                            Foreground quantile
      -bq BG_QUANTILE, --bg_quantile BG_QUANTILE
                            Background quantile
      -fb FG_BIAS, --fg_bias FG_BIAS
                            Foreground bias
      -bb BG_BIAS, --bg_bias BG_BIAS
                            Background bias
      -d, -debug            Save intermediate image for debugging
