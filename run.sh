# !/bin/sh

python ca_slic.py test/cat_origin.jpg test/cat_saliency.jpg test/cat_origin_after_cut.png saliencymap/refined_saliency_cat_origin.png 500 0.05 0.25 1
python ca_slic.py test/origin_label1_img_0008.png test/saliency_label6_img_0008.jpg test/origin_label1_img_0008_after_cut.png saliencymap/refined_origin_label1_img_0008.png 500 0.05 0.25 0
python ca_slic.py source/origin_0241.jpg source/saliency_label3_0241.jpg source/origin_0241_aftercut.png saliencymap/refined_saliency_label3_0241.png 500 0.05 0.25 0
