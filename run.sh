# !/bin/sh

python CellularAutomata.py test/origin_label17_img_0041.jpg test/saliency_label17_img_0041.jpg test/origin_label17_img_0041_after_cut.png 20 20 saliencymap/refined_saliency_label17_img_0041.png
python ca_slic.py test/origin_label17_img_0041.jpg test/saliency_label17_img_0041.jpg test/origin_label17_img_0041_after_cut.png saliencymap/refined_saliency_label17_img_0041.png

python ca_slic.py test/cat_origin.jpg test/cat_saliency.jpg test/cat_origin_after_cut.png saliencymap/refined_saliency_cat_origin.png

python ca_slic.py test/origin_label1_img_0008.png test/saliency_label6_img_0008.jpg test/origin_label1_img_0008_after_cut.png saliencymap/refined_origin_label1_img_0008.png
# python CellularAutomata.py test/origin_label16_img_0041.jpg test/saliency_label16_img_0041.jpg s 50 50 saliencymap_
