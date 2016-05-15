# !/bin/sh

python CellularAutomata.py test/cat_origin.jpg test/cat_saliency.jpg test/cat_origin_after_cut.png saliencymap/refined_saliency_cat_origin.png 0 0

python ca_slic.py -i source/origin_0241.jpg -sl source/saliency_label3_0241.jpg -rsl saliencymap/refined_saliency_label3_0241.png -ns 500 -fql 0.05 -bq 0.25 -fb 0.5 -bb -0.3 -d -f feature.npy
python ca_cut.py -i source/origin_0241.jpg -ac saliencymap/saliency_label3_0241_aftercut.png -s source/saliency_label3_0241.jpg -rs saliencymap/refined_saliency_label3_0241.png -ns 500 -fq 0.05 -bq 0.25 -fb 0.5 -bb -0.3 -d -f feature.npy