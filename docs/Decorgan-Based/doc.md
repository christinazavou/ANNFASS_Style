
export:
python splits/decorgan/create_building_or_component_txt.py

python mymain.py --config_yml settings/local/export/trained_on_buildnet_component/religious_and_annfass/style_encoder/i16o128/g32d32z8.yml
python mymain.py --config_yml settings/local/export/trained_on_buildnet_component/religious_and_annfass/original/i16o128/g32d32z8.yml
python mymain.py --config_yml settings/local/export/trained_on_buildnet_component/religious_and_annfass/adain/i16o128/g32d32.yml

svm:
sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan_results/from_turing/trained_on_buildnet_component_setA/style_encoder/i16o128/g32d32z8/encodings_religious_and_annfass decor_style_encoder "--layer discr_all" simple True False /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv
sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan_results/from_turing/trained_on_buildnet_component_setA/original/i16o128/g32d32z8/encodings_windowdoorcolumndometower decor_original "--layer discr_all" simple True False /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv True
sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan_results/from_turing/trained_on_buildnet_component_setA/adain/i16o128/g32d32/encodings_religious_and_annfass decor_adain "--layer discr_all" simple True False /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv

gia random:
sh ./repeat_experiments.sh /media/graphicslab/BigData1/zavou/ANNFASS_CODE/decorgan_results/from_turing/trained_on_buildnet_component_setA/original/i16o128/g32d32z8/encodings_windowdoorcolumndometower decor_original "--layer discr_all" random True False /media/graphicslab/BigData1/zavou/ANNFASS_DATA/Combined_Buildings/selected_components_with_style.csv
