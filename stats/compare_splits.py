import pandas as pd
import numpy as np


def get_components(file):
    components = pd.read_csv(file, header=None, sep=";")[0]
    components = [c.replace("_refinedTextures", "").replace("style_mesh_", "") for c in components if
                  any(p.lower() in c.lower() for p in ['window', 'door', 'dome', 'column', 'tower'])]
    components = ["/".join([c.split("/")[-2], c.split("/")[-1].split("_")[0]]).lower() for c in components]
    return set(components)


def get_components_with_labels(file):
    components = pd.read_csv(file, header=None, sep=";").values
    components = [(c.replace("_refinedTextures", "").replace("style_mesh_", ""), l) for (c, l) in components if
                  any(p.lower() in c.lower() for p in ['window', 'door', 'dome', 'column', 'tower'])]
    components = [("/".join([c.split("/")[-2], c.split("/")[-1].split("_")[0]]).lower(), np.load(l)) for (c, l) in components]
    components = [(c, np.argmax(l)) for (c, l) in components]
    return set(components)


files_mink_ps_prefix = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/svm_mink_ps/data/buildnetply100Knocolor/combined_splits_final_unique_selected_common/layer_n-2_features_weighted_sum_per_component_max/classification_cross_val/"
files_ocnn_prefix = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/svm_ocnn/data/buildnetnocolordepth6/combined_splits_final_unique_selected_common/feature_concat_as_is_per_component_avg/classification_cross_val/"
files_decor_prefix = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/svm_decorgan/data/ORIGINAL/combined_splits_final_unique_selected_common/layer_n-1_features_avg/classification_cross_val/"

for split in range(12):
    # mink_ps_components = get_components(files_mink_ps_prefix + f"trainfold{split}.csv")
    # mink_ps_components.union(get_components(files_mink_ps_prefix + f"testfold{split}.csv"))
    #
    # ocnn_components = get_components(files_ocnn_prefix + f"trainfold{split}.csv")
    # ocnn_components.union(get_components(files_ocnn_prefix + f"testfold{split}.csv"))
    #
    # decor_components = get_components(files_decor_prefix + f"trainfold{split}.csv")
    # decor_components.union(get_components(files_decor_prefix + f"testfold{split}.csv"))
    #
    # print(len(mink_ps_components), len(ocnn_components), len(decor_components))
    mink_ps_components = get_components_with_labels(files_mink_ps_prefix + f"trainfold{split}.csv")
    mink_ps_components.union(get_components_with_labels(files_mink_ps_prefix + f"testfold{split}.csv"))

    ocnn_components = get_components_with_labels(files_ocnn_prefix + f"trainfold{split}.csv")
    ocnn_components.union(get_components_with_labels(files_ocnn_prefix + f"testfold{split}.csv"))

    decor_components = get_components_with_labels(files_decor_prefix + f"trainfold{split}.csv")
    decor_components.union(get_components_with_labels(files_decor_prefix + f"testfold{split}.csv"))

    print(len(mink_ps_components), len(ocnn_components), len(decor_components))
