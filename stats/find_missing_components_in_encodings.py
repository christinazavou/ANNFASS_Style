import json
import os


groups_dirs = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/groups_june17,/media/graphicslab/BigData/zavou/ANNFASS_DATA/ANNFASS_Buildings_may/groups"


def missing_groups(encodings_dir):
    components = set()
    missing = 0
    for building in os.listdir(encodings_dir):
        group_file = None
        for group_dir in groups_dirs.split(","):
            if building in os.listdir(group_dir):
                group_file = os.path.join(group_dir, building, "groups.json")
        if group_file is None:
            print(f"AHA!! No group file for building {building}")
        else:
            with open(group_file, "r") as fin:
                groups = json.load(fin)
            groups = {key: value for key, value in groups.items()
                      if any(any(p in v.lower() for v in value)
                      for p in ['window', 'door', 'tower', 'column', 'dome'])}
            if len(groups) == 0:
                print(f"{building} has zero groups to be included")
            for component in os.listdir(os.path.join(encodings_dir, building)):
                group = component.replace('style_mesh_', '').split("_")[0].replace("group", "")
                if group in groups:
                    del groups[group]
                components.add(f"{building.replace('_refinedTextures', '')}/{group}")
            if len(groups) > 0:
                print(f"{building}, {groups}")
                missing += len(groups)
    print(f"missing {missing}")
    print(f"existing {len(components)}")
    return components


print("===================================DECOR=========================================")
decor_encodings = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan_results/from_turing/july27/groups_june17_uni_nor_components/original_clean/s8/encodings/layer_n-1_features/avg"
decor_encodings = missing_groups(decor_encodings)

print("===================================MINK PS=========================================")
mink_ps_encodings = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_minkowski_ps/buildnet_ply_100K/BuildnetVoxelization0_01Dataset/PS-HRNet3S2BD256/b32-i100000/latent_features_combined/test_split/layer_n-1_features/max_per_component/avg"
mink_ps_encodings = missing_groups(mink_ps_encodings)

# print("===================================MINK VAE=========================================")
# mink_vae_encodings = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/mink_results/gypsum/july9/various/export/encodings"
# missing_groups(mink_vae_encodings)

print("===================================OCNN=========================================")
ocnn_encodings = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/ocnn_jobs/nocolor_depth6_export/best_total_loss/encodings/feature_concat/as_is_per_component/avg"
ocnn_encodings = missing_groups(ocnn_encodings)

# print("===================================3D-AAE=========================================")
# aae_encodings = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/3daae-results/buildnet/gypsum/autoencoder/experiment_ae/encodings/02000_z_e"
# missing_groups(aae_encodings)

common = decor_encodings & mink_ps_encodings & ocnn_encodings
print(f"common: {len(common)}")
out_file = "/media/graphicslab/BigData/zavou/ANNFASS_DATA/Combined_Buildings/common_ocnn_decor_minkps.txt"
with open(out_file, "w") as fout:
    for c in common:
        fout.write(c+"\n")
