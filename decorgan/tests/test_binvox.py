from common_utils import binvox_rw

binvox_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components/RELIGIOUSchurch_mesh3135/style_mesh_group44_129_129_door__unknown/model.binvox"
with open(binvox_file, "rb") as fin:
    h = binvox_rw.read_header(fin)
print(h)
