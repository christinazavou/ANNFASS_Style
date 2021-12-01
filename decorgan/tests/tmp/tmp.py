import numpy as np
import os
from common_utils import binvox_rw_customized
import cutils
import mcubes

from utils import write_ply_triangle


def run_voxelize(in_file, out_file, vox_num=512):
    if os.path.exists(out_file):
        return
    print(in_file)

    maxx = 0.5
    maxy = 0.5
    maxz = 0.5
    minx = -0.5
    miny = -0.5
    minz = -0.5

    command = "./binvox -bb "+str(minx)+" "+str(miny)+" "+str(minz)+" "+str(maxx)+" "+str(maxy)+" "+str(maxz)+" "+f" -d {vox_num} -e "+in_file

    os.system(command)


def run_flood_fill(in_name, out_name, vox_num=512):
    queue = np.zeros([vox_num*vox_num*64,3], np.int32)
    state_ctr = np.zeros([vox_num*vox_num*64,2], np.int32)

    if not os.path.exists(out_name):
        print(in_name)

        voxel_model_file = open(in_name, 'rb')
        vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file,fix_coords=False)

        batch_voxels = vox_model.data.astype(np.uint8)+1
        cutils.floodfill(batch_voxels,queue,state_ctr)

        with open(out_name, 'wb') as fout:
            binvox_rw_customized.write(vox_model, fout, state_ctr)

        voxel_model_file = open(out_name, 'rb')
        vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file)
        batch_voxels = vox_model.data.astype(np.uint8)
        vertices, triangles = mcubes.marching_cubes(batch_voxels, 0.5)
        write_ply_triangle(out_name.replace(".binvox", ".ply"), vertices, triangles)


if __name__ == '__main__':
    obj_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components/RELIGIOUSchurch_mesh3135/style_mesh_group44_129_129_door__unknown/model_christina.obj"
    binvox_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components/RELIGIOUSchurch_mesh3135/style_mesh_group44_129_129_door__unknown/model_christina.binvox"
    floodfill_file = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/groups_june17_uni_nor_components/RELIGIOUSchurch_mesh3135/style_mesh_group44_129_129_door__unknown/model_filled_christina.binvox"
    run_voxelize(obj_file, binvox_file, 256)
    run_flood_fill(binvox_file, floodfill_file, 256)
