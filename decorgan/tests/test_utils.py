import numpy as np
import torch
import torch.nn.functional as F


def get_vox_from_binvox_1over2():
    voxel_model_6 = np.array([
        [1,2,3,4,13,16],
        [5,6,7,8,14,17],
        [9,10,11,12,15,18],
        [19,20,21,22,23,24],
        [1,2,3,4,5,6],
        [1,2,1,2,3,2]
    ]).astype(np.uint8)
    step_size = 3
    output_padding = 2-(4//step_size)
    voxel_model_2 = voxel_model_6[0::step_size,0::step_size]
    for i in range(step_size):
        for j in range(step_size):
                voxel_model_2 = np.maximum(voxel_model_2,voxel_model_6[i::step_size,j::step_size])
    voxel_model_4 = np.zeros([4,4],np.uint8)
    voxel_model_4[output_padding:-output_padding,output_padding:-output_padding] = voxel_model_2
    return voxel_model_4


def get_voxel_bbox(vox):
    #minimap
    vox_tensor = torch.from_numpy(vox).to(device).unsqueeze(0).unsqueeze(0).float()
    smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size = upsample_rate, stride = upsample_rate, padding = 0)
    smallmaskx = smallmaskx_tensor.detach().cpu().numpy()[0,0]
    smallmaskx = np.round(smallmaskx).astype(np.uint8)
    smallx,smally,smallz = smallmaskx.shape
    #x
    ray = np.max(smallmaskx,(1,2))
    xmin = 0
    xmax = 0
    for i in range(smallx):
        if ray[i]>0:
            if xmin==0:
                xmin = i
            xmax = i
    indices = np.where(ray == 1)
    assert max(indices[0][0], 1) == xmin and indices[0][-1] == xmax
    #y
    ray = np.max(smallmaskx,(0,2))
    ymin = 0
    ymax = 0
    for i in range(smally):
        if ray[i]>0:
            if ymin==0:
                ymin = i
            ymax = i
    indices = np.where(ray == 1)
    assert max(indices[0][0], 1) == ymin and indices[0][-1] == ymax
    #z
    ray = np.max(smallmaskx,(0,1))
    if asymmetry:
        zmin = 0
        zmax = 0
        for i in range(smallz):
            if ray[i]>0:
                if zmin==0:
                    zmin = i
                zmax = i
        indices = np.where(ray == 1)
        assert max(indices[0][0], 1) == zmin and indices[0][-1] == zmax
    else:
        zmin = smallz//2
        zmax = 0
        for i in range(zmin,smallz):
            if ray[i]>0:
                zmax = i

    return xmin,xmax+1,ymin,ymax+1,zmin,zmax+1


def test_get_points_from_voxel():
    from utils import get_vox_from_binvox_1over2, get_points_from_voxel, get_bound_points_from_voxel
    from utils.open3d_utils import PointCloud
    from utils.open3d_vis import interactive_plot

    voxels = np.ones((10,10,10))
    points = get_points_from_voxel(voxels)
    blues = np.zeros_like(points)
    blues[:,2] = 255
    points = PointCloud(points, blues)
    interactive_plot([points])

    bound_points = get_bound_points_from_voxel(voxels)
    reds = np.zeros_like(bound_points)
    reds[:, 0] = 255
    bound_points = PointCloud(bound_points, reds)
    interactive_plot([points, bound_points])

    infile = "/media/graphicslab/BigData/zavou/ANNFASS_CODE/decorgan-logs/preprocessed_data/buildnet_component_v2_orient_debug/RELIGIOUSchurch_mesh1270/style_mesh_group22_47_23_tower_steeple__unknown/model_filled.binvox"
    voxels = get_vox_from_binvox_1over2(infile)
    points = get_points_from_voxel(voxels)
    points = PointCloud(points)
    interactive_plot([points])


if __name__ == '__main__':
    test_get_points_from_voxel()
    exit()
    r = get_vox_from_binvox_1over2()
    assert np.array_equal(r, np.array([
        [0,0,0,0],
        [0,11,18,0],
        [0,21,24,0],
        [0,0,0,0]
    ]))

    upsample_rate = 2
    device = torch.device('cpu')
    asymmetry = True

    input = np.random.randint(0,2,(128,128,128))
    print(get_voxel_bbox(input))  # due to randomness it will probably be the whole box .. i.e. 128/2