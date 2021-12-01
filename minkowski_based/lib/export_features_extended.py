import logging
import os
import sys

import numpy as np
import torch
from scipy import spatial

sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.

from lib.utils import get_torch_device, count_parameters, Timer
from lib.dataset_extended import initialize_data_loader
from models import load_model
from lib.pc_utils import read_plyfile

from MinkowskiEngine import SparseTensor


def write_ply_with_features(ply_fn, vertices, features, labels=None):
    os.makedirs(os.path.dirname(ply_fn), exist_ok=True)

    # Create header
    header = 'ply\n' \
             'format ascii 1.0\n' \
             'element vertex ' + str(len(vertices)) + '\n' \
                                                      'property float x\n' \
                                                      'property float y\n' \
                                                      'property float z\n'
    for i in range(features.shape[1]):
        header += 'property float feature{}\n'.format(i)
    if labels is not None:
        header += 'property int label\n'
    header += 'end_header\n'

    with open(ply_fn, 'w') as f_ply:
        # Write header
        f_ply.write(header)

        # Write vertices + normals + label
        for idx, vertex in enumerate(vertices):
            row = ' '.join(vertex.astype(str))
            row = ' '.join(features[idx].astype(str))
            if labels is not None:
                row += ' ' + str(labels[idx])
            row += "\n"
            f_ply.write(row)


def get_feats(DatasetClass, config):
    """ Export per point output features """

    # Get data loaders
    dataset_dict = load_dataset(DatasetClass, config)

    # Set input and output features
    if dataset_dict["test_split"].dataset.NUM_IN_CHANNEL is not None:
        num_in_channel = dataset_dict["test_split"].dataset.NUM_IN_CHANNEL
    else:
        num_in_channel = 3  # RGB color
    num_labels = dataset_dict["test_split"].dataset.NUM_LABELS

    logging.info('===> Building model')
    NetClass = load_model(config.model)
    model = NetClass(num_in_channel, num_labels, config)
    target_device = get_torch_device(config.is_cuda)
    model = model.to(target_device)
    logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__, count_parameters(model)))
    logging.info(model)

    # Load weights if specified by the parameter.
    if config.weights.lower() != 'none':
        logging.info('===> Loading weights: ' + config.weights)
        state = torch.load(config.weights)
        # model.load_state_dict(state['state_dict'])

        # load only common layer (i.e. if last layer is not same it wont be loaded and wont complain)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state['state_dict'].items() if
                           k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print("ERROR: No model weights provided")
        exit(-1)

    for split, data_loader in dataset_dict.items():
        feed_forward(model, data_loader, split, config)


def feed_forward(model, data_loader, split, config, weights=None):
    # Init
    device = get_torch_device(config.is_cuda)
    dataset = data_loader.dataset
    global_timer = Timer()

    logging.info('===> Start feed forward - {}'.format(split))

    global_timer.tic()
    data_iter = data_loader.__iter__()
    max_iter = len(data_loader)

    # Fix batch normalization running mean and std
    model.eval()

    # Clear cache (when run in val mode, cleanup training cache)
    torch.cuda.empty_cache()

    # Create save dir
    save_pred_dir = os.path.join(os.path.dirname(config.log_dir), config.save_pred_dir, split)
    save_pred_layer_1_feat_dir = os.path.join(save_pred_dir, "layer_n-1_features")
    save_pred_layer_2_feat_dir = os.path.join(save_pred_dir, "layer_n-2_features")
    save_pred_layer_3_feat_dir = os.path.join(save_pred_dir, "layer_n-3_features")
    os.makedirs(save_pred_dir, exist_ok=True)
    os.makedirs(save_pred_layer_1_feat_dir, exist_ok=True)
    os.makedirs(save_pred_layer_2_feat_dir, exist_ok=True)
    os.makedirs(save_pred_layer_3_feat_dir, exist_ok=True)

    with torch.no_grad():
        for iteration in range(max_iter):
            # Get data
            # coords example: shape [15734,4] with first column being zero and last three columns being
            # the transformed coordinates
            # example row: [0, 32, 14, -25]
            # transformation example: shape [4,4] converted to a list and represents coordinates transformation
            # example with no augmentation and with pixel size 0.01: [100,0,0,0],[0,100,0,0],[0,0,100,0],[0,0,0,1]
            coords, input, target, component_ids, component_names, transformation = data_iter.next()

            # Preprocess input
            if config.normalize_color:
                input[:, :3] = input[:, :3] / 255. - config.color_offset
            sinput = SparseTensor(input, coords).to(device)

            # Feed forward
            soutput, soutput_layer_1, soutput_layer_2, soutput_layer_3 = model(sinput)

            # Undo voxelizer transformation.
            curr_transformation = transformation[0][:16].numpy().reshape(4, 4)

            fullply_f = dataset.data_root / dataset.data_paths[iteration]
            query_pointcloud = read_plyfile(fullply_f)
            # query_xyz has the original points in the point cloud, while origi_coords has the original points
            # of the voxelized model (i.e. more points)
            query_xyz = query_pointcloud[:, :3]

            for sout_layer, save_pred_layer_feat_dir in zip(
                    [soutput_layer_1, soutput_layer_2, soutput_layer_3],
                    [save_pred_layer_1_feat_dir, save_pred_layer_2_feat_dir, save_pred_layer_3_feat_dir]
            ):
                if sout_layer is None:
                    continue
                out_features_layer = sout_layer.F.detach().cpu().numpy()
                coords_layer = sout_layer.C.detach().cpu().numpy()

                # ----- Inverse distance weighting using 4-nn-----

                coords_layer = coords_layer[:, 1:]

                # xyz_layer example: shape [15734, 4] with first three columns being the transformed coordinates
                # and last column being equal to one
                xyz_layer = np.hstack((coords_layer, np.ones((coords_layer.shape[0], 1))))

                # we get original coordinates by applying the inverse transformation
                # example row with above example coords_layer and transformation: [0.32, 0.14, -0.25, 1]
                orig_coords_layer = (np.linalg.inv(curr_transformation) @ xyz_layer.T).T
                orig_coords_layer = orig_coords_layer[:, :3]

                orig_coords_layer_tree = spatial.cKDTree(orig_coords_layer, leafsize=500)

                k, pow = 4, 2
                dist, k_nn = orig_coords_layer_tree.query(query_xyz, k=k)

                dist_pow = dist ** pow
                if not np.amin(dist_pow) > 0.0:
                    print(iteration)
                    dist_pow = np.maximum(dist_pow, 2.0 * np.finfo(np.float32).eps)
                norm = np.sum(1 / dist_pow, axis=1, keepdims=True)
                norm = np.tile(norm, [1, k])
                weight = (1 / dist_pow) / norm
                assert (np.isclose(np.sum(weight, axis=1, keepdims=True), np.ones_like(norm)).all())

                # Export hidden features per point
                filename = '%s.npy' % (os.path.basename(dataset.data_paths[iteration])[:-4])

                feats_layer = np.multiply(weight[..., np.newaxis], out_features_layer[k_nn])

                os.makedirs(os.path.join(save_pred_layer_feat_dir, "weighted_sum"), exist_ok=True)
                os.makedirs(os.path.join(save_pred_layer_feat_dir, "max"), exist_ok=True)

                for component_idx, component in enumerate(component_names):
                    filename = '{}_{}.ply'.format(os.path.basename(dataset.data_paths[iteration])[:-4], component)
                    indices = np.where(component_ids == component_idx)
                    per_point_feat_layer = np.sum(feats_layer, axis=1).astype(np.float32)
                    component_features = per_point_feat_layer[indices]
                    outf = os.path.join(save_pred_layer_feat_dir, "weighted_sum", filename)
                    write_ply_with_features(outf, query_xyz[indices], component_features)
                    per_point_feat_layer = np.max(out_features_layer[k_nn], axis=1).astype(np.float32)
                    component_features = per_point_feat_layer[indices]
                    outf = os.path.join(save_pred_layer_feat_dir, "max", filename)
                    write_ply_with_features(outf, query_xyz[indices], component_features)

            if iteration % config.empty_cache_freq == 0:
                # Clear cache
                torch.cuda.empty_cache()

    global_time = global_timer.toc(False)
    logging.info("Finished feed forward. Elapsed time: {:.4f}".format(global_time))


def load_dataset(DatasetClass, config):
    """ Load test data split """
    test_data_loader = initialize_data_loader(
        DatasetClass,
        config,
        num_workers=config.num_workers,
        phase=config.test_phase,
        augment_data=False,
        shift=False,
        jitter=False,
        rot_aug=False,
        scale=False,
        shuffle=False,
        repeat=False,
        batch_size=1,
        limit_numpoints=False)

    return dict({
        "test_split": test_data_loader
    })
