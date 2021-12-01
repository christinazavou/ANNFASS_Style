from unittest import TestCase
import numpy as np

from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset
from lib.utils import AverageMeter

from config import get_config

import sys
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


class Test(TestCase):

    def train(self, data_loader, val_data_loader, config, transform_data_fn=None):
        # Set up the train flag for batch normalization
        print("set training phase for the model")

        # Configuration
        losses = AverageMeter()

        # Train the network
        print('===> Start training')
        best_val_loss, best_val_loss_iter = np.Inf, 0
        curr_iter, epoch, is_training = 1, 1, True

        data_iter = data_loader.__iter__()
        while is_training:
            for iteration in range(len(data_loader)):
                print("Reset gradients to zero")

                coords, input, target = data_iter.next()
                print("data_iter.next() gives {} (voxelized) coords, merged from {} loaded 3D models.".format(
                    coords.shape[0], data_loader.batch_size))
                print("one feed forward step & (batch) loss calculation")
                batch_loss = 2

                # Compute gradient
                print("Backward step")

                # Update number of steps
                print("Optimizer step update")
                if config.scheduler != "ReduceLROnPlateau":
                    print("Scheduler update if not ReduceLROnPlateau")

                losses.update(batch_loss, target.size(0))

                if curr_iter >= config.max_iter:
                    is_training = False
                    break

                if curr_iter % config.stat_freq == 0 or curr_iter == 1:
                    debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\t".format(
                        epoch, curr_iter, len(data_loader), losses.avg)
                    print(debug_str)
                    # Write logs
                    print("writing training/loss as losses.avg at {} and resetting losses".format(curr_iter))
                    losses.reset()

                # Save current status, save before val to prevent occational mem overflow
                if curr_iter % config.save_freq == 0:
                    print("save model at iter {}".format(curr_iter))

                # Validation
                if curr_iter % config.val_freq == 0:
                    print("Validate model at iter {}".format(curr_iter))

                    # Recover back
                    print("recovering training phase for the model")

                # End of iteration
                curr_iter += 1

            if config.scheduler == "ReduceLROnPlateau":
                print("Scheduler update if ReduceLROnPlateau")
            epoch += 1

        # Explicit memory cleanup
        if hasattr(data_iter, 'cleanup'):
            data_iter.cleanup()

        print("Save final model")

    def test_train(self):

        testargs = ["python test_train_ae.py",
                    "--stylenet_path",
                    "/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_reconstruction_splits/ply_100K/split_train_val_test_debug",
                    "--dataset", "StylenetXYZAEVoxelization0_01Dataset",
                    "--input_feat", "coords",
                    "--batch_size", "4"]
        with patch.object(sys, 'argv', testargs):
            config = get_config()

        DatasetClass = load_dataset(config.dataset)

        train_data_loader = initialize_data_loader(
            DatasetClass,
            config,
            phase=config.train_phase,
            num_workers=config.num_workers,
            augment_data=False,
            shift=config.shift,
            jitter=config.jitter,
            rot_aug=config.rot_aug,
            scale=config.scale,
            shuffle=True,
            repeat=True,
            batch_size=config.batch_size,
            limit_numpoints=config.train_limit_numpoints)

        self.train(train_data_loader, None, config, None)

