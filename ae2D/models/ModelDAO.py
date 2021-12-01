import os
import shutil

import torch


class ModelDAO:

    def __init__(self, model, optimizer, checkpoint_dir):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.best_model_path = os.path.join(checkpoint_dir, "best.pt")

    def checkpoint(self, epoch, valid_loss):
        return {
            'epoch': epoch,
            'valid_loss_min': valid_loss,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def save_ckp(self, epoch, valid_loss, is_best):
        state = self.checkpoint(epoch, valid_loss)
        f_path = os.path.join(self.checkpoint_dir, "{}.pt".format(epoch))
        print("Saving {} model {}".format("best" if is_best else "", f_path))
        torch.save(state, f_path)
        if is_best:
            shutil.copyfile(f_path, self.best_model_path)

    def load_ckp(self, checkpoint_name):
        checkpoint_fpath = os.path.join(self.checkpoint_dir, checkpoint_name)
        print("Loading model {}".format(checkpoint_fpath))
        checkpoint = torch.load(checkpoint_fpath)

        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint['epoch'] + 1, checkpoint['valid_loss_min']
