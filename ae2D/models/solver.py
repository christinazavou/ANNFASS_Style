import os
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from ae2D.models.ModelDAO import ModelDAO
from ae2D.utils import get_device
from ae2D.utils.visual import show_initial_and_reconstructed_batch


class Solver:

    def __init__(self, model, epochs, service_log, save_freq=25):
        self.device = get_device()
        model.to(self.device)
        self.model = model
        self.criterion = nn.MSELoss()
        # criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.epochs = epochs
        self.service_log = service_log
        self.writer = SummaryWriter(service_log)
        self.m_dao = ModelDAO(self.model, self.optimizer, service_log + "/checkpoints")
        if os.path.exists(os.path.join(service_log, "checkpoints", "best.pt")):
            self.epoch, self.min_val_loss = self.m_dao.load_ckp("best.pt")
        else:
            self.epoch, self.min_val_loss = 1, None
        self.save_freq = save_freq

    def save_one_encoding_per_test_batch(self, key, encodings, labels, encodings_dir):
        average_encoding = encodings.mean(0)
        assert np.array_equal(labels.mean(0).astype(np.int), labels[0])
        labels = labels[0]
        encoding_path = os.path.join(encodings_dir, '{}.npy'.format(key))
        os.makedirs(os.path.dirname(encoding_path), exist_ok=True)
        with open(encoding_path, 'wb') as f:
            np.save(f, average_encoding)
        with open(encoding_path.replace(".npy", "_labels.npy"), 'wb') as f:
            np.save(f, labels)

    def save_encodings(self, key, encodings, labels, encodings_dir):
        for idx, (encoding, label) in enumerate(zip(encodings, labels)):
            encoding_path = os.path.join(encodings_dir, '{}_{}.npy'.format(key, idx))
            os.makedirs(os.path.dirname(encoding_path), exist_ok=True)
            with open(encoding_path, 'wb') as f:
                np.save(f, encoding)
            with open(encoding_path.replace(".npy", "_labels.npy"), 'wb') as f:
                np.save(f, label)

    def generate_encodings(self, encodings_loader, encodings_dir):
        print("Start generating encodings")
        assert self.min_val_loss is not None, "Can't generate encodings from non-trained model"

        os.makedirs(encodings_dir, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(encodings_loader):
                batch_features = batch['image']
                labels = batch['labels']
                batch_key = batch['key']
                t_batch_features = batch_features.type(torch.FloatTensor).to(self.device)
                encodings = self.model.encoder(t_batch_features).cpu().numpy()
                labels = labels.cpu().numpy()
                max_encodings = encodings.max(axis=(2, 3))
                self.save_encodings(batch_key, max_encodings, labels, join(encodings_dir, "z_dim_max", "as_is"))
                avg_encodings = encodings.mean(axis=(2, 3))
                self.save_encodings(batch_key, avg_encodings, labels, join(encodings_dir, "z_dim_avg", "as_is"))
                all_encodings = encodings.reshape((encodings.shape[0], -1))
                self.save_encodings(batch_key, all_encodings, labels, join(encodings_dir, "z_dim_all", "as_is"))

        return self

    def evaluate_epoch(self, val_loader, epoch, iteration):
        self.model.eval()
        with torch.no_grad():
            test_size = len(val_loader)
            # test_loader = iter(test_loader)
            loss = 0
            for idx, batch in enumerate(val_loader):
                batch_features = batch['image']
                t_batch_features = batch_features.type(torch.FloatTensor).to(self.device)
                outputs = self.model(t_batch_features)
                val_loss = self.criterion(outputs, t_batch_features)
                curr_loss = float(val_loss)
                self.writer.add_scalar('Loss/val', curr_loss, iteration)
                loss += curr_loss * t_batch_features.size(0)
                if idx % 100 == 0:
                    grid1, grid2 = show_initial_and_reconstructed_batch(batch_features, outputs, epoch, idx, False)
                    self.writer.add_image('epoch {}/batch {}/init'.format(epoch, idx), grid1)
                    self.writer.add_image('epoch {}/batch {}/reconstructed'.format(epoch, idx), grid2)
            loss = loss / test_size
            print('Epoch: {} \tEval Loss: {:.6f}'.format(epoch, loss))
            if self.min_val_loss is None or (loss < self.min_val_loss):
                self.min_val_loss = loss
                self.m_dao.save_ckp(epoch, loss, True)
            else:
                if epoch % self.save_freq == 0:
                    self.m_dao.save_ckp(epoch, loss, False)

    def run(self, train_loader, val_loader):
        print("Start training")
        iteration = 0
        for epoch in tqdm(range(self.epoch, self.epochs + 1)):
            self.model.train()
            train_loss = 0.0
            for data in train_loader:
                iteration += 1
                images = data['image']
                images = images.type(torch.FloatTensor).to(self.device)
                # ===================forward=====================
                output = self.model(images)
                loss = self.criterion(output, images)
                self.writer.add_scalar('Loss/train', loss.item(), iteration)
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += float(loss) * images.size(0)
            # ===================log========================
            train_loss = train_loss / len(train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
            self.evaluate_epoch(val_loader, epoch, iteration)
