from unittest import TestCase
import numpy as np
import torch

from utils.nt_xent import triplet_loss


class Test(TestCase):
    def test_triplet_loss(self):
        q = np.array([[0, 1., 2.]])
        p = np.array([[0., 1., 2.]])
        n = np.array([[100.,-100.,2.]])
        q, p, n = torch.tensor(q, requires_grad=True), torch.tensor(p, requires_grad=True), torch.tensor(n, requires_grad=True)
        loss = triplet_loss(q, p, n)
        print(loss)
        loss.backward()

        n = np.array([[1.,10.,2.]])
        q, p, n = torch.tensor(q, requires_grad=True), torch.tensor(p, requires_grad=True), torch.tensor(n, requires_grad=True)
        loss = triplet_loss(q, p, n)
        print(loss)
        loss.backward()
