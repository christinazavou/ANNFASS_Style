import scipy.spatial.distance  # THIS IS COSINE DISTANCE. I.E. DIFFERENT FROM COSINE SIMILARITY;
# I THINK Cosine Distance=1−Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# in contrastive loss, we are given an array X and an array X',
# and the pairs with same indices in X,X' are considered positive,
# all the rest are considered negative

def cosine_similarity_check():

    def _cosine_similarity(a, b):
        if a.shape[0] == 1 and b.shape[0] == 1:
            return torch.mm(a, torch.t(b)) / (torch.norm(a) * torch.norm(b))
        elif a.shape[1] == 1 and b.shape[1] == 1:
            return torch.mm(torch.t(a), b) / (torch.norm(a) * torch.norm(b))
        else:
            raise Exception("shapes?")

    z1 = np.array([[1.,1,1,1]])
    z2 = np.array([[2.,2,2,2]])
    z3 = np.array([[3.,3,3,3]])
    z1, z2, z3 = torch.from_numpy(z1), torch.from_numpy(z2), torch.from_numpy(z3)

    print(torch.nn.CosineSimilarity()(z1, z2), _cosine_similarity(z1, z2))
    print(torch.nn.CosineSimilarity()(z1, z3), _cosine_similarity(z1, z3))

    z1 = np.array([[1.,6,1,6]])
    z2 = np.array([[2.,2,2,2]])
    z1, z2 = torch.from_numpy(z1), torch.from_numpy(z2)

    print(torch.nn.CosineSimilarity()(z1, z2), _cosine_similarity(z1, z2))


def exp_check():  # gives values in (0,infinity)
    import math
    print(math.exp(-2), math.exp(-1), math.exp(0), math.exp(0.5), math.exp(1), math.exp(2))


def log_check():  # gives values in (-infinity, 0) for x < 1, gives 0 for x = 1, gives values in (0,1) for x > 1
    import math
    print(math.log(0.5), math.log(1), math.log(2))


class contrastive_loss(nn.Module):
    def __init__(self, tau=1, normalize=False):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss


def _similarity_mat_nt_xent(a, b):
    a_norm = torch.norm(a, dim=1).reshape(-1,1)
    a_cap = torch.div(a, a_norm)

    b_norm = torch.norm(b, dim=1).reshape(-1,1)
    b_cap = torch.div(b, b_norm)

    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)

    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    return sim


def original_nt_xent(a, b, tau=0.5):
    # same indices of a,b correspond to positive pairs. all rest are considered negative
    a_norm = torch.norm(a, dim=1).reshape(-1,1)
    a_cap = torch.div(a, a_norm)

    b_norm = torch.norm(b, dim=1).reshape(-1,1)
    b_cap = torch.div(b, b_norm)

    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)

    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    # sim_a1_a1 sim_a1_a2 sim_a1_a3 sim_a1_b1 sim_a1_b2 sim_a1_b3
    # sim_a2_a1 sim_a2_a2 sim_a2_a3 sim_a2_b1 sim_a2_b2 sim_a2_b3
    # sim_a3_a1 ...
    # sim_b1_a1 sim_b1_a2 sim_b1_a3 sim_b1_b1 sim_b1_b2 sim_b1_b3
    # sim_b2_a1 ...
    # sim_b3_a1 ...
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)

    sim_by_tau = torch.div(sim, tau)
    exp_sim_by_tau = torch.exp(sim_by_tau)

    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)  # to be removed from sum_of_rows (k=i in definition's denominator)

    # as said in paper l_i_j is different from l_j_i, thus calculate loss for pairs:
    # a1, b1
    # a2, b2
    # a3, b3
    # b1, a1
    # b2, a2
    # b3, a3
    # WEIRD !!!!!
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = - torch.log(num_by_den)
    return torch.mean(neglog_num_by_den)


def nt_xent_vx1_vx2(query: torch.FloatTensor,
                    positive: torch.FloatTensor,
                    negatives: torch.FloatTensor,
                    tau=0.5):
    assert tau > 0
    assert query.shape[1] == positive.shape[1] == negatives.shape[1]
    assert query.shape[0] == positive.shape[0] == 1
    assert len(query.shape) == len(positive.shape) == len(negatives.shape) == 2
    exp_num = torch.exp(torch.nn.CosineSimilarity()(query, positive) / tau)
    exp_den = torch.exp(torch.nn.CosineSimilarity()(query, negatives) / tau)
    return - torch.log(exp_num / torch.sum(exp_den))


def nt_xent_vx1_vx2_negatives_weighted(query: torch.FloatTensor,
                                        positive: torch.FloatTensor,
                                        negatives: torch.FloatTensor,
                                        tau=0.5):
    assert tau > 0
    assert query.shape[1] == positive.shape[1] == negatives.shape[1]
    assert query.shape[0] == positive.shape[0] == 1
    assert len(query.shape) == len(positive.shape) == len(negatives.shape) == 2
    exp_num = torch.exp(torch.nn.CosineSimilarity()(query, positive) / tau)
    exp_den = torch.exp(torch.nn.CosineSimilarity()(query, negatives) / tau)
    sum_exp_den = torch.sum(exp_den)
    if negatives.shape[0] == 1:
        sum_exp_den = sum_exp_den * 2
    return - torch.log(exp_num / sum_exp_den)


def triplet_loss(query: torch.FloatTensor,
                 positive: torch.FloatTensor,
                 negative: torch.FloatTensor,
                 margin=1.):
    assert query.shape == positive.shape == negative.shape
    # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    # loss = triplet_loss(query, positive, negative)

    loss = (1 - torch.nn.CosineSimilarity()(query, positive)) - (1 - torch.nn.CosineSimilarity()(query, negative)) + margin
    loss = F.relu(loss)
    loss = torch.mean(loss)
    return loss


def check_1():

    z1 = np.array([[1.,1,1,1]])
    z2 = np.array([[1.,1,1,1]])
    z3 = np.array([
        [2,2,2,2],[3,3,3,3]
    ])
    z1, z2, z3 = torch.from_numpy(z1), torch.from_numpy(z2), torch.from_numpy(z3)
    # similarity of positive pair and negative pairs are the same .. so should be moderate loss ...
    print(nt_xent_vx1_vx2(z1, z2, z3, tau=0.5),
          nt_xent_vx1_vx2(z1, z2, z3, tau=0.9),
          nt_xent_vx1_vx2(z1, z2, z3, tau=0.1))

    z1 = np.array([[1.,1,1,1]])
    z2 = np.array([[2.,2,2,2]])
    z3 = np.array([
        [1.,1,1,1],[3.,3,3,3]
    ])
    # similarity of positive pair and negative pairs are the same and equal to above example so loss must be same ...
    z1, z2, z3 = torch.from_numpy(z1), torch.from_numpy(z2), torch.from_numpy(z3)
    print(nt_xent_vx1_vx2(z1, z2, z3, tau=0.5),
          nt_xent_vx1_vx2(z1, z2, z3, tau=0.9),
          nt_xent_vx1_vx2(z1, z2, z3, tau=0.1))

    z1 = np.array([
        [1., 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]
    ])
    z2 = np.array([
        [1., 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]
    ])
    z1 = torch.from_numpy(z1)
    z2 = torch.from_numpy(z2)
    # all these have same cosine similarity thus loss is moderate (to big)
    print("\n=======================CHECK 1 all same original_nt_xent=============================\n")
    print(original_nt_xent(z1, z2), contrastive_loss(normalize=True)(z1, z2))


def check_2():

    z1 = np.array([[1.,2,3,4]])
    z2 = np.array([[1.,2,3,4]])
    z3 = np.array([
        [20.,10,20,10],[30.,40,30,40]
    ])
    z1, z2, z3 = torch.from_numpy(z1), torch.from_numpy(z2), torch.from_numpy(z3)
    print(nt_xent_vx1_vx2(z1, z2, z3, tau=0.5),
          nt_xent_vx1_vx2(z1, z2, z3, tau=0.9),
          nt_xent_vx1_vx2(z1, z2, z3, tau=0.1))

    z1 = np.array([[1.,2,3,4]])
    z2 = np.array([[20.,1,20,1]])
    z3 = np.array([
        [1.,2,3,4],[30.,40,30,40]
    ])
    z1, z2, z3 = torch.from_numpy(z1), torch.from_numpy(z2), torch.from_numpy(z3)
    print(nt_xent_vx1_vx2(z1, z2, z3, tau=0.5),
          nt_xent_vx1_vx2(z1, z2, z3, tau=0.9),
          nt_xent_vx1_vx2(z1, z2, z3, tau=0.1))

    z1 = np.array([[1.,2,3,4]])
    z2 = np.array([[20.,10,20,10]])
    z3 = np.array([
        [30.,40,30,40]
    ])
    z1, z2, z3 = torch.from_numpy(z1), torch.from_numpy(z2), torch.from_numpy(z3)
    print(nt_xent_vx1_vx2(z1, z2, z3, tau=0.5),
          nt_xent_vx1_vx2(z1, z2, z3, tau=0.9),
          nt_xent_vx1_vx2(z1, z2, z3, tau=0.1))

    z1 = np.array([
        [1.,2,3,4], [20.,10,20,10], [30.,40,30,40]
    ])
    z2 = np.array([
        [1.,2,3,4], [20.,10,20,10], [30.,40,30,40]
    ])
    z1 = torch.from_numpy(z1)
    z2 = torch.from_numpy(z2)
    print(original_nt_xent(z1, z2), contrastive_loss(normalize=True)(z1, z2))

    z1 = np.array([
        [1.,2,3,4], [2.,10,200,1], [30.,4,30,400]
    ])
    z2 = np.array([
        [50.,10,50,10], [60.,40,60,40], [10.,2,30,4]
    ])
    z1 = torch.from_numpy(z1)
    z2 = torch.from_numpy(z2)
    print(original_nt_xent(z1, z2), contrastive_loss(normalize=True)(z2, z1))

    # COSINE SIMILARITIES == > dont care about absolute numbers but about vector directions ;)
    z1 = np.array([
        [1.,1,1,1], [100.,100,100,100], [300.,300,300,300]
    ])
    z2 = np.array([
        [1.,1,1,1], [100.,100,100,100], [300.,300,300,300]
    ])
    z1 = torch.from_numpy(z1)
    z2 = torch.from_numpy(z2)
    print(original_nt_xent(z1, z2), contrastive_loss(normalize=True)(z2, z1))

    # COSINE SIMILARITIES == > dont care about absolute numbers but about vector directions ;)
    z1 = np.array([
        [1.,1,1,1], [0.1,0.0,0,0], [-1.,-1,-1,-1]
    ])
    z2 = np.array([
        [1.,1,1,1], [0.1,0,0,0], [-1.,-1,-1,-1]
    ])
    z1 = torch.from_numpy(z1)
    z2 = torch.from_numpy(z2)
    print(original_nt_xent(z1, z2), contrastive_loss(normalize=True)(z2, z1))
    print(original_nt_xent(z1, z2, tau=0.5), original_nt_xent(z1, z2, tau=0.9), original_nt_xent(z1, z2, tau=0.1))


def check_3():
    print("\n========================CHECK WEIGHTED NTXENT ==============================\n")
    z1 = np.array([[1., 1, 1, 1]])
    z2 = np.array([[0.1, 0.1, 0, 0]])
    z3 = np.array([[-1., 2, 3, 0], [-1., 2, 3, 0]])
    z1 = torch.from_numpy(z1)
    z2 = torch.from_numpy(z2)
    z3 = torch.from_numpy(z3)
    print(nt_xent_vx1_vx2(z1, z2, z3))
    print(nt_xent_vx1_vx2_negatives_weighted(z1, z2, z3))
    z3 = np.array([[-1., 2, 3, 0]])
    z3 = torch.from_numpy(z3)
    print(nt_xent_vx1_vx2(z1, z2, z3))
    print(nt_xent_vx1_vx2_negatives_weighted(z1, z2, z3))
    print("\n==========================CHECK TRIPLET LOSS ================================\n")
    print(triplet_loss(z1, z2, z3))

    z1 = np.array([[1., 5, 20, 40]])
    z2 = np.array([[0.1, 0.1, 0.1, 0.1]])
    z3 = np.array([[1., 2.5, 10., 20]])
    z1 = torch.from_numpy(z1)
    z2 = torch.from_numpy(z2)
    z3 = torch.from_numpy(z3)
    print(triplet_loss(z1, z2, z3))
    z1 = np.array([[1., 5, 20, 40]])
    z2 = np.array([[1., 2.5, 10., 20]])
    z3 = np.array([[0.1, 0.1, 0.1, 0.1]])
    z1 = torch.from_numpy(z1)
    z2 = torch.from_numpy(z2)
    z3 = torch.from_numpy(z3)
    print(triplet_loss(z1, z2, z3))

    z1 = np.array([[1., 5, 20, 40], [1., 5, 20, 40], [1., 5, 20, 40]])
    z2 = np.array([[1., 2.5, 10., 20], [1., 2.5, 10., 20], [1., 2.5, 10., 20]])
    z3 = np.array([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
    z1 = torch.from_numpy(z1)
    z2 = torch.from_numpy(z2)
    z3 = torch.from_numpy(z3)
    print(triplet_loss(z1, z2, z3))


if __name__ == '__main__':
    check_3()
    exit()
    z1 = np.array([
        [1., 1, 1, 1], [0.1, 0.0, 0, 0], [-1., -1, -1, -1]
    ])
    z2 = np.array([
        [1., 1, 1, 1], [0.2, 0, 0, 0], [-1.5, -1.8, -1, -1]
    ])
    print("\n=======================SCIPY COSINE CHECK=============================\n")
    print(cosine_similarity(z1, z2))
    print(cosine_similarity(z1[0].reshape([1,-1]), z1[1].reshape([1,-1])),
          cosine_similarity(z1[0].reshape([1,-1]), z1[2].reshape([1,-1])),
          cosine_similarity(z2[0].reshape([1,-1]), z2[2].reshape([1,-1])))
    print(cosine_similarity(z1[0].reshape([1,-1]), z2[0].reshape([1,-1])),
          cosine_similarity(z1[1].reshape([1,-1]), z2[1].reshape([1,-1])),
          cosine_similarity(z1[2].reshape([1,-1]), z2[2].reshape([1,-1])))
    print("\n=====================NUMPY VECTOR NORM CHECK==========================\n")
    print(np.linalg.norm(z1), np.linalg.norm(z1, axis=1))
    print(z1/np.linalg.norm(z1), np.divide(z1, np.linalg.norm(z1, axis=1).reshape([-1,1])))

    print("\n=================NUMPY _similarity_mat_nt_xent ? =====================\n")
    z1_z2 = np.concatenate([z1, z2])
    print(cosine_similarity(z1_z2, z1_z2))

    print("\n=======================TORCH COSINE CHECK=============================\n")
    z1 = torch.from_numpy(z1)
    z2 = torch.from_numpy(z2)
    print(torch.nn.CosineSimilarity()(z1, z2))
    print("\n====================_similarity_mat_nt_xent==========================\n")
    print(_similarity_mat_nt_xent(z1, z2))
    print(_similarity_mat_nt_xent(z1[0].view([1, -1]), z2[0].view([1, -1])))

    # print("\n====================COSINE SIMILARITY CHECK==========================\n")
    # cosine_similarity_check()
    # print("\n=========================EXPONENT CHECK==============================\n")
    # exp_check()
    # print("\n========================LOGARITHM CHECK==============================\n")
    # log_check()
    print("\n===========================CHECK 1===================================\n")
    check_1()
    print("\n===========================CHECK 2===================================\n")
    check_2()
