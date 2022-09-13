import numpy as np
import torch
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_bilinear(adj):
    adj_loop = adj + torch.eye(adj.shape[0])
    D_all = adj_loop.sum(1)
    N_all = 0.5 * torch.mul(D_all, D_all - np.ones((adj.shape[0])))
    N_all = torch.diag(1. / N_all)
    N_all[np.isinf(N_all)] = 0.
    N_all = sp.coo_matrix(N_all)
    adj_loop = sp.coo_matrix(adj_loop)

    return adj_loop, N_all

if __name__ == '__main__':
    adj1 = sp.load_npz("./PP.adj.npz").toarray()
    adj1 = np.where(adj1 > 0, 1, 0)
    adj2 = np.zeros((adj1.shape[0], adj1.shape[0]))
    adj3 = np.zeros((adj1.shape[1], adj1.shape[1]))
    adj4 = np.transpose(adj1)

    # concatenated into a large matrix
    adj_2_4 = np.vstack((adj2, adj4))
    adj_1_3 = np.vstack((adj1, adj3))
    adj = np.hstack((adj_2_4, adj_1_3))
    adj = torch.Tensor(adj)

    x,y = preprocess_bilinear(adj)
    sp.save_npz("O.adj_loop.npz", x)
    sp.save_npz("O.N_all.npz", y)