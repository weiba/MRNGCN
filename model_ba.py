import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Linear
import scipy.sparse as sp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sparse_to_tuple(matrix):
    if not sp.isspmatrix_coo(matrix):
        matrix=matrix.tocoo()
        coords = np.vstack((matrix.row, matrix.col)).transpose()
        values = matrix.data
        shape = matrix.shape
        return coords, values, shape

def sparse_to_matrix(matrix):
    tmp = sp.csr_matrix(matrix)
    coords, values, shape = sparse_to_tuple(tmp)
    coords = torch.LongTensor(coords.transpose())
    values = torch.FloatTensor(values)
    mat = torch.sparse.FloatTensor(coords, values, shape)
    mat = mat.to(device)

    return mat

def BILinear_pooling(adj_,XW):
    #step1 sum_squared
    sum = torch.mm(adj_, XW)
    sum_squared = torch.mul(sum,sum)

    #step2 squared_sum
    squared = torch.mul(XW, XW)
    squared_sum = torch.mm(adj_, squared)

    #step3
    new_embedding = 0.5 * (sum_squared - squared_sum)

    return new_embedding

class BA(torch.nn.Module):
    def __init__(self,input_dim, output_dim):
        super(BA, self).__init__()
        self.lin1 = Linear(input_dim,output_dim)

    def forward(self, feat, adj_loop, diag_mat):
        adj_loop = sparse_to_matrix(adj_loop)
        diag_mat = sparse_to_matrix(diag_mat)
        x = F.dropout(feat,training=self.training)
        pre_sup = self.lin1(x)
        x = BILinear_pooling(adj_loop, pre_sup)
        out_bi = torch.mm(diag_mat, x)

        return out_bi

