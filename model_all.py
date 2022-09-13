import numpy as np
import scipy.sparse as sp
import torch
import torch_sparse
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.utils import negative_sampling
from model_other import CNN, CNN1, MLP
from model_ba import BA

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

    coords_trans, values_trans = torch_sparse.transpose(coords, values, shape[0], shape[1])
    mat_trans = torch.sparse.FloatTensor(coords_trans, values_trans, (shape[1],shape[0]))
    mat_trans = mat_trans.to(device)

    return mat, mat_trans

def graph_inception_unit(network, Korder, weight):
    temp_Korder = Korder
    temp_Korder = torch.sparse.mm(network, temp_Korder)
    relfeature = weight(temp_Korder)

    return relfeature, temp_Korder

def Multiply(network, l_feat, r_feat, weight):
    l_feat = torch.mm(network, l_feat)
    mul = torch.mul(l_feat,r_feat)
    relfeature = weight(mul)

    return relfeature, mul

class layers(torch.nn.Module):
    def __init__(self):
        super(layers,self).__init__()
        self.conv1 = graph_inception(1, 64, 256)
        self.conv2 = graph_inception(1, 256, 128)
    def forward(self, l_feat, r_feat, network, weight1, weight2, weight3, weight4):
        x1 = self.conv1(l_feat, r_feat, network, weight1, weight2)
        y1 = torch.relu(x1[0]) 
        z1 = torch.relu(x1[1])
        x2 = self.conv2(y1, z1, network, weight3, weight4)
        y2 = torch.relu(x2[0])
        z2 = torch.relu(x2[1]) 

        return y2

class graph_inception(torch.nn.Module):
    def __init__(self, hop, inputdims, outputdims):
        super(graph_inception, self).__init__()
        self.hop = hop
        self.inputdims = inputdims
        self.outputdims = outputdims
    def forward(self, l_feat, r_feat, network, weight1, weight2):
        Korder_list = [[l_feat],[r_feat]]
        convresults = [[] for _ in range(2)]
        l_index = 0
        r_index = 1
        network_l, network_r = sparse_to_matrix(network)
        for i in range(self.hop):
            temp_Korders = [[] for _ in range(2)]
            # l_node
            x0 = F.dropout(Korder_list[r_index][i],training=self.training)
            l_reFeat, l_Korder = graph_inception_unit(network_l, x0, weight1)
            convresults[l_index].append(l_reFeat)
            temp_Korders[l_index].append(l_Korder)

            # r_node
            x1 = F.dropout(Korder_list[l_index][i], training=self.training)
            r_reFeat, r_Korder = graph_inception_unit(network_r, x1, weight1)
            convresults[r_index].append(r_reFeat)
            temp_Korders[r_index].append(r_Korder)

            # Dot product feature with neighbors
            l_mul, l_Korder_mul = Multiply(network_l, x0, x1, weight2)
            r_mul, r_Korder_mul = Multiply(network_r, x1, x0, weight2)
            convresults[l_index].append(l_mul)
            convresults[r_index].append(r_mul)
            temp_Korders[l_index].append(l_Korder_mul)
            temp_Korders[r_index].append(r_Korder_mul)

            for j in range(len(temp_Korders)):
                if len(temp_Korders[j]) == 1:
                    temp = temp_Korders[j][0]
                else:
                    temp = temp_Korders[j][0]
                    for k in range(1, len(temp_Korders[j])):
                        temp = temp + temp_Korders[j][k]
                Korder_list[j].append(temp)

        final_convs = []
        for convresult in convresults:
            if len(convresult) == 1:
                final_convs.append(convresult[0])
            else:
                temp = convresult[0]
                for z in range(1, len(convresult)):
                    temp = temp + convresult[z]
                final_convs.append(temp)

        return final_convs

class attention(torch.nn.Module):
    def __init__(self, inputdims, outputdims):
        super(attention, self).__init__()
        self.inputdims = inputdims
        self.outputdims = outputdims

        self.lin1 = Linear(self.inputdims, self.outputdims)
        self.lin2 = Linear(self.inputdims, self.outputdims)
        self.lin3 = Linear(self.inputdims, self.outputdims)
    def forward(self, feat):
        feat = F.dropout(feat, p=0.2, training=self.training)
        q = self.lin1(feat)
        k = self.lin2(feat)
        v = self.lin3(feat)
        att = torch.mm(torch.softmax(torch.mm(q, torch.transpose(k, 0, 1))/torch.sqrt(torch.tensor(self.outputdims, dtype=torch.float)), dim=1), v)

        return att

class Net(torch.nn.Module):
    def __init__(self,l_feat, r_feat, network1, network2, hop, inputdims, hiddendims, outputdims, edge_index, edge_index1):
        super(Net, self).__init__()
        self.l_feat = l_feat
        self.r_feat = r_feat
        self.network1 = network1
        self.network2 = network2
        self.hop = hop
        self.edge_index = edge_index
        self.edge_index1 = edge_index1

        self.gcn1 = layers()
        self.gcn2 = layers()
        self.gcn3 = layers()
        self.ba = BA(inputdims, outputdims)
        self.att = attention(outputdims, outputdims)
        self.cnn1 = CNN()
        self.cnn2 = CNN()
        self.cnn3 = CNN()
        self.cnn4 = CNN1()
        self.mlp = MLP()
        self.lin1 = Linear(32, 64)
        self.lin2 = Linear(2, 19)
        self.lin3 = Linear(inputdims, hiddendims)
        self.lin4 = Linear(inputdims, hiddendims)
        self.lin5 = Linear(hiddendims, outputdims)
        self.lin6 = Linear(hiddendims, outputdims)
    def forward(self):
        # gene-gene network
        x = self.gcn1(self.l_feat[0], self.r_feat[0], self.network1[0], self.lin3, self.lin4, self.lin5, self.lin6)
        x = self.att(x) + x

        # # gene-outlying network
        r_feature1 = self.lin1(self.r_feat[1])      # pan-cancer
        # r_feature1 = self.lin2(self.r_feat[1])      # specific cancer,e.g.luad
        y1 = self.gcn2(self.l_feat[1], r_feature1, self.network1[1], self.lin3, self.lin4, self.lin5, self.lin6)
        feat = torch.cat((self.l_feat[0],r_feature1),dim=0)
        y2 = self.ba(feat, self.network2[0], self.network2[1])[0:13627,:]
        y2 = torch.relu(y2)
        y = (1-0.2) * y1 + 0.2 * y2
        y = self.att(y) + y

        # gene-miRNA network
        z = self.gcn3(self.l_feat[2], self.r_feat[2], self.network1[2], self.lin3, self.lin4, self.lin5, self.lin6)
        z = self.att(z) + z

        # Feature fusion
        # 1D convolution module
        a = self.cnn1(x)
        b = self.cnn2(y)
        c = self.cnn3(z)

        # 2D convolution module
        feat = []
        feat.append(a.unsqueeze(0))
        feat.append(b.unsqueeze(0))
        feat.append(c.unsqueeze(0))
        feat = torch.cat(feat,dim=0)
        w1 = self.cnn4(feat)

        # mlp module
        w0 = self.l_feat[0]
        w2 = self.mlp(w0)
        w = w1 + w2         # final feature

        # Logistic Regression Module input features
        w3 = torch.cat((a, b, c, w1, w2),dim=1)

        # Calculate the reconstruction network loss
        pos_loss = -torch.log(torch.sigmoid((a[self.edge_index[0]] * a[self.edge_index[1]]).sum(dim=1)) + 1e-15).mean()
        neg_edge_index = negative_sampling(self.edge_index1, 13627, 504378)
        neg_loss = -torch.log(
            1 - torch.sigmoid((a[neg_edge_index[0]] * a[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()
        r_loss = (pos_loss + neg_loss)/2

        return w, w1, r_loss, w3
