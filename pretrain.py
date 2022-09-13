import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from model_pretrain import pretrain

EPOCH = 1000        # pan-cancer
# EPOCH = 2000        # specific cancer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(path):
    # load network
    adj = sp.load_npz(path + "PR.adj.npz")
    network = adj.tocsc()

    # load node features
    feat1 = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]
    l_feature = torch.Tensor(feat1).to(device)
    feat2 = pd.read_csv(path + "R.feat-final.csv", sep=",").values[:, 1:]
    r_feature = torch.Tensor(feat2).to(device)

    # load edge
    pos_edge = np.array(np.loadtxt(path + "PR_pos.txt").transpose())
    pos_edge = torch.from_numpy(pos_edge).long()

    neg_edge = np.array(np.loadtxt(path + "PR_neg.txt").transpose())# gene-miRNA网络
    neg_edge = torch.from_numpy(neg_edge).long()


    return network, l_feature, r_feature, pos_edge, neg_edge

def train():
    model.train()
    optimizer.zero_grad()
    loss, l_node, r_node = model()

    print(loss)
    loss.backward()
    optimizer.step()

    return l_node, r_node

if __name__ == '__main__':
    path = "./data/pan-cancer/"     # pan-cancer
    # path = "./data/LUAD/cancer name/"       # specific cancer,e.g.luad
    network, feature1, feature2, pos_edge, neg_edge = load_data(path)       # pan-cancer
    # network, feature1, feature2, pos_edge, neg_edge = load_data(path)      # specific cancer
    model = pretrain(feature1, feature2, network, 1, 64, 256, 64, pos_edge, neg_edge).to(device)  # hop = 1
    # model = pretrain(feature1, feature2, network, 1, 19, 128, 19, pos_edge, neg_edge).to(device)  # specific cancer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train
    for epoch in range(1, EPOCH + 1):
        print(epoch)
        l_node, r_node = train()

    # Save node features
    P_feat = l_node.cpu().detach().numpy()
    R_feat = r_node.cpu().detach().numpy()
    pd.DataFrame(P_feat).to_csv(path + "P.feat-pre.csv")
    pd.DataFrame(R_feat).to_csv(path + "R.feat-pre.csv")
