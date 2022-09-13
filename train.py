import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
import pickle
import random
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn import metrics
from model_all import Net
from sklearn import linear_model
import torch.backends.cudnn as cudnn

# fixed seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
torch.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True

EPOCH = 1065        # pan-cancer
# EPOCH = 1000        # specific cancer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_label_single(path):
    label = np.loadtxt(path + "label_file-P.txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)
    label_pos = np.loadtxt(path + "pos.txt", dtype=int)
    label_neg = np.loadtxt(path + "neg.txt", dtype=int)


    return Y, label_pos, label_neg

def sample_division_single(pos_label, neg_label, l, l1, l2, i):
    # pos_label：Positive sample index
    # neg_label：Negative sample index
    # l：number of genes
    # l1：Number of positive samples
    # l2：number of negative samples
    # i：number of folds
    pos_test = pos_label[i * l1:(i + 1) * l1]
    pos_train = list(set(pos_label) - set(pos_test))
    neg_test = neg_label[i * l2:(i + 1) * l2]
    neg_train = list(set(neg_label) - set(neg_test))
    indexs1 = [False] * l
    indexs2 = [False] * l
    for j in range(len(pos_train)):
        indexs1[pos_train[j]] = True
    for j in range(len(neg_train)):
        indexs1[neg_train[j]] = True
    for j in range(len(pos_test)):
        indexs2[pos_test[j]] = True
    for j in range(len(neg_test)):
        indexs2[neg_test[j]] = True
    tr_mask = torch.from_numpy(np.array(indexs1))
    te_mask = torch.from_numpy(np.array(indexs2))

    return tr_mask, te_mask

def load_data(path):
    # load network
    network1 = []
    adj1 = sp.load_npz(path + "PP.adj.npz")      # gene-gene network
    adj2 = sp.load_npz(path + "PO.adj.npz")      # gene-outlying gene network
    adj3 = sp.load_npz(path + "PR.adj.npz")      # gene-miRNA network

    network1.append(adj1.tocsc())
    network1.append(adj2.tocsc())
    network1.append(adj3.tocsc())

    # netwroks for bilinear aggregation layer
    network2 = []
    adj4 = sp.load_npz(path + "O.adj_loop.npz")
    adj5 = sp.load_npz(path + "O.N_all.npz")

    network2.append(adj4.tocsc())
    network2.append(adj5.tocsc())

    # load node features
    l_feature = []      # gene
    feat1 = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]
    feat1 = torch.Tensor(feat1).to(device)
    feat2 = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]
    feat2 = torch.Tensor(feat2).to(device)
    feat3 = pd.read_csv(path + "P.feat-pre.csv", sep=",").values[:, 1:]
    feat3 = torch.Tensor(feat3).to(device)

    l_feature.append(feat1)
    l_feature.append(feat2)
    l_feature.append(feat3)

    r_feature = []
    feat4 = pd.read_csv(path + "P.feat-final.csv", sep=",").values[:, 1:]       # gene
    feat4 = torch.Tensor(feat4).to(device)
    feat5 = pd.read_csv(path + "O.feat-final.csv", sep=",").values[:, 1:]       # outlying gene
    feat5 = torch.Tensor(feat5).to(device)
    feat6 = pd.read_csv(path + "R.feat-pre.csv", sep=",").values[:, 1:]         # miRNA
    feat6 = torch.Tensor(feat6).to(device)

    r_feature.append(feat4)
    r_feature.append(feat5)
    r_feature.append(feat6)

    # load edge
    pos_edge = np.array(np.loadtxt(path + "PP_pos.txt").transpose())
    pos_edge = torch.from_numpy(pos_edge).long()

    pb, _ = remove_self_loops(pos_edge)
    pos_edge1, _ = add_self_loops(pb)

    # divisions of ten-fold cross-validation
    with open(path + "/k_sets.pkl", 'rb') as handle:
        k_sets = pickle.load(handle)

    # label
    label = np.loadtxt(path + "label_file.txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)

    return network1, network2, l_feature, r_feature, pos_edge, pos_edge1, k_sets, Y

def LR(train_x, train_y, test_x):
    regr = linear_model.LogisticRegression(max_iter=10000)
    regr.fit(train_x, train_y.ravel())
    pre = regr.predict_proba(test_x)
    pre = pre[:,1]

    return pre

def train(mask, Y):
    model.train()
    optimizer.zero_grad()

    pred, pred1, r_loss, _ = model()
    loss = F.binary_cross_entropy_with_logits(pred[mask], Y[mask])
    loss1 = F.binary_cross_entropy_with_logits(pred1[mask], Y[mask])
    loss = loss + 0.1 * loss1 + 0.01 * r_loss

    loss.backward()
    optimizer.step()

@torch.no_grad()
def test(mask1, mask2, Y):
    model.eval()
    _, _, _, x = model()

    # logistic regression model
    train_x = torch.sigmoid(x[mask1]).cpu().detach().numpy()
    train_y = Y[mask1].cpu().numpy()
    test_x = torch.sigmoid(x[mask2]).cpu().detach().numpy()
    Yn = Y[mask2].cpu().numpy().reshape(-1)
    pred = LR(train_x, train_y, test_x)
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)

    return metrics.roc_auc_score(Yn, pred), area

if __name__ == '__main__':
    time_start = time.time()

    # results
    AUC_test = np.zeros(shape=(10,5))
    AUPRC_test = np.zeros(shape=(10,5))

    # load data
    path = "./data/pan-cancer/"     # pan-cancer
    # path = "./data/LUAD/cancer name/"     # specific cancer,e.g.luad
    network1, network2, l_feature, r_feature, pos_edge, pos_edge1, k_sets, Y = load_data(path)

    # 十次五倍交叉
    # pan-cancer
    for i in range(10):
        print("\n", "times:", i, "\n")
        for cv_run in range(5):
            print("the %s five-fold cross:\n" % cv_run)
            _, _, tr_mask, te_mask = k_sets[i][cv_run]
            # load model
            model = Net(l_feature, r_feature, network1, network2, 1, 64, 256, 128, pos_edge, pos_edge1).to(device)    # hop=1
            optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0005)
            decayRate = 0.96
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
            for epoch in range(1,EPOCH+1):
                print(epoch)
                train(tr_mask, Y)
                if epoch % 50 ==0:
                    my_lr_scheduler.step()
            AUC, AUPRC = test(tr_mask, te_mask, Y)
            print(AUC)
            print(AUPRC)
            AUC_test[i][cv_run] = AUC
            AUPRC_test[i][cv_run] = AUPRC
            print(time.time() - time_start)
            # 保存结果
            np.savetxt("./AUC_test.txt", AUC_test, delimiter="\t")
            np.savetxt("./AUPR_test.txt", AUPRC_test, delimiter="\t")

    # specific cancer
    # for k in range(10):
    #     print("\n", "times:", k, "\n")
    #     label, label_pos, label_neg = load_label_single(path)
    #     random.shuffle(label_pos)
    #     random.shuffle(label_neg)
    #     l = len(label)
    #     l1 = int(len(label_pos)/5)
    #     l2 = int(len(label_neg)/5)
    #     for i in range(5):
    #         print("the %s five-fold cross:\n" % i)
    #         tr_mask, te_mask = sample_division_single(label_pos, label_neg, l, l1, l2, i)
    #         model = Net(l_feature, r_feature, network1, network2, 1, 19, 256, 128, pos_edge, pos_edge1).to(device)     # hop=1
    #         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #         for epoch in range(1, EPOCH + 1):
    #             train(tr_mask,label)
    #         AUC, AUPRC = test(tr_mask, te_mask, label)
    #         print(AUC)
    #         print(AUPRC)
    #         AUC_test[k][i] = AUC
    #         AUPRC_test[k][i] = AUPRC
    #         print(time.time() - time_start)
    #         np.savetxt("./AUC_test.txt", AUC_test, delimiter="\t")
    #         np.savetxt("./AUPRC_test.txt", AUPRC_test, delimiter="\t")
