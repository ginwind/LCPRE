from utils.data import negExCreating, shufflePosNeg, MADataset
from utils.rcGraph import createRCGraph, readGraphMetaData
from utils.encode import *
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score,f1_score
from sklearn.model_selection import KFold

import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from model.lcpre import LCPRE
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# modify your pretrained language model path here
bert_path = None

ap = argparse.ArgumentParser(description='LCPRE training')
ap.add_argument('--datapath', type=str, default="./dataset/LB2")
ap.add_argument('--lr', type=float, default=5e-4)
ap.add_argument('--epoch', type=int, default=50)
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--random_percent', type=float, default=0.5, help="the percent of random negative examples.")
ap.add_argument('--Kf', type=int, default=4, help="K fold cross-evaluation.")
ap.add_argument('--max_intra', type=int, default=128, help="the number of the maximum intra learningpath when modeling.")
ap.add_argument('--learningpaths', type=str, default="CC,CRC")
ap.add_argument('--patience', type=int, default=30)

ap.add_argument('--node_dim', type=int, default=128)
ap.add_argument('--num_heads', type=int, default=4)
ap.add_argument('--out_dim', type=int, default=64)

args = ap.parse_args()
data_path = args.datapath
lr = args.lr
epoch = args.epoch
batch_size = args.batch_size
weight_decay = 1e-3
random_percent = args.random_percent
Kf = args.Kf
metaname = args.learningpaths.split(",")
maxIntra = args.max_intra
patience = args.patience

node_dim = args.node_dim
num_heads = args.num_heads
out_dim = args.out_dim


def trainMAGNN(num_resources, Xtrain, Ytrain, Xtest, Ytest, rcg):
    net = LCPRE(embedding_matrix=embedding_matrix, num_resource=num_resources, metaname=metaname, 
                embedding_dim=embedding_matrix.shape[1], projection_dim=node_dim, num_heads=num_heads, node_dim=node_dim,
                 attn_vec_dim=out_dim, out_dim=out_dim, attn_drop=0.6).to(device)
    optimizer = AdamW(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                milestones=[int(epoch * 0.64), int(epoch *0.87)],
                gamma=0.5, last_epoch=-1)
    loss1 = nn.BCELoss()
    plist, rlist, f1list = [], [], []
    vplist, vrlist, vf1list = [], [], []
    maxF1, pati = 0, 0

    for i in range(0, epoch):
        net.train()
        print(f"epoch{i}:===")
        sum_loss = 0
        y_pred, y_true = [], []
        
        trainset = MADataset(Xtrain, Ytrain, rcg, maxIntraPath=maxIntra, metaName=metaname)
        testset = MADataset(Xtest[: len(Xtest) // 2], Ytest[: len(Xtest) // 2], rcg, maxIntraPath=maxIntra, metaName=metaname)
        valset = MADataset(Xtest[len(Xtest) // 2: ], Ytest[len(Xtest) // 2: ], rcg, maxIntraPath=maxIntra, metaName=metaname)

        trainloader, testloader, valloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True), \
                                             DataLoader(dataset=testset, batch_size=batch_size, shuffle=True), \
                                             DataLoader(dataset=valset, batch_size=batch_size, shuffle=True),
        train_iter, test_iter, val_iter = iter(trainloader), iter(testloader), iter(valloader)
        for iteration in tqdm(range(0, len(trainloader))):
            mini_p1, mini_mk1, mini_p2, mini_mk2, mini_labels = next(train_iter)
            mini_p1 = [each.to(device) for each in mini_p1]
            mini_mk1= [each.to(device) for each in mini_mk1]
            mini_p2 = [each.to(device) for each in mini_p2]
            mini_mk2 = [each.to(device) for each in mini_mk2]
            mini_labels = mini_labels.to(device)
            
            embeddings1, embeddings2 = net(mini_p1, mini_mk1, mini_p2, mini_mk2)
            out = torch.sigmoid(torch.bmm(torch.unsqueeze(embeddings1, -2), torch.unsqueeze(embeddings2, -1))).flatten()
            train_loss = loss1(out, mini_labels)

            sum_loss += train_loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            y_pred.extend(out.cpu().detach().numpy())
            y_true.extend(mini_labels.cpu().numpy())
        print(f"loss:{sum_loss}")
        scheduler.step()
        net.eval()
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        f1maxt = f1_score(y_true, [1 if each > 0.5 else 0 for each in y_pred])
        trecall = recall_score(y_true, [1 if each > 0.5 else 0 for each in y_pred])
        tprecision = precision_score(y_true, [1 if each > 0.5 else 0 for each in y_pred])
        

        y_pred, y_true = [], []
        with torch.no_grad():
            for iteration in range(0, len(test_iter)):
                mini_p1, mini_mk1, mini_p2, mini_mk2, mini_labels = next(test_iter)
                mini_p1 = [each.to(device) for each in mini_p1]
                mini_mk1= [each.to(device) for each in mini_mk1]
                mini_p2 = [each.to(device) for each in mini_p2]
                mini_mk2 = [each.to(device) for each in mini_mk2]
                mini_labels = mini_labels.to(device)

                embeddings1, embeddings2 = net(mini_p1, mini_mk1, mini_p2, mini_mk2)
                out = torch.sigmoid(torch.bmm(torch.unsqueeze(embeddings1, -2), torch.unsqueeze(embeddings2, -1))).flatten()

                y_pred.extend(out.cpu().detach().numpy())
                y_true.extend(mini_labels.cpu().numpy())

        y_pred, y_true = np.array(y_pred), np.array(y_true)
        f1max = f1_score(y_true, [1 if each > 0.5 else 0 for each in y_pred])
        recall = recall_score(y_true, [1 if each > 0.5 else 0 for each in y_pred])
        precision = precision_score(y_true, [1 if each > 0.5 else 0 for each in y_pred])
        rlist.append(recall)
        plist.append(precision)
        f1list.append(f1max)

        y_pred, y_true = [], []
        with torch.no_grad():
            for iteration in range(0, len(val_iter)):
                mini_p1, mini_mk1, mini_p2, mini_mk2, mini_labels = next(val_iter)
                mini_p1 = [each.to(device) for each in mini_p1]
                mini_mk1= [each.to(device) for each in mini_mk1]
                mini_p2 = [each.to(device) for each in mini_p2]
                mini_mk2 = [each.to(device) for each in mini_mk2]
                mini_labels = mini_labels.to(device)

                embeddings1, embeddings2 = net(mini_p1, mini_mk1, mini_p2, mini_mk2)
                out = torch.sigmoid(torch.bmm(torch.unsqueeze(embeddings1, -2), torch.unsqueeze(embeddings2, -1))).flatten()

                y_pred.extend(out.cpu().detach().numpy())
                y_true.extend(mini_labels.cpu().numpy())

        y_pred, y_true = np.array(y_pred), np.array(y_true)
        valrecall = recall_score(y_true, [1 if each > 0.5 else 0 for each in y_pred])
        valprecision = precision_score(y_true, [1 if each > 0.5 else 0 for each in y_pred])
        valf1 = f1_score(y_true, [1 if each > 0.5 else 0 for each in y_pred])
        vrlist.append(valrecall)
        vplist.append(valprecision)
        vf1list.append(valf1)

        if f1max-maxF1 < -0.04:
            pati += 1
        maxF1 = max(maxF1, f1max)
        if pati >= patience:
            break
        print(f"Epoch:{i},  Train, precision:{tprecision}, recall:{trecall}, F1:{f1maxt}({0.5})\n    \
                           \t Test, precision:{precision}, recall:{recall}, F1:{f1max}({0.5}) \n    \
                           \t Val, precision:{valprecision}, recall:{valrecall}, F1:{valf1}({0.5})")
    ap_csv = pd.DataFrame({"tP": plist, "tR":rlist, "tF1": f1list, "P":vplist, "R":vrlist, "F1":vf1list})
    max3, index = 0, 0
    for i in range(0, len(ap_csv)):
        ite_sum = ap_csv["tP"][i] + ap_csv["tR"][i] + ap_csv["tF1"][i]
        if ite_sum > max3:
            max3 = ite_sum 
            index = i
    return ap_csv.iloc[index][["P", "R", "F1"]].values

if __name__ == "__main__":
    resources, concepts, cc_edges, rc_edges, rr_edges, descriptions, node2Idx, resource_text, concept_text, _ = readGraphMetaData(data_path)
    nodes = resource_text + concept_text
    if os.path.exists(data_path+"/embedding.pt"):
        embedding_matrix = torch.load(data_path+"/embedding.pt")
    else:
        print("-------start encoding resources and concepts-------------------")
        embedding_matrix = bertEncoding(data_path+"/embedding.pt", bert_path, nodes, device)
    print("--------finish loading graphMeta and embedding matrix-----------------")

    df = pd.read_csv(f"{data_path}/dataset.csv")
    pos, neg = df[df["label"]==1].values[:,:2].tolist(), df[df["label"]==0].values[:,:2].tolist()
    pos_examples = np.array([[node2Idx[each[0]], node2Idx[each[1]]] for each in pos])
    neg_examples = np.array([[node2Idx[each[0]], node2Idx[each[1]]] for each in neg])

    posKF, negKF = KFold(n_splits=Kf, shuffle=True), KFold(n_splits=Kf, shuffle=True)
    average_metrics = [0, 0, 0]

    for k in range(0, Kf):
        print(f"-----------start K fold training {k}-----------------")
        train_pos, test_pos = next(posKF.split(pos_examples))
        train_neg, test_neg = next(negKF.split(neg_examples))

        train_pos, train_neg, test_pos, test_neg = pos_examples[train_pos], neg_examples[train_neg], pos_examples[test_pos], neg_examples[test_neg]
        rcg = createRCGraph(resources, concepts, train_pos, rc_edges, rr_edges, maxIntra, data_path)

        train_x, train_y = shufflePosNeg(train_pos.tolist(), train_neg.tolist())
        test_x, test_y = shufflePosNeg(test_pos.tolist(), test_neg.tolist())
        print(f"train examples:{len(train_x)}, test examples:{len(test_x)}")

        average_metrics += trainMAGNN(len(resources), train_x, train_y, test_x, test_y, rcg)
        print(f"-----------finish K fold training {k}-----------------")
        print()
        rcg.clearCache()
    print(f"final metrics from K fold evaluation:{[each/Kf for each in average_metrics]}")

    