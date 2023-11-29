import numpy as np
import torch
import dgl
import random
import seaborn
import os
from sklearn.metrics import roc_curve, roc_auc_score, \
    precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

## https://raw.githubusercontent.com/gu-yaowen/MilGNet/a0a129311a07feb928580ee7695d4321f24fd5c3/baseline/model_zoo.py
import dgl
import torch
import math
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl.nn.pytorch import GATConv
from dgl.nn.functional import edge_softmax


class PGCN(nn.Module):

    def __init__(self, in_feats, rel_names):
        super().__init__()
        self.drug_emb = nn.Linear(in_feats[0], 64, bias=False)
        self.dis_emb = nn.Linear(in_feats[1], 64, bias=False)
        HeteroGraphdict_1 = {}
        for rel in rel_names:
            graphconv = dglnn.GraphConv(64, 64)
            nn.init.xavier_normal_(graphconv.weight)
            HeteroGraphdict_1[rel] = graphconv
        self.embedding_1 = dglnn.HeteroGraphConv(HeteroGraphdict_1, aggregate='sum')
        HeteroGraphdict_2 = {}
        for rel in rel_names:
            graphconv = dglnn.GraphConv(64, 32)
            nn.init.xavier_normal_(graphconv.weight)
            HeteroGraphdict_2[rel] = graphconv
        self.embedding_2 = dglnn.HeteroGraphConv(HeteroGraphdict_2, aggregate='sum')
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()
        self.weights = nn.Linear(32, 32, bias=False)
        nn.init.xavier_uniform_(self.weights.weight)

    def forward(self, g, feature):
        h = {'drug': self.drug_emb(feature['drug']),
             'disease': self.dis_emb(feature['disease'])}

        h = self.embedding_1(g, h)
        h = {k: self.relu(self.dropout(v)) for k, v in h.items()}

        h = self.embedding_2(g, h)
        h = {k: self.relu(self.dropout(v)) for k, v in h.items()}
        outputs = torch.matmul(self.weights(h['drug']), h['disease'].T)
        return outputs


class HeteroRGCNLayer(nn.Module):
    """Base heterodrugous RGCN layer.
    """

    def __init__(self, etypes, in_feats, out_feats):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_feats, out_feats) for name in etypes
        })

    def forward(self, g, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            # 计算每一类etype的 W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            g.nodes[srctype].data['Wh_%s' % etype] = Wh
            # 消息函数 copy_u: 将源节点的特征聚合到'm'中; reduce函数: 将'm'求均值赋值给 'h'
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        g.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, in_feats, rel_names):
        super(HeteroRGCN, self).__init__()
        self.drug_emb = nn.Linear(in_feats[0], 64, bias=False)
        self.dis_emb = nn.Linear(in_feats[1], 64, bias=False)
        self.embedding_1 = HeteroRGCNLayer(rel_names, 64, 32)
        self.embedding_2 = HeteroRGCNLayer(rel_names, 32, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.weights = nn.Linear(32, 32, bias=False)
        nn.init.xavier_uniform_(self.weights.weight)

    def forward(self, g, feature):
        h = {'drug': self.drug_emb(feature['drug']),
             'disease': self.dis_emb(feature['disease'])}

        h = self.embedding_1(g, h)
        h = {k: self.relu(self.dropout(v)) for k, v in h.items()}

        h = self.embedding_2(g, h)
        h = {k: self.relu(self.dropout(v)) for k, v in h.items()}
        outputs = torch.matmul(self.weights(h['drug']), h['disease'].T)
        return outputs


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout=0.4,
                 use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype: (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer='mean')

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGT(nn.Module):
    def __init__(self, in_feats, node_dict, edge_dict):
        super(HGT, self).__init__()
        self.drug_emb = nn.Linear(in_feats[0], 64, bias=False)
        self.dis_emb = nn.Linear(in_feats[1], 64, bias=False)
        self.embedding = HGTLayer(64, 64, node_dict, edge_dict, 2, 0.1)
        self.relu = nn.ReLU()
        self.weights = nn.Linear(64, 64, bias=False)
        nn.init.xavier_uniform_(self.weights.weight)

    def forward(self, g, feature):
        h = {'drug': self.drug_emb(feature['drug']),
             'disease': self.dis_emb(feature['disease'])}

        h = self.embedding(g, h)
        h = {k: self.relu(v) for k, v in h.items()}

        outputs = torch.matmul(self.weights(h['drug']), h['disease'].T)

        return outputs


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterodrugous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                    g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_feats):
        super(HAN, self).__init__()
        self.drug_emb = nn.Linear(in_feats[0], 64, bias=False)
        self.dis_emb = nn.Linear(in_feats[1], 64, bias=False)
        self.embedding_1_drug = HANLayer([meta_paths[0]], 64, 32, 5, 0.4)
        self.embedding_1_dis = HANLayer([meta_paths[1]], 64, 32, 5, 0.4)
        self.drug_emb_1 = nn.Linear(32 * 5, 32, bias=False)
        self.dis_emb_1 = nn.Linear(32 * 5, 32, bias=False)
        self.embedding_2_drug = HANLayer([meta_paths[0]], 32, 32, 5, 0.4)
        self.embedding_2_dis = HANLayer([meta_paths[1]], 32, 32, 5, 0.4)
        self.drug_emb_2 = nn.Linear(32 * 5, 32, bias=False)
        self.dis_emb_2 = nn.Linear(32 * 5, 32, bias=False)
        self.weights = nn.Linear(32, 32, bias=False)
        nn.init.xavier_uniform_(self.weights.weight)

    def forward(self, g, feature):
        import ipdb; ipdb.set_trace()
        h = {'drug': self.drug_emb(feature['drug']),
             'disease': self.dis_emb(feature['disease'])}

        h['drug'] = self.drug_emb_1(self.embedding_1_drug(g, h['drug']))
        h['disease'] = self.dis_emb_1(self.embedding_1_dis(g, h['disease']))
        h['drug'] = self.drug_emb_2(self.embedding_2_drug(g, h['drug']))
        h['disease'] = self.dis_emb_2(self.embedding_2_dis(g, h['disease']))
        outputs = torch.matmul(self.weights(h['drug']), h['disease'].T)

        return outputs


## https://raw.githubusercontent.com/gu-yaowen/MilGNet/a0a129311a07feb928580ee7695d4321f24fd5c3/baseline/utils.py
def topk_filtering(d_d: np.array, k: int):
    """Convert the Topk similarities to 1 and generate the Topk interactions.
    """
    for i in range(len(d_d)):
        sorted_idx = np.argpartition(d_d[i], -k - 1)
        d_d[i, sorted_idx[-k - 1:-1]] = 1
    return np.array(np.where(d_d == 1)).T


def remove_graph(g, test_drug_id, test_disease_id):
    etype = ('drug', 'drug-disease', 'disease')

    edges_id = g.edge_ids(torch.tensor(test_drug_id),
                          torch.tensor(test_disease_id),
                          etype=etype)
    g = dgl.remove_edges(g, edges_id, etype=etype)
    etype = ('disease', 'disease-drug', 'drug')
    edges_id = g.edge_ids(torch.tensor(test_disease_id),
                          torch.tensor(test_drug_id),
                          etype=etype)
    g = dgl.remove_edges(g, edges_id, etype=etype)
    return g


def get_metrics_auc(real_score, predict_score):
    AUC = roc_auc_score(real_score, predict_score)
    AUPR = average_precision_score(real_score, predict_score)
    return AUC, AUPR


def get_metrics(real_score, predict_score):
    """Calculate the performance metrics.
    Resource code is acquired from:
    Yu Z, Huang F, Zhao X et al.
     Predicting drug-disease associations through layer attention graph convolutional network,
     Brief Bioinform 2021;22.

    Parameters
    ----------
    real_score: true labels
    predict_score: model predictions

    Return
    ---------
    AUC, AUPR, Accuracy, F1-Score, Precision, Recall, Specificity
    """
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc[0, 0], aupr[0, 0], accuracy, f1_score, precision, recall, specificity


def set_seed(seed=0):
    print('seed = {}'.format(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    # dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_result_auc(args, label, predict, auc):
    """Plot the ROC curve for predictions.
    Parameters
    ----------
    args: argumentation
    label: true labels
    predict: model predictions
    auc: calculated AUROC score
    """
    seaborn.set_style()
    fpr, tpr, threshold = roc_curve(label, predict)
    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.saved_path, 'result_auc.png'))
    plt.clf()


def plot_result_aupr(args, label, predict, aupr):
    """Plot the ROC curve for predictions.
    Parameters
    ----------
    args: argumentation
    label: true labels
    predict: model predictions
    aupr: calculated AUPR score
    """
    seaborn.set_style()
    precision, recall, thresholds = precision_recall_curve(label, predict)
    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 8))
    plt.plot(precision, recall, color='darkorange',
             lw=lw, label='AUPR Score (area = %0.4f)' % aupr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('RPrecision/Recall Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.saved_path, 'result_aupr.png'))
    plt.clf()
