#coding: utf-8

#https://github.com/gu-yaowen/MilGNet/blob/a0a129311a07feb928580ee7695d4321f24fd5c3/baseline/HAN_imp.py
from stanscofi.models import BasicModel
from stanscofi.preprocessing import CustomScaler
from benchscofi.implementations import HANImplementation as han_implementation

import os
import dgl
import torch
import pandas as pd
import numpy as np
from scipy.sparse import coo_array

class HAN(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(HAN, self).__init__(params)
        self.scalerS, self.scalerP = None, None
        self.device = torch.device('cpu')
        self.scalerS, self.scalerP = None, None
        self.name = "HAN" 
        self.estimator = None

    def default_parameters(self):
        params = {
            "k": 15, "learning_rate": 1e-3, "epoch": 1000, 
            "weight_decay": 0.0, "seed": 42, 
        }
        return params

    def preprocessing(self, dataset, inf=2, is_training=True):
        if (self.scalerS is None):
            self.scalerS = CustomScaler(posinf=inf, neginf=-inf)
        drug_drug = self.scalerS.fit_transform(dataset.items.T.toarray().copy(), subset=None)
        drug_drug = np.nan_to_num(drug_drug, nan=0) ##
        drug_drug = drug_drug if (drug_drug.shape[0]==drug_drug.shape[1]) else np.corrcoef(drug_drug)
        if (self.scalerP is None):
            self.scalerP = CustomScaler(posinf=inf, neginf=-inf)
        disease_disease = self.scalerP.fit_transform(dataset.users.T.toarray().copy(), subset=None)
        disease_disease = np.nan_to_num(disease_disease, nan=0) ##       
        disease_disease = disease_disease if (disease_disease.shape[0]==disease_disease.shape[1]) else np.corrcoef(disease_disease)
        drug_drug_link = han_implementation.topk_filtering(drug_drug, self.k)
        disease_disease_link = han_implementation.topk_filtering(disease_disease, self.k)
        drug_disease = dataset.ratings.toarray()
        drug_disease[drug_disease<0] = 0
        drug_disease_link = np.array(np.where(drug_disease == 1)).T
        disease_drug_link = np.array(np.where(drug_disease.T == 1)).T
        graph_data = {('drug', 'drug-drug', 'drug'): (torch.tensor(drug_drug_link[:, 0]),
                                                  torch.tensor(drug_drug_link[:, 1])),
                  ('drug', 'drug-disease', 'disease'): (torch.tensor(drug_disease_link[:, 0]),
                                                        torch.tensor(drug_disease_link[:, 1])),
                  ('disease', 'disease-drug', 'drug'): (torch.tensor(disease_drug_link[:, 0]),
                                                        torch.tensor(disease_drug_link[:, 1])),
                  ('disease', 'disease-disease', 'disease'): (torch.tensor(disease_disease_link[:, 0]),
                                                              torch.tensor(disease_disease_link[:, 1]))}
        g = dgl.heterograph(graph_data)
        drug_feature = np.hstack((drug_drug, np.zeros(drug_disease.shape)))
        dis_feature = np.hstack((np.zeros(drug_disease.T.shape), disease_disease))
        g.nodes['drug'].data['h'] = torch.from_numpy(drug_feature).to(torch.float32)
        g.nodes['disease'].data['h'] = torch.from_numpy(dis_feature).to(torch.float32)
        data = torch.tensor(np.column_stack((dataset.folds.row, dataset.folds.col)).astype('int64')).to(self.device)
        label = torch.tensor(drug_disease[dataset.folds.row, dataset.folds.col].flatten()).float().to(self.device)
        shp = dataset.ratings.shape
        return [g, data, label, shp] if (is_training) else [g]
        
    def model_fit(self, g_train, train_data, train_label, shp):
        han_implementation.set_seed(self.seed)
        feature = {'drug': g_train.nodes['drug'].data['h'],
                   'disease': g_train.nodes['disease'].data['h']}

        model = han_implementation.HAN(in_feats=[feature['drug'].shape[1],
                              feature['disease'].shape[1]],
                    meta_paths=[['drug-disease', 'disease-drug'],
                                ['disease-drug', 'drug-disease'],
                                ['drug-drug', 'drug-disease', 'disease-drug'],
                                ['disease-disease', 'disease-drug', 'drug-disease']])

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(torch.where(train_label == 0)[0]) /
                                                                       len(torch.where(train_label == 1)[0])))
        print('BCE loss pos weight: {:.3f}'.format(
            len(torch.where(train_label == 0)[0]) / len(torch.where(train_label == 1)[0])))

        train_idx = torch.tensor([(i.item() - 1) * shp[1] + j
                                  for (i, j) in train_data]).to(self.device)
        for epoch in range(1, self.epoch + 1):
            model.train()

            pred = model(g_train, feature).flatten().flatten()[train_idx]
            pred_score = torch.sigmoid(pred)

            optimizer.zero_grad()
            loss = criterion(pred, train_label)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0 or epoch == self.epoch - 1:
                model.eval()
                train_label_ = train_label.detach().cpu().numpy()
                pred_score_ = pred_score.detach().cpu().numpy()
                AUC, AUPR = han_implementation.get_metrics_auc(
                    train_label_, pred_score_
                )
                print('Epoch {} Loss: {:.3f}; Train: AUC {:.3f}, '
                      'AUPR {:.3f}'.format(epoch, loss.item(), AUC, AUPR))
        self.estimator = {"predictions": torch.sigmoid(model(g_train, feature)).detach().cpu().numpy()}

    def model_predict_proba(self, g):
        return self.estimator["predictions"]