import torch
import dgl
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from stanscofi.datasets import Dataset
from stanscofi.training_testing import random_simple_split
from stanscofi.preprocessing import CustomScaler

from benchscofi.utils import rowwise_metrics
from benchscofi.HAN import HAN
from benchscofi.implementations import HANImplementation

SEED = 1234
DATA_FOLDER = "data"
PARAMS = {
    "k": 15,
    "learning_rate": 1e-3,
    "epoch": 1000, #1000,
    "weight_decay": 0.0,
    "seed": SEED,
}
np.random.seed(SEED)
lambda_ = 1e2

def select_max_var_features(features, n=50):
    features_var = np.nanvar(features, axis=1)
    sorted_indexes = (-features_var).argsort()[:n]
    selected_features = features[sorted_indexes]
    return selected_features

class HANBaseline(HAN):
    def __init__(self, params=None):
        super().__init__(params)
        self.name = "HAN1"

    def preprocessing(self, dataset, inf=2, is_training=True):
        if (self.scalerS is None):
            self.scalerS = CustomScaler(posinf=inf, neginf=-inf)
        drug_drug = self.scalerS.fit_transform(dataset.items.T.toarray().copy(), subset=50)
        drug_drug = np.nan_to_num(drug_drug, nan=0) ##
        drug_drug = drug_drug if (drug_drug.shape[0]==drug_drug.shape[1]) else np.corrcoef(drug_drug)
        if (self.scalerP is None):
            self.scalerP = CustomScaler(posinf=inf, neginf=-inf)
        disease_disease = self.scalerP.fit_transform(dataset.users.T.toarray().copy(), subset=50)
        disease_disease = np.nan_to_num(disease_disease, nan=0) ##
        disease_disease = disease_disease if (disease_disease.shape[0]==disease_disease.shape[1]) else np.corrcoef(disease_disease)
        drug_drug_link = HANImplementation.topk_filtering(drug_drug, self.k)
        disease_disease_link = HANImplementation.topk_filtering(disease_disease, self.k)
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


class HANBench(HAN):
    def __init__(self, params=None):
        super().__init__(params)
        self.name = "HAN1"

    def preprocessing(self, dataset, inf=2, is_training=True):
        ################## TO MODIFY (BEGIN)
        if (self.scalerS is None):
            self.scalerS = IterativeImputer(max_iter=25, random_state=SEED)
        if (self.scalerP is None):
            self.scalerP = IterativeImputer(max_iter=25, random_state=SEED)
        ################## TO MODIFY (END)

        item_features = dataset.items.toarray().copy()
        users_features = dataset.users.toarray().copy()
        item_features = select_max_var_features(item_features).T
        users_features = select_max_var_features(users_features).T

        print('Preprocessing starts...')
        print('Missing value imputation starts...')

        drug_drug = self.scalerS.fit_transform(item_features)
        disease_disease = self.scalerP.fit_transform(users_features)
        print(np.sum(np.isnan(disease_disease)))

        print('Missing value imputation finished!')
        drug_drug = np.nan_to_num(drug_drug, copy=True, nan=np.nan, posinf=inf, neginf=-inf)
        disease_disease = np.nan_to_num(disease_disease, copy=True, nan=np.nan, posinf=inf, neginf=-inf)
        print(np.sum(np.isnan(disease_disease)))
        #drug_drug = np.nan_to_num(drug_drug, nan=0) ##
        #disease_disease = np.nan_to_num(disease_disease, nan=0) ##
        sigma = 1e-4
        perturbation = np.random.normal(scale=sigma, size=disease_disease.shape)
        print('*****************************')
        print(np.min(perturbation), np.max(perturbation))
        disease_disease += perturbation

        drug_drug = np.corrcoef(drug_drug)
        disease_disease = np.corrcoef(disease_disease)
        #
        # print(np.sum(np.isnan(disease_disease)))
        # print(np.max(drug_drug), np.min(drug_drug))
        # print(np.max(disease_disease), np.min(disease_disease))

        drug_drug_link = HANImplementation.topk_filtering(drug_drug, self.k)
        disease_disease_link = HANImplementation.topk_filtering(disease_disease, self.k)
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
        print('Preprocessing finished!')
        return [g, data, label, shp] if (is_training) else [g]


def get_dataset(folder):
	A = pd.read_csv(f"{folder}/ratings_mat.csv", index_col=0)
	P = pd.read_csv(f"{folder}/users.csv", index_col=0)
	S = pd.read_csv(f"{folder}/items.csv", index_col=0)
	return Dataset(**{"ratings": A, "users": P, "items": S})


def main():

	dataset = get_dataset(folder=DATA_FOLDER)
	(train_folds, test_folds), _ = random_simple_split(dataset, 0.20)
	train_dataset = dataset.subset(train_folds, subset_name="Train")
	test_dataset = dataset.subset(test_folds, subset_name="Test")
	train_dataset.summary()
	test_dataset.summary()

	model = HANBench(PARAMS)
	model.subset = 50
	model.fit(train_dataset, seed=SEED)  # single fit

	scores = model.predict_proba(test_dataset)
	predictions = model.predict(scores)

	model.print_scores(scores)
	model.print_classification(predictions)

	lin_auc = np.mean(rowwise_metrics.calc_auc(scores, test_dataset))

	y_test = (
        test_dataset.folds.toarray()
        * test_dataset.ratings.toarray()
    ).ravel()
	y_test[y_test < 1] = 0

	auc = roc_auc_score(y_test, scores.toarray().ravel())

	print(f"NS-AUC {lin_auc}, AUC = {auc}")


if __name__ == "__main__":
    main()
