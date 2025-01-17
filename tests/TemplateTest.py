import unittest
import numpy as np
import random
from subprocess import Popen
import gc

import stanscofi.datasets
import stanscofi.training_testing
import stanscofi.validation
import stanscofi.utils
from time import time

import sys ##
sys.path.insert(0, "../src/") ##
import benchscofi
import benchscofi.XXXXXX

class TestModel(unittest.TestCase):

    ## Generate example
    def generate_dataset(self, random_seed):
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = stanscofi.datasets.generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std, random_state=random_seed)
        dataset = stanscofi.datasets.Dataset(**data_args)
        return dataset

    ## Load existing dataset
    def load_dataset(self, dataset_name):
        Popen("mkdir -p ../datasets/".split(" "))
        dataset = stanscofi.datasets.Dataset(**stanscofi.utils.load_dataset(dataset_name, "../datasets/"))
        return dataset

    ## Test whether basic functions of the model work
    def test_model(self): 
        random_seed = 124565 
        test_size = 0.3
        batch_ratio=1
        np.random.seed(random_seed)
        random.seed(random_seed)
        if ("YYYYYYYYYYY"==("Y"*11)):
            dataset = self.generate_dataset(random_seed)
        else:
            dataset = self.load_dataset("YYYYYYYYYYY")
        if (batch_ratio<1):
            print("Random batch of size %d (ratio=%f perc.)" % (batch_ratio*dataset.nitems*dataset.nusers, batch_ratio))
            (_, red_folds), _ = stanscofi.training_testing.random_simple_split(dataset, batch_ratio, metric="euclidean", random_state=random_seed)
            dataset = dataset.subset(red_folds)
        (train_folds, test_folds), _ = stanscofi.training_testing.random_simple_split(dataset, test_size, metric="euclidean", random_state=random_seed)
        model = benchscofi.XXXXXX.XXXXXX()
        train_dataset = dataset.subset(train_folds)
        test_dataset = dataset.subset(test_folds)
        fit_start = time()
        model.fit(train_dataset)
        fit_runtime = time()-fit_start
        predict_start = time()
        scores = model.predict_proba(test_dataset)
        predict_runtime = time()-predict_start
        print("\n\n"+('_'*27)+"\nMODEL "+model.name)
        model.print_scores(scores)
        predictions = model.predict(scores, threshold=0)
        model.print_classification(predictions)
        metrics, _ = stanscofi.validation.compute_metrics(scores, predictions, test_dataset, metrics=["AUC", "NDCGk"], k=dataset.nitems, beta=1, verbose=False)
        print(metrics)
        from stanscofi.validation import AUC, NDCGk
        y_test = (test_dataset.folds.toarray()*test_dataset.ratings.toarray()).ravel()
        y_test[y_test<1] = 0
        print("(global) AUC = %.3f" % AUC(y_test, scores.toarray().ravel(), 1, 1))
        print("(global) NDCG@%d = %.3f" % (dataset.nitems, NDCGk(y_test, scores.toarray().ravel(), dataset.nitems, 1)))
        print("Runtime (model fitting) %.3f sec." % fit_runtime)
        print("Runtime (model prediction) %.3f sec." % predict_runtime)
        print(("_"*27)+"\n\n")
        gc.collect()
        #from sklearn.metrics import ndcg_score, roc_auc_score
        ## if it ends without any error, it is a success