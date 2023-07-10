#coding: utf-8

## https://github.com/luckymengmeng/SCPMF/tree/9158aa22d53687938bf6402dbcd9812aabf557e9
from stanscofi.models import BasicModel, create_overscores
from stanscofi.preprocessing import CustomScaler

import numpy as np
import os
from subprocess import call

import calendar
import time
current_GMT = time.gmtime()

## /!\ Only tested on Linux
class SCPMF(BasicModel):
    def __init__(self, params=None):
        try:
            call("octave -v", shell=True)
        except:
            raise ValueError("Please install Octave.")
        params = params if (params is not None) else self.default_parameters()
        super(SCPMF, self).__init__(params)
        self.scalerS, self.scalerP = None, None
        self.name = "SCPMF" 
        self.model = None
        self.use_masked_dataset = True
        self.SCPMF_filepath = None
        self.r = params["r"]

    def default_parameters(self):
        params = {
            "r": 15,
            "decision_threshold": 1, 
        }
        return params

    def preprocessing(self, dataset, inf=2):
        if (self.scalerS is None):
            self.scalerS = CustomScaler(posinf=inf, neginf=-inf)
        S_ = self.scalerS.fit_transform(dataset.items.T.copy(), subset=None)
        X_s = S_ if (S_.shape[0]==S_.shape[1]) else np.corrcoef(S_)
        if (self.scalerP is None):
            self.scalerP = CustomScaler(posinf=inf, neginf=-inf)
        P_ = self.scalerP.fit_transform(dataset.users.T.copy(), subset=None)
        X_p = P_ if (P_.shape[0]==P_.shape[1]) else np.corrcoef(P_)
        A_sp = np.copy(dataset.ratings_mat) # items x users
        return X_s, X_p, A_sp
        
    def fit(self, train_dataset):
        X_s, X_p, A_sp = self.preprocessing(train_dataset)
        time_stamp = calendar.timegm(current_GMT)
        filefolder = "SCPMF_%s" % time_stamp 
        call("mkdir -p %s/" % filefolder, shell=True)
        call("wget -qO %s/SCPMFDR.m 'https://raw.githubusercontent.com/luckymengmeng/SCPMF/master/SCPMFDR.m'" % filefolder, shell=True)
        np.savetxt("%s/X_p.csv" % filefolder, X_p, delimiter=",")
        np.savetxt("%s/A_sp.csv" % filefolder, A_sp, delimiter=",")
        np.savetxt("%s/X_s.csv" % filefolder, X_s, delimiter=",")
        cmd = "A_sp = csvread('A_sp.csv');X_s = csvread('X_s.csv');X_p = csvread('X_p.csv');recMatrix=SCPMFDR(A_sp,X_s,X_p,%d);csvwrite('recMatrix.csv', recMatrix);" % (self.r)
        call("cd %s/ && octave --silent --eval \"%s\"" % (filefolder, cmd), shell=True)
        self.model = np.loadtxt("%s/recMatrix.csv" % filefolder, delimiter=",")
        call("rm -rf %s/" % filefolder, shell=True)
  

    def model_predict(self, test_dataset):
        assert test_dataset.folds is not None
        folds = np.copy(test_dataset.folds)
        folds[:,2] = [self.model[user,item] for user, item in folds[:,:2].tolist()]
        scores = create_overscores(folds, test_dataset)
        return scores