
from __future__ import unicode_literals

import collections as co
import os
import sys
import csv
import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate
from datetime import date
from sklearn import metrics
import time
import pyreadr

#packages for LR
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
import sys

import xgboost
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from numpy import argmax
from numpy import arange

from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import StratifiedKFold

from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import auc as AUC
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_fscore_support

#matplotlib inline
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms


infile = sys.argv[1] #ad file
R_data_ad = pyreadr.read_r(infile)
keys_addata_k = R_data_ad.keys()
print(keys_addata_k)
df_addata = R_data_ad['adallfinal']

infile = sys.argv[2] #om file
R_data_o = pyreadr.read_r(infile)
keys_odata_k = R_data_o.keys()
print(keys_odata_k)
df_odata = R_data_o['oallfinal']

#Concat both
frames = [df_addata, df_odata]
df_adodata= pd.concat(frames)
print(df_adodata.info())

dataset_train = df_adodata
Y_train = dataset_train['adverse'].dropna().astype(int)
SYsize = len(Y_train)
dataset_train = dataset_train.drop(['adverse'], axis = 1)
X_train = dataset_train 
Ssize = len(X_train)

print('Sample size'+str(Ssize)+' Check label size:'+str(SYsize))

random_state = 7
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=random_state)
max_iter=50
imputer = IterativeImputer(max_iter=max_iter)

DEPTH = 7
n_estimators = 80 
learning_rate = 0.1 
model = xgboost.XGBClassifier(max_depth=DEPTH, learning_rate=learning_rate, n_estimators=n_estimators,
                             objective='binary:logistic', booster='gbtree')#RandomForestClassifier()

pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
classifier = pipeline
classifierName = 'xgboost_lr'+str(learning_rate)


original_stdout = sys.stdout # Save a reference to the original standard output

   
with open(classifierName+'_TrnSA_Eval80-20split_IMPUTE_'+str(DEPTH)+"_"+str(n_estimators)+'_contvar.txt', 'w') as f:
    sys.stdout = f # map the standard output to the file we created.
    fig, ax = plt.subplots()
    classifier.fit(x_train,y_train)
    model = classifier
    yhat = model.predict_proba(x_test) #(testX)
    # keep probabilities for the positive outcome only
    yhat = yhat[:, 1]
    testy = y_test #ground truth
    probs = yhat
    fpr, tpr, _ = metrics.roc_curve(testy,  probs)
    auc = metrics.roc_auc_score(testy, probs)
    print('AUC: ',auc)
    sys.stdout = original_stdout
    #Save testy, probs for external plot
    cal_plot_data = pd.DataFrame()
    cal_plot_data['testy'] = testy
    cal_plot_data['probs'] = probs
    cal_plot_data.to_csv('trueY_predY_for_calib_plot_imp_SA80-20split.csv')

    from sklearn.calibration import calibration_curve
    n_bins= 10
    logreg_y, logreg_x = calibration_curve(testy, probs, n_bins=n_bins)
    fig, ax = plt.subplots()
    # only these two lines are calibration curves
    plt.plot(logreg_x, logreg_y, marker='o', linewidth=1, label='xgb')
    calib_data = pd.DataFrame()
    calib_data['fop'] = logreg_y
    calib_data['apv'] = logreg_x
    calib_data.to_csv('calibrationData_nbin'+str(n_bins)+'.csv')

    # reference line, legends, and axis labels
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle('Calibration plot - SA 80-20 split')
    ax.set_xlabel('average predicted probability')
    ax.set_ylabel('fraction of true probability')
    plt.legend()
    plt.savefig('xgb_calib_imp_contvar_ext_SA-80-20split_'+str(n_bins)+'.pdf')
    plt.show()

