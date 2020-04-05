# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:22:09 2020

@author: javier.moral.hernan1
"""

# %% Import packages
import pandas as pd
import pickle
from src.preprocessing.datatreatment import DataTreatment
from src.imbalance.imbalanced_treatment import UnbalancedDataSampling
from src.models.default_model import DefaultSingleModel
from src.models.optimized_model import OptimizedSingleModel
from src.models.cascading import CascadingEnsemble
from src.models.majority_minority_ensemble import MajorityMinorityEnsemble
from src.models.majority_opt_ensemble import MajorityOptEnsemble

# %% Load data
modelar = pd.read_csv(r'data/Modelar_UH2020.txt', sep='|')
estimar = pd.read_csv(r'data/Estimar_UH2020.txt', sep='|')

# %% Data preprocessing
data_procesor = DataTreatment(modelar, estimar)
(X_train, X_test, y_train, y_test, test) = data_procesor.getData()

# %% Resampling
sampler = UnbalancedDataSampling(X_train, y_train, 'random_under', 35000)
X_train, y_train = sampler.getData()

# %% Default Single Model
model = DefaultSingleModel(X_train, X_test, y_train, y_test, test, 'lightgbm')
model.get_score()
model.visualize_predict('test')
model.get_var_importance()
preds = model.get_predictions('external')

# %% Optimized Model
model = OptimizedSingleModel(X_train, X_test, y_train,
                             y_test, test, 'lightgbm', 'random')
model.get_score()
model.visualize_predict('test')
model.get_var_importance()
preds = model.get_predictions('external')

# %% Cascading Ensemble
model = CascadingEnsemble(X_train, X_test, y_train, y_test, test, 0.85, 0.8)
model.get_score()
model.get_var_importance()
preds = model.get_predictions('external')

# %% Majority Sequential Ensemble
model = MajorityMinorityEnsemble(X_train, X_test,  y_train, y_test,
                                 test, 'xgboost', 'randomforest')
model.get_score()
model.get_var_importance()
preds = model.get_predictions('external')

# %% Majority Optimized Sequential Ensemble
model = MajorityOptEnsemble(X_train, X_test,  y_train, y_test,
                            test, 'randomforest', 'randomforest', 'random')
model.get_score()
model.get_var_importance()
preds = model.get_predictions('external')

# %% Save Best Model
filename = 'src/models/BestModel.sav'
pickle.dump(model, open(filename, 'wb'))

# %% Load Best Model
pickle_ = open(filename, "rb")
model = pickle.load(pickle_)
model.get_score()
model.get_predictions('external')

# %% Process predictions file

preds['ID'] = estimar['ID']
preds.columns = ['preds', 'ID']
preds = preds[['ID', 'preds']]
preds.to_csv('Minsait_AfiEscuela_DataShow.txt', index=False,
             sep='|', header=True)
preds['preds'].value_counts()
