# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:44:49 2020

@author: javier.moral.hernan1
"""

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer


class MajorityOptEnsemble():

    def __init__(self, X_train, X_test,
                 y_train, y_test, model_1_selected,
                 model_2_selected, method):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.method = method
        self.binary_target()
        self.model_1_selected = model_1_selected
        self.select_first_model()
        self.model_2_selected = model_2_selected
        self.select_second_model()
        self.fit()
        self.predict()
        self.score()

    def binary_target(self):
        '''
        Creates a binary target where value 1 matches the majority class
        intances and 0 all remaining classes.

        Returns
        -------
        None.
        '''
        self.y_train_binary = self.y_train[
            'CLASE'] .map(lambda x: 1 if x == 'RESIDENTIAL' else 0)

    def select_first_model(self):
        '''
        Gets the first model seleted and its param grid.

        Returns
        -------
        None.
        '''
        if self.model_1_selected == 'randomforest':
            self.model1 = RandomForestClassifier(class_weight='balanced')
            self.param_grid1 = {'bootstrap': [True, False],
                                'max_depth': [10, 20, 40, 60,
                                              80, 100, None],
                                'max_features': ['auto', 'sqrt'],
                                'min_samples_leaf': [1, 2, 4],
                                'min_samples_split': [2, 5, 10],
                                'n_estimators': [200, 600, 800, 1000,
                                                 1200, 1800, 2000]}
            self.param_grid_bayes1 = {
                'max_depth': Integer(10, 100, None),
                'min_samples_split': Integer(2, 20, None),
                'min_samples_leaf': Integer(2, 20, None),
                'n_estimators': Integer(10, 2000, None),
                'bootstrap': [True, False]}
        if self.model_1_selected == 'gradientboosting':
            self.model1 = GradientBoostingClassifier()
            self.param_grid1 = {'max_depth': [10, 40, 60, 80, 100, 200, None],
                                'validation_fraction': [0.1, 0.2, 0.3, None],
                                'n_iter_no_change': [1, 2, 3, 4, 5],
                                'min_samples_split': [2, 5, 10, 20, 50],
                                'n_estimators': [200, 600, 800, 1000,
                                                 1200, 1800, 2000]}
            self.param_grid_bayes1 = {'max_depth': Integer(10, 200, None),
                                      'min_samples_split': Integer(2, 50,
                                                                   None),
                                      'validation_fraction': Real(0.05,
                                                                  0.3, None),
                                      'n_estimators': Integer(10, 2000, None),
                                      'n_iter_no_change': Integer(1, 5)}
        if self.model_1_selected == 'xgboost':
            self.model1 = XGBClassifier()
            self.param_grid1 = {
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [10, 40, 80, 100, 200, None],
                'n_estimators': [200, 600, 1000,
                                 1200, 1800, 2000],
                'learning_rate': [0.001, 0.01, 0.05, 0.15,
                                  0.2, 0.3, 0.6], }
            self.param_grid_bayes1 = {
                'min_child_weight': Integer(1, 15, None),
                'n_estimators': Integer(10, 2000, None),
                'learning_rate': Real(0.001, 0.5, None),
                'gamma': Real(0.1, 5, None),
                'subsample': Real(0.6, 1.0, None),
                'colsample_bytree': Real(0.6, 1.0, None),
                'max_depth': Integer(10, 200, None), }

    def select_second_model(self):
        '''
        Gets the second model seleted and its param grid.

        Returns
        -------
        None.
        '''
        self.target_encoder = LabelEncoder()
        self.y_train_enc = self.target_encoder.fit_transform(self.y_train)
        if self.model_2_selected == 'randomforest':
            self.model2 = RandomForestClassifier(class_weight='balanced')
            self.param_grid2 = {'bootstrap': [True, False],
                                'max_depth': [10, 20, 40, 60,
                                              80, 100, None],
                                'max_features': ['auto', 'sqrt'],
                                'min_samples_leaf': [1, 2, 4],
                                'min_samples_split': [2, 5, 10],
                                'n_estimators': [200, 600, 800, 1000,
                                                 1200, 1800, 2000]}
            self.param_grid_bayes2 = {
                'max_depth': Integer(10, 100, None),
                'min_samples_split': Integer(2, 20, None),
                'min_samples_leaf': Integer(2, 20, None),
                'n_estimators': Integer(10, 2000, None),
                'bootstrap': [True, False]}
        if self.model_2_selected == 'gradientboosting':
            self.model2 = GradientBoostingClassifier()
            self.param_grid2 = {'max_depth': [10, 40, 60, 80, 100, 200, None],
                                'validation_fraction': [0.1, 0.2, 0.3, None],
                                'n_iter_no_change': [1, 2, 3, 4, 5],
                                'min_samples_split': [2, 5, 10, 20, 50],
                                'n_estimators': [200, 600, 800, 1000,
                                                 1200, 1800, 2000]}
            self.param_grid_bayes2 = {
                'max_depth': Integer(10, 200, None),
                'min_samples_split': Integer(2, 50,
                                             None),
                'validation_fraction': Real(0.05,
                                            0.3, None),
                'n_estimators': Integer(10, 2000, None),
                'n_iter_no_change': Integer(1, 5)}
        if self.model_2_selected == 'xgboost':
            self.model2 = XGBClassifier()
            self.param_grid2 = {
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [10, 40, 80, 100, 200, None],
                'n_estimators': [200, 600, 1000,
                                 1200, 1800, 2000],
                'learning_rate': [0.001, 0.01, 0.05, 0.15,
                                  0.2, 0.3, 0.6], }
            self.param_grid_bayes2 = {
                'min_child_weight': Integer(1, 15, None),
                'n_estimators': Integer(10, 2000, None),
                'learning_rate': Real(0.001, 0.5, None),
                'gamma': Real(0.1, 5, None),
                'subsample': Real(0.6, 1.0, None),
                'colsample_bytree': Real(0.6, 1.0, None),
                'max_depth': Integer(10, 200, None), }

    def fit(self):
        if self.method == 'random':
            search1 = RandomizedSearchCV(
                self.model1, self.param_grid1, cv=3, scoring='accuracy',
                n_jobs=-1, refit=True, verbose=10, n_iter=1)
            self.model1 = search1.fit(self.X_train, self.y_train_binary)
            search2 = RandomizedSearchCV(self.model2, self.param_grid2, cv=3,
                                         scoring='accuracy', n_jobs=-1,
                                         refit=True, verbose=10, n_iter=1)
            self.model2 = search1.fit(self.X_train, self.y_train)
        if self.method == 'bayes':
            search1 = BayesSearchCV(
                self.model2, self.param_grid_bayes1, scoring='accuracy', cv=3,
                refit=True, n_jobs=-1, iid=True, return_train_score=True,
                n_points=5, n_iter=20, verbose=10)
            self.model1 = search1.fit(self.X_train, self.y_train_binary)
            search2 = BayesSearchCV(
                self.model2, self.param_grid_bayes2, scoring='accuracy', cv=3,
                refit=True, n_jobs=-1, iid=True, return_train_score=True,
                n_points=5, n_iter=20, verbose=10)
            self.model2 = search2.fit(self.X_train, self.y_train)

    def inverse_encoding(self, data):
        '''
        Decodes previously encoded data.

        Returns
        -------
        None.
        '''
        data_uncoded = self.target_encoder.inverse_transform(data)
        return data_uncoded

    def predictDataset(self, preds1, preds2):
        predictions = pd.concat([preds1,
                                 preds2], axis=1)
        predictions.columns = ['model1', 'model2']
        predictions['preds'] = 'nan'
        for row in predictions.iterrows():
            if row[1]['model1'] == 1:
                predictions.at[row[0], 'preds'] = 'RESIDENTIAL'
            else:
                predictions.at[row[0], 'preds'] = row[1]['model2']
        return predictions

    def predict(self):
        '''
        Computes predictions for each datset using the fitted model.

        Returns
        -------
        None.
        '''
        preds_train1 = pd.DataFrame(self.model1.predict(self.X_train))
        preds_test1 = pd.DataFrame(self.model1.predict(self.X_test))
        preds_ext1 = pd.DataFrame(self.model1.predict(self.external_data))
        preds_train2 = pd.DataFrame(
            self.inverse_encoding(self.model2.predict(self.X_train)))
        preds_test2 = pd.DataFrame(
            self.inverse_encoding(self.model2.predict(self.X_test)))
        preds_ext2 = pd.DataFrame(
            self.inverse_encoding(self.model2.predict(self.external_data)))
        preds_train_df = self.predictDataset(preds_train1, preds_train2)
        preds_test_df = self.predictDataset(preds_test1, preds_test2)
        preds_ext_df = self.predictDataset(preds_ext1, preds_ext2)
        self.preds_train = preds_train_df['preds']
        self.preds_test = preds_test_df['preds']
        self.preds_ext = preds_ext_df['preds']


    def score(self):
        '''
        Computes and stores the accuracy, balanced accuracy, confussion matrix
        and f1_macro metrics for train an tests predictions.

        Returns
        -------
        None.
        '''
        self.acc_train = accuracy_score(self.y_train, self.preds_train)
        self.acc_test = accuracy_score(self.y_test, self.preds_test)
        self.balanced_acc_train = balanced_accuracy_score(self.y_train,
                                                          self.preds_train)
        self.balanced_acc_test = balanced_accuracy_score(self.y_test,
                                                         self.preds_test)
        self.f1_train = f1_score(self.y_train, self.preds_train,
                                 average='macro')
        self.f1_test = f1_score(self.y_test, self.preds_test,
                                average='macro')
        self.cf_train = confusion_matrix(self.y_train, self.preds_train)
        self.cf_test = confusion_matrix(self.y_test, self.preds_test)

    def get_score(self):
        '''
        Computes the score using some metrics for train and test predictions.

        Returns
        -------
        Scores. dict. Dictionary with all the stored metrics
        '''
        scores = {'Accuracy_train': self.acc_train,
                  'Accuracy_test': self.acc_test,
                  'Balanced_Accuracy_train': self.balanced_acc_train,
                  'Balanced_Accuracy_test': self.balanced_acc_test,
                  'F1_train': self.f1_train,
                  'F1_test': self.f1_test,
                  'CM train': self.cf_train,
                  'CM test': self.cf_test}
        return scores
    
    def get_predictions(self, dataset='test'):
        '''
        Computes the preditions for the selected dataset using the
        fitted model.

        Returns
        -------
        preds_comp. pandas.DataFrame.
        '''
        if dataset == 'train':
            dict_preds = {'y_train': list(self.y_train),
                          'preds': list(self.preds_train)}
            preds_comp = pd.DataFrame(dict_preds)

        if dataset == 'test':
            dict_preds = {'y_test': list(self.y_test),
                          'preds': list(self.preds_test)}
            preds_comp = pd.DataFrame(dict_preds)
        if dataset == 'external':
            preds_comp = pd.DataFrame(self.preds_ext)
            preds_comp = pd.DataFrame(preds_comp)
        return preds_comp

    def get_var_importance(self):
        '''
        Shows the second models' variable importance of the trained model.

        Returns
        -------
        fig. matplotlib.pyplot.figure. Barplot with feature importances
        '''
        features_imp = {}
        self.model_selected = self.model2
        for column, importance in zip(
                self.X_train.columns,
                self.model_selected.feature_importances_):
            features_imp[column] = importance
        features_imp_df_train = (pd.DataFrame.from_dict(features_imp,
                                                        orient='index')
                                 .reset_index()
                                 .sort_values(by=[0], ascending=False))
        features_imp_df_train.columns = ['Variable', 'Importance']
        features_imp_df_train = features_imp_df_train.iloc[0:10, :]
        fig = plt.figure(figsize=(5, 4))
        sns.barplot(y='Variable', x='Importance',
                    data=features_imp_df_train)
        plt.title('Model Variable Importance')
        return fig
