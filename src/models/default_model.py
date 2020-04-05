# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:45:22 2020

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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier


class DefaultSingleModel():

    def __init__(self, X_train, X_test, y_train, y_test,
                 external_data, model_selected):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.external_data = external_data
        self.model_selected = model_selected
        self.select_model()
        self.fit()
        self.predict()
        self.score()

    def select_model(self):
        '''
        Gets the model seleted.

        Returns
        -------
        None.
        '''
        if self.model_selected == 'randomforest':
            self.model = RandomForestClassifier(class_weight='balanced')
        if self.model_selected == 'gradientboosting':
            self.model = GradientBoostingClassifier()
        if self.model_selected == 'adaboost':
            self.model = AdaBoostClassifier()
        if self.model_selected == 'lightgbm':
            self.model = LGBMClassifier(n_jobs=-1, class_weight='balanced')
        if self.model_selected == 'xgboost':
            self.model = XGBClassifier(n_jobs=-1)
            self.target_encoder = LabelEncoder()
            self.y_train = self.target_encoder.fit_transform(self.y_train)
            self.y_test = self.target_encoder.fit_transform(self.y_test)

    def fit(self):
        '''
        Fits the model seleted using train daata.
        Returns
        -------
        None.
        '''
        self.model.fit(self.X_train, self.y_train)

    def inverse_encoding(self, data):
        '''
        Decodes previously encoded data.

        Returns
        -------
        None.
        '''
        data_uncoded = self.target_encoder.inverse_transform(data)
        return data_uncoded

    def predict(self):
        '''
        Computes predictions for each datset using the fitted model.

        Returns
        -------
        None.
        '''
        self.preds_train = self.model.predict(self.X_train)
        self.preds_test = self.model.predict(self.X_test)
        self.preds_ext = self.model.predict(self.external_data)
        if self.model_selected == 'xgboost':
            self.y_train = self.inverse_encoding(self.y_train)
            self.y_test = self.inverse_encoding(self.y_test)
            self.preds_train = self.inverse_encoding(self.preds_train)
            self.preds_test = self.inverse_encoding(self.preds_test)
            self.preds_ext = self.inverse_encoding(self.preds_ext)

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
        self.f1_train = f1_score(
            self.y_train, self.preds_train, average='macro')
        self.f1_test = f1_score(
            self.y_test, self.preds_test, average='macro')
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
                  'Confussion_Matrix_train': self.cf_train,
                  'Confussion_Matrix_test': self.cf_test}
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

    def visualize_predict(self, dataset='test'):
        '''
        Plots the confussion matrix of the seleted predictions dataset.

        Returns
        -------
        fig.figure_. matplotlib.pyplot.figure.
        '''
        if self.model_selected == 'xgboost':
            y_train_aux = self.target_encoder.fit_transform(self.y_train)
            y_test_aux = self.target_encoder.fit_transform(self.y_test)
        else:
            y_train_aux = self.y_train.copy()
            y_test_aux = self.y_test.copy()
        if dataset == 'train':
            fig = plot_confusion_matrix(self.model,
                                        self.X_train,
                                        y_train_aux,
                                        cmap=plt.cm.Blues)
            fig.figure_
        if dataset == 'test':
            fig = plot_confusion_matrix(self.model,
                                        self.X_test,
                                        y_test_aux,
                                        cmap=plt.cm.Blues)
            fig.figure_
        return fig.figure_

    def get_var_importance(self):
        '''
        Shows the variable importance of the trained model.

        Returns
        -------
        fig. matplotlib.pyplot.figure. Barplot with feature importances
        '''
        features_imp = {}
        for column, importance in zip(self.X_train.columns,
                                      self.model.feature_importances_):
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
