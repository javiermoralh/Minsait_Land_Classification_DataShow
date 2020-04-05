# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:44:45 2020

@author: javier.moral.hernan1
"""

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelBinarizer
from src.models.neural_net.createNeuralNetwork import NeuralNetModel


class CascadingEnsemble():

    def __init__(self, X_train, X_test,
                 y_train, y_test, external_data,
                 thres_1, thres_2):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.external_data = external_data
        self.binary_target()
        self.select_models()
        self.thres1 = thres_1
        self.thres2 = thres_2
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
        self.y_train_binary = pd.DataFrame((self.y_train[
            'CLASE'].map(lambda x: 1 if x == 'RESIDENTIAL' else 0)))
    

    def select_models(self):
        '''
        Gets the model seleted.

        Returns
        -------
        None.
        '''
        self.model1 = LogisticRegression(class_weight='balanced',
                                         random_state=42)
        self.model2 = XGBClassifier(n_jobs=-1, max_depth=10, learning_rate=0.01,
                                    n_estimators=100)
        self.NN = NeuralNetModel(self.X_train)
        self.model3 = self.NN.model

    def cascadingfilter(self, model, X, y, threshold):
        preds = pd.DataFrame(model.predict_proba(X))
        preds.set_index(X.index.values, inplace=True)
        train_mask = (preds.iloc[:, 1] <= threshold)
        X_train_filt = X[train_mask]
        y_train_filt = y[train_mask]
        return X_train_filt, y_train_filt, train_mask

    def fit(self):
        '''
        Fits each model with its phase data and sequentially filters 
        the not classied instances to train next classifier.

        Returns
        -------
        None.
        '''
        # Fisrt Model
        self.model1.fit(self.X_train, self.y_train_binary)
        (X_train_filt1,
         y_train_filt1,
         train_mask) = self.cascadingfilter(
             self.model1, self.X_train, self.y_train_binary, self.thres1)
        print('First Model Fitted')
        pred_print = self.model1.predict(self.X_train[~train_mask])
        acc_ = accuracy_score(self.y_train_binary[~train_mask], pred_print)
        cm = confusion_matrix(self.y_train_binary[~train_mask], pred_print)
        print('Acc first model', acc_)
        print('CM first model', cm)
        print(y_train_filt1['CLASE'].value_counts())

        # Second Model
        self.model2.fit(X_train_filt1, y_train_filt1)
        (X_train_filt2,
         y_train_filt2,
         train_mask) = self.cascadingfilter(
             self.model2, X_train_filt1, y_train_filt1, self.thres2)
        print('Second Model Fitted')
        pred_print = self.model2.predict(X_train_filt1[~train_mask])
        acc_ = accuracy_score(y_train_filt1[~train_mask], pred_print)
        cm = confusion_matrix(y_train_filt1[~train_mask], pred_print)
        print('Acc second model', acc_)
        print('CM second model', cm)
        print(y_train_filt2['CLASE'].value_counts())

        # Third Model
        self.binary_encoder = LabelBinarizer()
        y_train_filt2 = self.y_train.loc[y_train_filt2.index.values]
        y_train_filt2_enc = self.binary_encoder.fit_transform(y_train_filt2)
        self.model3.fit(X_train_filt2, y_train_filt2_enc,
                        batch_size=300, class_weight='balanced',
                        epochs=70,
                        verbose=2,
                        validation_split=0.30,
                        callbacks=[self.NN.ES])
        print('Thrid Model Fitted')

    def inverse_encoding(self, data):
        '''
        Creates a binary target where value 1 matches the majority class
        intances and 0 all remaining classes.

        Returns
        -------
        None.
        '''
        data_uncoded = self.binary_encoder.inverse_transform(data)
        return data_uncoded

    def getPredProba(self, model, train, test, ext):
        '''
        Computes the predicted probabilities of the 3 datasets using the
        input model.

        Returns
        -------
        · preds_train. pandas.DataFrame
        · preds_test. pandas.DataFrame
        · preds_ext. pandas.DataFrame
        '''
        preds_train = pd.DataFrame(model.predict_proba(train))
        preds_train.set_index(train.index.values, inplace=True)
        preds_test = pd.DataFrame(model.predict_proba(test))
        preds_ext = pd.DataFrame(model.predict_proba(ext))
        preds_ext.set_index(ext.index.values, inplace=True)
        return preds_train, preds_test, preds_ext

    def predictDataset(self, preds1, preds2, preds3):
        '''
        Computes the final prediction of a dataset combining the single 
        predictions of the 3 trained models by filtering with the selected 
        probability thresholds.

        Returns
        -------
        · predictions. pandas.DataFrame
        '''
        predictions = pd.concat([preds1.iloc[:, 1],
                                 preds2.iloc[:, 1],
                                 preds3], axis=1)
        predictions.columns = ['model1', 'model2', 'model3']
        predictions['preds'] = 'nan'
        for row in predictions.iterrows():
            if row[1]['model1'] >= self.thres1:
                predictions.at[row[0], 'preds'] = 'RESIDENTIAL'
            elif row[1]['model2'] >= self.thres2:
                predictions.at[row[0], 'preds'] = 'RESIDENTIAL'
            else:
                predictions.at[row[0], 'preds'] = row[1]['model3']
        return predictions

    def predict(self):
        '''
        Computes final predictions for each datset using the fitted models.

        Returns
        -------
        None.
        '''
        (preds_train1,
         preds_test1,
         preds_ext1) = self.getPredProba(self.model1, self.X_train,
                                         self.X_test, self.external_data)
        (preds_train2,
         preds_test2,
         preds_ext2) = self.getPredProba(self.model2, self.X_train,
                                         self.X_test, self.external_data)                             
        preds_train3 = pd.DataFrame(
            self.inverse_encoding(self.model3.predict(self.X_train)))
        preds_train3.set_index(self.X_train.index.values, inplace=True)
        preds_test3 = pd.DataFrame(
            self.inverse_encoding(self.model3.predict(self.X_test)))
        preds_test3.set_index(self.X_test.index.values, inplace=True)
        preds_ext3 = pd.DataFrame(
            self.inverse_encoding(self.model3.predict(self.external_data)))
        preds_ext3.set_index(
            self.external_data.index.values, inplace=True)
        preds_train_df = self.predictDataset(
            preds_train1, preds_train2, preds_train3)
        preds_test_df = self.predictDataset(
            preds_test1, preds_test2, preds_test3)
        preds_ext_df = self.predictDataset(
            preds_ext1, preds_ext2, preds_ext3)
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
