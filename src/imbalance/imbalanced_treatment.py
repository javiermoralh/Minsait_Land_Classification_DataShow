# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:05:17 2020

@author: javier.moral.hernan1
"""

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler


class UnbalancedDataSampling():

    def __init__(self, X_train, y_train, method, num_obs):
        self.X_train = X_train
        self.y_train = y_train
        if method == 'smote_enn':
            self.Smote_ENN()
        if method == 'smote_enn_manual':
            self.Smote_ENN_manual(num_obs)
        if method == 'smote_enn_manual_label':
            self.Smote_ENN_manual_labelencoder(num_obs)
        if method == 'smote_tomek':
            self.Smote_Tomek()
        if method == 'random_under':
            self.Random_UnderSampling(num_obs)

    def Smote_ENN(self):
        '''
        First oversamples the minority classes using SMOTE and then cleans
        all the data using ENN.

        Returns
        -------
        None.
        '''
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        sme = SMOTEENN(random_state=2020)
        (self.X_train_balanced,
         self.y_train_balanced) = sme.fit_resample(X_train, y_train)

    def Smote_ENN_manual(self, num_obs):
        '''
        First oversamples the minority classes using SMOTE based on the number
        of instances selected and then cleans all the data using ENN.

        Returns
        -------
        None.
        '''
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        sm = SMOTEENN(random_state=2020,
                      sampling_strategy={'INDUSTRIAL': num_obs,
                                         'PUBLIC': num_obs,
                                         'RETAIL': num_obs,
                                         'OFFICE': num_obs,
                                         'OTHER': num_obs,
                                         'AGRICULTURE': num_obs})
        (self.X_train_balanced,
         self.y_train_balanced) = sm.fit_resample(X_train, y_train)

    def Smote_ENN_manual_labelencoder(self, num_obs):
        '''
        First oversamples the minority classes using SMOTE based on the number
        of instances selected and then cleans all the data using ENN.

        Returns
        -------
        None.
        '''
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        sme = SMOTEENN(random_state=2020,
                       sampling_strategy={1: num_obs,
                                          2: num_obs,
                                          3: num_obs,
                                          4: num_obs,
                                          6: num_obs,
                                          0: num_obs})
        (self.X_train_balanced,
         self.y_train_balanced) = sme.fit_resample(X_train, y_train)

    def Smote_Tomek(self):
        '''
        First oversamples the minority classes using SMOTE based on the number
        of instances selected and then cleans all the data using Tomek Links.

        Returns
        -------
        None.
        '''
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()
        smt = SMOTETomek(random_state=2020)
        (self.X_train_balanced,
         self.y_train_balanced) = smt.fit_resample(X_train, y_train)

    def Random_UnderSampling(self, num_obs):
        '''
        Undersamples the majority class randomly based on the number
        of instances selected.

        Returns
        -------
        None.
        '''
        X_train = self.X_train.copy()
        y_train = self.y_train.copy()

        rus = RandomUnderSampler(random_state=42,
                                 sampling_strategy={'RESIDENTIAL': num_obs})
        (self.X_train_balanced,
         self.y_train_balanced) = rus.fit_resample(X_train, y_train)

    def getData(self):
        '''
        Returns your sampled datasets.

        Returns
        -------
        · X_train_balanced. pandas.DataFrame
        · y_train_balanced. pandas.DataFrame
        '''
        return self.X_train_balanced, self.y_train_balanced
