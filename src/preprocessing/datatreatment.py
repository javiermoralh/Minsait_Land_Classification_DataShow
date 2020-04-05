# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:09:25 2020

@author: javier.moral.hernan1
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import DataConversionWarning
import category_encoders as ce
from src.imputation.missings import Missings
from sklearn.preprocessing import LabelEncoder
from src.aux_functs.harvesianDist import single_pt_haversine
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from src.aux_functs.mapCoordinates import rangeTranfer
from src.aux_functs.mapCoordinates import getCenterPoint
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


class DataTreatment():

    def __init__(self, data, external_data):
        self.X = data.drop(labels=['ID', 'CLASE'], axis=1)
        self.external_data = external_data.drop(labels=['ID'], axis=1)
        self.y = data[['CLASE']]
        (self.X_train, self.X_test,
         self.y_train, self.y_test) = self.splitData()
        self.imputeMissings('simple')
        self.dropConstants()
        self.featureEngineeringStatictics()
        self.clusteringCommunities()
        self.categoricalEncoding()
        self.featureEngineeringCoordinates()
        self.featureScaling()
        self.deleteCorrelations(0.97)

    def splitData(self):
        '''
        This functions splits the data into Train and Test

        Parameters
        ----------
        data : TYPE pandas.DataFrame
            input data.

        Returns
        -------
        trainX : TYPE pandas.DataFrame
            features of the training sample.
        testX : TYPE
            features of the testing sample.
        trainY : TYPE
            target of the training sample..
        testY : TYPE
            target of the testing sample.

        '''
        print('Spliting train and test')
        X = self.X
        y = self.y

        (X_train, X_test,
         y_train, y_test) = (train_test_split(X, y,
                                              test_size=0.30,
                                              stratify=y,
                                              random_state=2020))

        return (X_train, X_test, y_train, y_test)

    def imputeMissings(self, method='simple'):
        '''
        This function imputes the missing values of all variables using
        the class Missings and the method selected.

        Returns
        -------
        None.
        '''
        print('Imputing missing...')
        imputer = Missings(self.X_train, self.X_test, self.external_data)
        if method == 'simple':
            (self.X_train, self.X_test,
             self.external_data) = imputer.simple_imputation()
        if method == 'datawig':
            (self.X_train, self.X_test,
             self.external_data) = imputer.datawig_imputation()
        if method == 'delete':
            (self.X_train, self.X_test,
             self.external_data) = imputer.delete_missings()

    def dropConstants(self):
        '''
        This function drops the constant columns in the Training set and maps
        it to the test set.

        Returns
        -------
        None.
        '''
        print('Cleaning data...')
        data2keep = (self.X_train != self.X_train.iloc[0]).any()
        self.X_train = self.X_train.loc[:, data2keep]
        self.X_test = self.X_test.loc[:, data2keep]
        self.external_data = self.external_data.loc[:, data2keep]

    def featureScaling(self):
        '''
        Fits a min-max scaler using the train data and transforms train,
        test and external data using the scaler.

        Returns
        -------
        None.

        '''
        print('Scaling variables...')
        warnings.filterwarnings(action='ignore',
                                category=DataConversionWarning)
        scaler = MinMaxScaler()
        numeric = list(self.X_train.select_dtypes(include=np.number))
        self.X_train[numeric] = (scaler
                                 .fit_transform(self.X_train[numeric]))
        self.X_test[numeric] = (scaler
                                .transform(self.X_test[numeric]))
        self.external_data[numeric] = (scaler
                                       .transform(self.external_data[numeric]))

        warnings.filterwarnings(action='default',
                                category=DataConversionWarning)

    def categoricalEncoding(self):
        '''
        Fits a catBoostEncoder using the train data and transforms train,
        test and external categorical features data using the encoder.

        Returns
        -------
        None.

        '''
        print('Encoding categorical variable')
        target_encoder = LabelEncoder()
        y_train = target_encoder.fit_transform(self.y_train)
        catEncoder = ce.CatBoostEncoder(drop_invariant=True, return_df=True,
                                        random_state=2020)
        catEncoder.fit(self.X_train, y_train)
        self.X_train = catEncoder.transform(self.X_train)
        self.X_test = catEncoder.transform(self.X_test)
        self.external_data = catEncoder.transform(self.external_data)

    def featureEngineeringStatictics(self):
        '''
        Creates some new features based on computing variable statistics.

        Returns
        -------
        None.

        '''
        print('Creating new variables...')
        var_list_red = ['Q_R_4_0_0', 'Q_R_4_0_1', 'Q_R_4_0_2', 'Q_R_4_0_3',
                        'Q_R_4_0_4', 'Q_R_4_0_5', 'Q_R_4_0_6', 'Q_R_4_0_7',
                        'Q_R_4_0_8', 'Q_R_4_0_9', 'Q_R_4_1_0']
        var_list_green = ['Q_G_3_0_0', 'Q_G_3_0_1', 'Q_G_3_0_2', 'Q_G_3_0_3',
                          'Q_G_3_0_4', 'Q_G_3_0_5', 'Q_G_3_0_6', 'Q_G_3_0_7',
                          'Q_G_3_0_8', 'Q_G_3_0_9', 'Q_G_3_1_0']
        var_list_blue = ['Q_B_2_0_0', 'Q_B_2_0_1', 'Q_B_2_0_2', 'Q_B_2_0_3',
                         'Q_B_2_0_4', 'Q_B_2_0_5', 'Q_B_2_0_6', 'Q_B_2_0_7',
                         'Q_B_2_0_8', 'Q_B_2_0_9', 'Q_B_2_1_0']
        canal_vars = {'red': var_list_red,
                      'green': var_list_green,
                      'blue': var_list_blue}
        datasets = [self.X_train, self.X_test, self.external_data]
        for canal, var_list in canal_vars.items():
            for data in datasets:
                data['mean_pct_log_' +
                     canal] = (np.log(data[var_list]+1)
                               .pct_change(axis=1)
                               .iloc[:, 2:]
                               .mean(axis=1))
                data['std_pct_log_' +
                     canal] = (np.log(data[var_list]+1)
                               .pct_change(axis=1)
                               .iloc[:, 2:]
                               .std(axis=1))
                data['median_pct_log_' +
                     canal] = (np.log(data[var_list]+1)
                               .pct_change(axis=1)
                               .iloc[:, 2:]
                               .median(axis=1))
        for data in datasets:
            data['AREA_log'] = np.log(data['AREA'])
            decils = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
            for decil in decils:
                data['CADQUAL_AREA_' + str(decil)] = np.nan
                for i in data.groupby('CADASTRALQUALITYID'):
                    for decil in decils:
                        perc = np.percentile(i[1].AREA, decil)
                        mask_idx = list(i[1].AREA.index.values)
                        data.loc[mask_idx,
                                 'CADQUAL_AREA_' + str(decil)] = perc
            data['IDR_CADQUAL'] = data['CADQUAL_AREA_99'] - data['CADQUAL_AREA_1']
        for data in datasets:
            decils = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
            for decil in decils:
                data['MAXBUILD_AREA_' + str(decil)] = np.nan
                for i in data.groupby('MAXBUILDINGFLOOR'):
                    for decil in decils:
                        perc = np.percentile(i[1].AREA, decil)
                        mask_idx = list(i[1].AREA.index.values)
                        data.loc[
                            mask_idx, 'MAXBUILD_AREA_' + str(decil)] = perc
            data['IDR_MAXBUILD'] = data[
                'MAXBUILD_AREA_99'] - data['MAXBUILD_AREA_1']

    def clusteringCommunities(self):
        '''
        Creates some new features based on clustering techniques using
        coordinates variables.

        Returns
        -------
        None.

        '''
        print('Clustering coordinates')
        scaler = MinMaxScaler()
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        external = self.external_data.copy()
        data_map = [X_test, external]
        X_train.loc[
            :, ['X', 'Y']] = scaler.fit_transform(X_train.loc[:, ['X', 'Y']])
        for data in data_map:
            data.loc[
                :, ['X', 'Y']] = scaler.transform(data.loc[:, ['X', 'Y']])

        # Clustering using K-Means and Assigning Clusters to our Data
        kmeans = KMeans(n_clusters=15, init='k-means++')
        kmeans.fit(X_train.loc[:, ['X', 'Y']])
        X_train['comunity'] = kmeans.fit_predict(X_train.loc[:, ['X', 'Y']])
        for data in data_map:
            data['comunity'] = kmeans.predict(data.loc[:, ['X', 'Y']])
        centers = (pd.DataFrame(kmeans.cluster_centers_)
                   .reset_index()
                   .rename(columns={'index': 'comunity',
                                    0: 'X_comunity',
                                    1: 'Y_comunity'}))
        X_train = (
            X_train.reset_index()
            .merge(centers, 'left', 'comunity').set_index('index'))
        X_test = (
            X_test.reset_index()
            .merge(centers, 'left', 'comunity').set_index('index'))
        external = (
            external.reset_index()
            .merge(centers, 'left', 'comunity').set_index('index'))
        for data in [X_train, X_test, external]:
            data['dist_communities'] = np.sqrt(
                (data.X_comunity-data.X) ** 2 +
                (data.Y_comunity-data.Y) ** 2)
        self.X_train[[
            'comunity', 'X_comunity',
            'Y_comunity',
            'dist_communities']] = X_train[['comunity', 'X_comunity',
                                            'Y_comunity', 'dist_communities']]
        self.X_test[[
            'comunity', 'X_comunity',
            'Y_comunity',
            'dist_communities']] = X_test[['comunity', 'X_comunity',
                                           'Y_comunity', 'dist_communities']]
        self.external_data[[
            'comunity', 'X_comunity',
            'Y_comunity',
            'dist_communities']] = external[['comunity', 'X_comunity',
                                             'Y_comunity', 'dist_communities']]

    def featureEngineeringCoordinates(self):
        '''
        Creates some new features based on coordinates variables.

        Returns
        -------
        None.

        '''
        print('Creating coordinates variables...')
        datasets = [self.X_train, self.X_test, self.external_data]
        for data in datasets:
            data['X'] = data['X'].astype(float)
            data['Y'] = data['Y'].astype(float)
            data['X'] = rangeTranfer(list(data['X']), [-3.898044, -3.507131])
            data['Y'] = rangeTranfer(list(data['Y']), [40.337813, 40.545617])
            center = getCenterPoint(data, 'X', 'Y')
            data['Harvesian_dist'] = [single_pt_haversine(y, x)
                                      for y, x in zip(data.Y, data.X)]
            data['Euclidian_dist'] = [np.linalg.norm(
                np.array((x, y))-np.array((center[0], center[1])))
                                      for x, y in zip(data.X, data.Y)]
            data['comb_coord1'] = np.cos(data.Y)*np.cos(data.X)
            data['comb_coord2'] = np.cos(data.Y)*np.sin(data.X)
            data['comb_coord3'] = np.sin(data.Y)
            # data.drop(labels=['X', 'Y'], axis=1, inplace=True)

    def deleteCorrelations(self, threshold):
        '''
        Deletes highly correlated variables.
        Out of each pair of variables, only the one with the lowest prediciton
        power will be deleted

        Returns
        -------
        None.

        '''
        print('Deleting Correlated Variables...')
        numeric = list(self.X_train.select_dtypes(include=np.number).columns)
        corrMatrixCont = self.X_train[numeric].corr().abs()
        np.fill_diagonal(corrMatrixCont.values, np.nan)
        booleans = corrMatrixCont > threshold
        variables = list(booleans.loc[booleans.any(), booleans.any()].columns)
        aux = pd.DataFrame()
        for var in tqdm(variables, position=0):
            clf = (LogisticRegression(random_state=0, solver='lbfgs').
                   fit(np.array(self.X_train[var]).reshape(-1, 1),
                       np.array(self.y_train)))
            aux[var] = (pd.Series(clf.score(np.array(self.X_train[var])
                                            .reshape(-1, 1),
                                            np.array(self.y_train))))

        toDelete = []
        while booleans.any().any():
            var = aux[variables].min().index[0]
            toDelete.append(var)
            booleans[var] = False
            booleans.loc[var, :] = False
            variables = list(booleans.loc[booleans.any(),
                                          booleans.any()].columns)
        self.cols_to_delete = toDelete
        self.X_train = self.X_train.drop(columns=toDelete, axis=1)
        self.X_test = self.X_test.drop(columns=toDelete, axis=1)
        self.external_data = self.external_data.drop(columns=toDelete, axis=1)

    def getData(self):
        '''
       Returns the proprecessed datasets.

        Returns
        -------
        · X_train: pandas.DataFrame.
        · y_train: pandas.DataFrame.
        · X_test: pandas.DataFrame.
        · y_test: pandas.DataFrame.
        · external_data: pandas.DataFrame.
        '''
        return (self.X_train, self.X_test, self.y_train,
                self.y_test, self.external_data)
