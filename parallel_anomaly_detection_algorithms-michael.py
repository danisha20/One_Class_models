# Basic libraries
import numpy as np 
import pandas as pd
import os
import time
# Multiprocessing
import multiprocessing as mp
# Preprocessing libraries
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Anomaly detection algorithms
# BRM github
import brminer
# ocSVM sklearn
from sklearn.svm import OneClassSVM
# COF pyod
from pyod.models.cof import COF
# ABOD pyod
from pyod.models.abod import ABOD
# MOGAAL pyod
from pyod.models.mo_gaal import MO_GAAL
# SOGAAL pyod
from pyod.models.so_gaal import SO_GAAL
# OCKRA github
import sys
sys.path.append(r'C:\Users\mzenk\Google Drive\ITESM\Maestría\Semestre 3\ML2\Assignment5\m-OCKRA')
import m_ockra
# VAR LMDD pyOD
from pyod.models.lmdd import LMDD

from pyod.models.lscp import LSCP
from pyod.models.lof import LOF

class AnomalyTester():

    def __init__(self, model, model_name, rootDir, scaler=None):
        self.model = model
        self.model_name = model_name
        self.scaler = scaler
        self.rootDir = rootDir

        self.run_all_files()

    def importdata(self, trainfile_path, testfile_path):
        '''
        Imports the data of the train and test files
        '''
        self.df_train = pd.read_csv(trainfile_path, sep= ',') 
        self.df_test = pd.read_csv(testfile_path, sep= ',') 
        
        return self.df_train, self.df_test

    # Function to split target from data 
    def splitdataset(self, train, test):
        '''
        It splits the dataset
        '''
        # Call the OneHotEncoder class
        ohe = OneHotEncoder(sparse=True)
        # Concatenate all data 
        allData = pd.concat([train, test], ignore_index=True, sort =False, axis=0)
        # Omit the response variable (the last column)
        AllDataWihoutClass = allData.iloc[:, :-1]
        # Select only nominal types
        AllDataWihoutClassOnlyNominals = AllDataWihoutClass.select_dtypes(include=['object'])
        # Select numerical types
        AllDataWihoutClassNoNominals = AllDataWihoutClass.select_dtypes(exclude=['object'])
        # one hot encoding of all nominals
        encAllDataWihoutClassNominals = ohe.fit_transform(AllDataWihoutClassOnlyNominals)
        # get data without class to a dataframe
        encAllDataWihoutClassNominalsToPanda = pd.DataFrame(encAllDataWihoutClassNominals.toarray())
        encAllDataWihoutClassNominalsToPanda = encAllDataWihoutClassNominalsToPanda.astype(object)
        # If the dataset contain more tha 0 columns of nominals then concatenate if not pass them      
        if AllDataWihoutClassOnlyNominals.shape[1] > 0:
            codAllDataAgain = pd.concat([encAllDataWihoutClassNominalsToPanda,
                                AllDataWihoutClassNoNominals], ignore_index=True, sort =False, axis=1)
        else:
            codAllDataAgain = AllDataWihoutClass
        # Seperating the target variable 
        self.X = codAllDataAgain[:len(allData)]
        self.y_real = allData.values[:, -1]
        self.y_train = ['negative' for _ in range(len(allData))]

        return self.X, self.y_real, self.y_train
    
    def scaler_transform(self, X):
        '''
        Transform the NON-object type data to the selected scaler
        '''
        X_withoutobj = X.select_dtypes(exclude=['object'])
        # Fit transform the scaler if there are objects in dataset
        if X_withoutobj.shape[1] > 0:
            X_withoutobj = pd.DataFrame(self.scaler.fit_transform(X_withoutobj[X_withoutobj.columns]),
                                            index=X_withoutobj.index,
                                            columns=X_withoutobj.columns)
            # Concatenate the standard
            idx = X.columns.get_indexer(X.select_dtypes('object').columns)
            for i in range(len(idx)):
                X_withoutobj.insert(i, i, X[[i]], True)
            self.X = X_withoutobj
        else:
            self.X = X

        return self.X

    def get_score(self, X, clf_object):
        '''
        Gets the score
        '''
        try:
            self.y_score = clf_object.score_samples(X) 
        except AttributeError:
            try:
                self.y_score = clf_object.decision_function(X)
            except:
                self.y_score = clf_object.decision_function(X.values)
    
        return self.y_score
    
    def run_all_files(self):
        '''
        Iterates through all files in a root directory, trains and evaluates all data.
        '''
        print('Starting '+self.model_name)
        data = {'folder_name':[], self.model_name+'_auc':[], self.model_name+'_avgprecision':[]}
        for dirName, subdirList, fileList in os.walk(self.rootDir):
            if len(fileList) > 0:
                arr_auc = []
                arr_ave_precision = []
                arr_folder_name = dirName.split("\\")
                folder_name = arr_folder_name[len(arr_folder_name) - 1]
                completed_name = folder_name + "-5-"
                for i in range(1, int(len(fileList) / 2) + 1):
                    #print('Dataset in process...') 
                    trainFile = str(dirName) + '\\' + completed_name + str(i) +"tra.csv"
                    testFile = str(dirName) + '\\' + completed_name + str(i) +"tst.csv"
                    print('Model: '+self.model_name+' Train File: ' + completed_name + str(i))
                    # Loading the data
                    df_train, df_test = self.importdata(trainFile, testFile)
                    if "LOCI" in self.model_name:
                        df_train.columns = range(df_train.shape[1])
                        df_test.columns = range(df_test.shape[1])
                    # Split the dataset
                    X, y_real, y_train = self.splitdataset(df_train, df_test)
                    # Scale the data if indicated
                    if self.scaler != None:
                        X = self.scaler_transform(X)
                    # Fit the model (some linear models require only float or int)
                    if self.model_name == 'AAD_LMDD' or self.model_name == 'IQR_LMDD' or self.model_name == 'VAR_LMDD' or self.model_name == 'AAD_LMDD_std' or self.model_name == 'AAD_LMDD_mm' or self.model_name == 'IQR_LMDD_std' or self.model_name == 'IQR_LMDD_mm' or self.model_name == 'VAR_LMDD_std' or self.model_name == 'VAR_LMDD_mm':
                        X = X.astype(float)
                    if "ABOD" in self.model_name:
                        X = X.astype(float)
                    error = True
                    while(error): #To consider failures because of stochastic elements
                        try:
                            self.model.fit(X, y_train) # some algorithms require y_train, but it is all neg.
                            # Predict
                            y_score_classif = self.get_score(X, self.model)
                            error = False 
                        except: pass
                    # Get the evaluation
                    y_score_classif = np.nan_to_num(y_score_classif)
                    auc = metrics.roc_auc_score(y_real,  y_score_classif) # y originals
                
                    ave_precision = metrics.average_precision_score(y_real, # y originals
                                                                y_score_classif,
                                                                pos_label='positive')
                    # Append the results
                    arr_auc.append(1 - auc if auc < 0.5 else auc)
                    arr_ave_precision.append(1-ave_precision if ave_precision<0.5 else ave_precision)
                # Calculate the average of scores
                result_auc = sum(arr_auc) / len(arr_auc)
                result_ave_precision = sum(arr_ave_precision) / len(arr_ave_precision)
                # Save them to the resulting dataframe
                data['folder_name'].append(folder_name)
                data[self.model_name+'_auc'].append(result_auc)
                data[self.model_name+'_avgprecision'].append(result_ave_precision)
            else:
                pass
            # Save it in a dataframe
            df = pd.DataFrame(data, columns = ['folder_name', self.model_name+'_auc', self.model_name+'_avgprecision'])
            # Export the document to csv
            if "_mm" in self.model_name:
                results_path = os.path.abspath("Results\\MinMax\\")
            elif "_std" in self.model_name:
                results_path = os.path.abspath("Results\\Norm\\")
            else:
                results_path = os.path.abspath("Results\\No scaler\\")
            df.to_csv(results_path+"\\"+self.model_name+'_results.csv', index=False)
        print('\nFinished '+self.model_name)

        return None

if __name__ == '__main__':
    # Specify the root directory
    datasets_path = "Anomaly_Datasets_csv"
    rootDir = os.path.abspath(datasets_path)
    # specify the random state
    rs = 10
    # Save how to run the models

    detector_list = [LOF(), LOF()]

    models = [
            # BRM github
            (brminer.BRM(), 'BRM'),
            # ocSVM sklearn
            (OneClassSVM(gamma='auto'), 'ocSVM'),
            # COF pyod
            (COF(contamination=0.1, n_neighbors=20),'COF'),
            # ABOD pyod
            (ABOD(contamination=0.1, n_neighbors=5, method='fast'), 'ABOD'),
            # MO_GAAL pyod
            (MO_GAAL(k=10, stop_epochs=20, lr_d=0.01, lr_g=0.0001, decay=1e-06, momentum=0.9, contamination=0.1),'MO_GAAL'),
            # SO_GAAL pyod
            (SO_GAAL(stop_epochs=20, lr_d=0.01, lr_g=0.0001, decay=1e-06, momentum=0.9, contamination=0.1),'SO_GAAL'),
            # OCKRA github
            (m_ockra.m_OCKRA(), 'OCKRA'),
            # VAR LMDD pyOD
            (LMDD(dis_measure='var',random_state=rs),'VAR_LMDD'),
            # LOCI pyod
            (LSCP(detector_list, local_region_size=30, local_max_features=1.0, n_bins=10, random_state=None, contamination=0.1),'LSCP')]

    # Select the model location with i to run
    i = 8
    had_error = []
    # Initialize the class anomaly
    #for i in range(1,8):
    #    try:
    #        AnomalyTester(models[i][0],models[i][1], rootDir)
    #        AnomalyTester(models[i][0],models[i][1]+'_std', rootDir, StandardScaler())
    #        AnomalyTester(models[i][0],models[i][1]+'_mm', rootDir, MinMaxScaler())
    #    except:
    #        had_error.append(i)
    #        continue

    AnomalyTester(models[i][0],models[i][1], rootDir)
    AnomalyTester(models[i][0],models[i][1]+'_std', rootDir, StandardScaler())
    AnomalyTester(models[i][0],models[i][1]+'_mm', rootDir, MinMaxScaler())
    
    print("These had errors:")
    print(had_error)

    # Had errors:
    # LSCP 
    # OCKRA