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
from sklearn.ensemble import IsolationForest
from pyod.models.lmdd import LMDD
from pyod.models.copod import COPOD
from pyod.models.feature_bagging import FeatureBagging
from sklearn.covariance import EllipticEnvelope
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.sos import SOS
from pyod.models.mcd import MCD
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.sod import SOD
from pyod.models.xgbod import XGBOD
from pyod.models.pca import PCA
from pyod.models.loci import LOCI
# Neural network based outlier detection
from pyod.models.vae import VAE
from pyod.models.auto_encoder import AutoEncoder

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
                arr_folder_name = dirName.split("/")
                folder_name = arr_folder_name[len(arr_folder_name) - 1]
                completed_name = folder_name + "-5-"
                for i in range(1, int(len(fileList) / 2) + 1):
                    #print('Dataset in process...') 
                    trainFile = str(dirName) + '/' + completed_name + str(i) +"tra.csv"
                    testFile = str(dirName) + '/' + completed_name + str(i) +"tst.csv"
                    print('Model: '+self.model_name+' Train File: ' + completed_name + str(i))
                    # Loading the data
                    df_train, df_test = self.importdata(trainFile, testFile)
                    # Split the dataset
                    X, y_real, y_train = self.splitdataset(df_train, df_test)
                    # Scale the data if indicated
                    if self.scaler != None:
                        X = self.scaler_transform(X)
                    # Fit the model (some linear models require only float or int)
                    if self.model_name == 'AAD_LMDD' or self.model_name == 'IQR_LMDD' or self.model_name == 'AAD_LMDD_std' or self.model_name == 'AAD_LMDD_mm' or self.model_name == 'IQR_LMDD_std' or self.model_name == 'IQR_LMDD_mm':
                        X = X.astype(float)
                    self.model.fit(X, y_train) # some algorithms require y_train, but it is all neg.
                    # Predict
                    y_score_classif = self.get_score(X, self.model) 
                    # Get the evaluation
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
            df.to_csv(self.model_name+'_results.csv', index=False)
        print('\nFinished '+self.model_name)

        return None

if __name__ == '__main__':
    # Specify the root directory
    rootDir = 'G:/My Drive/Github/ml-group-col/One-Class-models/Anomaly_Datasets_csv/'
    # specify the random state
    rs = 10
    # Save how to run the models
    models = [(IsolationForest(random_state=rs),'ISOF'),
                (EllipticEnvelope(random_state=rs),'EE'),
                (LMDD(dis_measure='aad', random_state=rs),'AAD_LMDD'),
                (COPOD(),'COPOD'),
                (FeatureBagging(combination='average', random_state=rs),'AVE_Bagging'), # n_jobs
                (LMDD(dis_measure='iqr',random_state=rs),'IQR_LMDD'),
                (KNN(method='largest'),'Largest_KNN'), # n_jobs
                (LODA(),'LODA'),
                (FeatureBagging(combination='max', n_jobs=-1, random_state=rs),'MAX_Bagging'),
                (MCD(random_state=rs),'MCD'),
                (XGBOD(random_state=rs),'XGBOD'), # n_jobs
                (GaussianMixture(random_state=rs),'GMM'),
                (LocalOutlierFactor(novelty=True),'LOF'),
                (KNN(method='median'),'Median_KNN'), # n_jobs
                (KNN(method='mean'),'Avg_KNN'), # n_jobs
                (CBLOF(n_clusters=10,random_state=rs),'CBLOF'),
                (HBOS(),'HBOS'), 
                (SOD(), 'SOD'),
                (PCA(random_state=rs),'PCA'),
                (VAE(encoder_neurons=[3,4,3], decoder_neurons=[3,4,3],random_state=rs),'VAE'),
                (AutoEncoder(hidden_neurons=[3, 4, 4, 3], verbose=0,random_state=rs),'AE')]
    # Start the counter of time
    st = time.time()
    # Initialize the pool class with the number of required CPU's
    pool = mp.Pool(mp.cpu_count())
    # StarMap method
    pool.starmap_async(AnomalyTester, [(models[i][0],models[i][1], rootDir) for i in range(len(models))]).get()
    pool.close()
    # Finish the counter of time
    end = time.time()
    # Print the needed time to compute
    print('Time: '+str(round(end-st,2))+' seconds.')