#!/usr/bin/env python

import sqlite3 as sq
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

#error handling
import traceback

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn import model_selection

#Regression
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

#Classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Pipeline:

    models_reg=[('Ridge',linear_model.Ridge(alpha=.5)),
            ('OLS',linear_model.LinearRegression()),
            ('RandomForestRegressor',RandomForestRegressor())]

    models_class = [('LogReg', LogisticRegression()), 
              ('SVM', SVC()), 
              ('DecTree', DecisionTreeClassifier()),
              ('KNN', KNeighborsClassifier()),
              ('GaussianNB', GaussianNB())]
    
    def fetch_data(data_path, table):
        conn=sq.connect(db_path)
        try:
            df = pd.read_sql_query("select * from "+table, conn)
        except:
            print("Error reading Database, please check: ",traceback.print_stack())
        return df
    
    def df_clean_up(df):
    """
    Function to fill missing data. 'NA' for object columns. Impute median and dummy column for numeric columns.
    Also to drop duplicated rows.
    :param df: dataframe
    :return: dedup dataframe with filled missing values
    """
        #identify float/int columns once null is dropped
        int_col = []
        for col in df.columns:
            if df[~df[col].isnull()][col].dtype in (np.dtype('int64'), np.dtype('int32'), np.dtype('float64'), np.dtype('float32')):
                int_col.append(col)

        #for missing values in category, fillna with 'missing'
        df_clean_cat = df.drop(int_col,axis=1).fillna('NA')

        #for missing values in numeric columns, create dummy column for missing values then fillna with median
        df_num = df[int_col]
        for col in df_num.columns:
            if df_num[col].isnull().sum() > 0:
                df_num[col+'miss'] = df_num[col].isna()*1
        df_num = df_num.fillna(df_num.median())
        df = pd.concat([df_clean_cat,df_num],axis=1)
        df.drop_duplicates(inplace=True)
        return df

    def changetype(df,col,dtype):
        """
        Function to change
        :param df: dataframe
        :param col: string list of column names to change 
        :return: dataframe with filled missing values
        """
        return df[col].astype(dtype, copy=True, errors='raise')
    


    def scale_params(X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def prepare_params(df,target_var, test_size=0.4, scale=False, target_var_type = "cont"):

        temp_df=df.copy(deep=True)

        y_px=temp_df.pop(target_var)
        X_px=temp_df

        if(scale):
            X_transformed_px=scale_params(X_px)
        else:
            X_transformed_px=X_px


        X_train_px, X_valid_px, y_train_px, y_valid_px = train_test_split(X_transformed_px, y_px, test_size=test_size)

        if(target_var_type=="disc"):
            model=RandomForestClassifier(n_estimators = 100,
                               n_jobs = -1,
                               oob_score = True,
                               bootstrap = True,)
        elif(target_var_type=="cont"):
                model=RandomForestRegressor(n_estimators = 100,
                               n_jobs = -1,
                               oob_score = True,
                               bootstrap = True,
                               )


        rf_px = model
        rf_px.fit(X_train_px, y_train_px)

        print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(rf_px.score(X_train_px, y_train_px), 
                                                                                                     rf_px.oob_score_,
                                                                                                     rf_px.score(X_valid_px, y_valid_px)))

        perm_px = PermutationImportance(rf_px, cv = None, refit = False, n_iter = 50).fit(X_train_px, y_train_px)

        eli5.show_weights(perm_px, feature_names = X_train_px.columns.tolist())

        sel = SelectFromModel(perm_px, threshold=0.005, prefit=True)
        X_trans_px = sel.transform(X_train_px)
        X_valid_px = sel.transform(X_valid_px)

        return X_trans_px, X_valid_px, y_train_px, y_valid_px
    
    


    def build_regress_model(x,y,model_list,scoring_param="explained_variance"):
        outcome = []
        model_names = []
        model_dict={}

        for model_name, model in models:
            k_fold_validation = model_selection.KFold(n_splits=10)
            results = model_selection.cross_val_score(model, X_trans_px, y_px, cv=k_fold_validation, scoring=scoring_param)
            outcome.append(results)
            model_names.append(model_name)
            output_message = "%s| Mean=%f STD=%f" % (model_name, results.mean(), results.std())
            print(output_message)
            model_dict[model_name]=model

        fig = plt.figure(figsize=(20,20))


        fig.suptitle('Comparison by '+scoring_param )
        ax = fig.add_subplot(111)
        plt.boxplot(outcome)
        ax.set_xticklabels(model_names)
        plt.show()

        return model_dict
    
    def build_classification_model(x_train,x_test,y_train,y_test,class_models_list=[('GaussianNB', GaussianNB())]):
        output_list=[]
        for model_name,model in class_models_list:
            temp_model=model.fit(X_trans_bin,y_train_bin)
            output_list.append(tuple(model_name,model,calculate_accuracy(temp_model,x_test,y_test) )
        return output_list

        
    def calculate_accuracy(model,x_test,y_test):    
        accuracy_matrix=np.where(y_test==model.predict(x_test),1,0 )
        return accuracy_matrix.sum()/len(accuracy_matrix)
    
    def read_configfile(configfile_path):
        try:
            configfile_df= pd.read_csv(configfile_path)

            return configfile_df
        except:
            print("Error, unable to read config file")
                               
    def transform(path,configfile_path):
        configfile=read_configfile(configfile_path)

        df=fetch_data(data_path, configfilefile.table)
        df=df_clean_up(df)
        for row in configfile[["col_list","dtype"]].itertuples():
                               df=changetype(df,row[0],row[1])
        X_trans_px, X_valid_px, y_train_px, y_valid_px= prepare_params(df,configfile.target_var, test_size=configfile.test_size, scale=configfile.scale, target_var_type = configfile.target_var_type)
        
        model_list=[]

        if(configfile.target_var_type=="cont"):
                               model_list=build_regress_model(x,y,model_list,scoring_param=configfile.scoring_param)
        elif(configfile.target_var_type=="disc"):
                               model_list=build_classification_model(x_train,x_test,y_train,y_test,class_models_list=[('GaussianNB', GaussianNB())])
        
        return model_list