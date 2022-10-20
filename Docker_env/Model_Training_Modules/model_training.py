'''
Author: Liaw Yi Xian
Last Modified: 20th October 2022
'''

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import optuna
import joblib
import time
import shap
from BorutaShap import BorutaShap
from featurewiz import FeatureWiz
from yellowbrick.model_selection import LearningCurve
import feature_engine.selection as fes
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.polynomial import PolynomialEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.backward_difference import BackwardDifferenceEncoder
from category_encoders import CatBoostEncoder
from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
import datetime
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTEN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.model_selection import cross_validate, StratifiedKFold, learning_curve
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, precision_score, recall_score, make_scorer, f1_score,ConfusionMatrixDisplay, classification_report, PrecisionRecallDisplay, average_precision_score, precision_recall_curve, pairwise_distances
from sklearn.cluster import AffinityPropagation
from Application_Logger.logger import App_Logger


random_state=120


class model_trainer:


    def __init__(self, file_object):
        '''
            Method Name: __init__
            Description: This method initializes instance of model_trainer class
            Output: None

            Parameters:
            - file_object: String path of logging text file
        '''
        self.file_object = file_object
        self.log_writer = App_Logger()
        self.optuna_selectors = {
            'LogisticRegression': {'obj': model_trainer.lr_objective,'clf': LogisticRegression(random_state=random_state)},
            'LinearSVC': {'obj': model_trainer.svc_objective, 'clf': LinearSVC(random_state=random_state)},
            'KNeighborsClassifier': {'obj': model_trainer.knn_objective, 'clf': KNeighborsClassifier()},
            'GaussianNB': {'obj': model_trainer.gaussiannb_objective, 'clf': GaussianNB()},
            'DecisionTreeClassifier': {'obj': model_trainer.dt_objective, 'clf': DecisionTreeClassifier(random_state=random_state)},
            'RandomForestClassifier': {'obj': model_trainer.rf_objective, 'clf': RandomForestClassifier(random_state=random_state)},
            'ExtraTreesClassifier': {'obj': model_trainer.et_objective, 'clf': ExtraTreesClassifier(random_state=random_state)},
            'AdaBoostClassifier': {'obj': model_trainer.adaboost_objective, 'clf': AdaBoostClassifier(random_state=random_state)},
            'GradientBoostingClassifier': {'obj': model_trainer.gradientboost_objective, 'clf': GradientBoostingClassifier(random_state=random_state)},
            'XGBClassifier': {'obj': model_trainer.xgboost_objective, 'clf': XGBClassifier(random_state=random_state)},
            'LGBMClassifier': {'obj': model_trainer.lightgbm_objective, 'clf': LGBMClassifier(random_state=random_state)},
            'CatBoostClassifier': {'obj': model_trainer.catboost_objective,'clf': CatBoostClassifier(random_state=random_state)}
        }


    def setting_attributes(trial, cv_results):
        '''
            Method Name: setting_attributes
            Description: This method sets attributes of metric results for training and validation set from a given Optuna trial
            Output: None

            Parameters:
            - trial: Optuna trial object
            - cv_results: Dictionary object related to results from cross validate function
        '''
        trial.set_user_attr("train_balanced_accuracy", 
                            np.nanmean(cv_results['train_balanced_accuracy']))
        trial.set_user_attr("val_balanced_accuracy", 
                            cv_results['test_balanced_accuracy'].mean())
        trial.set_user_attr("train_precision_score", 
                            np.nanmean(cv_results['train_precision_score']))
        trial.set_user_attr("val_precision_score", 
                            cv_results['test_precision_score'].mean())
        trial.set_user_attr("train_recall_score", 
                            np.nanmean(cv_results['train_recall_score']))
        trial.set_user_attr("val_recall_score", 
                            cv_results['test_recall_score'].mean())
        trial.set_user_attr("train_f1_score", 
                            np.nanmean(cv_results['train_f1_score']))
        trial.set_user_attr("val_f1_score", 
                            cv_results['test_f1_score'].mean())
        trial.set_user_attr("train_matthews_corrcoef", 
                            np.nanmean(cv_results['train_matthews_corrcoef']))
        trial.set_user_attr("val_matthews_corrcoef", 
                            cv_results['test_matthews_corrcoef'].mean())


    def pipeline_feature_selection_step(
            pipeline, trial, fs_method, clf, scaling_indicator='no', cluster_indicator='no', damping=None):
        '''
            Method Name: pipeline_feature_selection_step
            Description: This method adds custom transformer with FeatureSelectionTransformer class into pipeline for performing feature selection.
            Output: None
    
            Parameters:
            - pipeline: imblearn pipeline object
            - trial: Optuna trial object
            - fs_method: String name indicating method of feature selection
            - clf: Model object
            - scaling_indicator: String that represents method of performing feature scaling. (Accepted values are 'Standard', 'MinMax', 'Robust', 'Combine' and 'no'). Default value is 'no'
            - cluster_indicator: String indicator of including cluster-related feature (yes or no). Default value is 'no'
            - damping: Float value (range from 0.5 to 1 not inclusive) as an additional hyperparameter for Affinity Propagation clustering algorithm. Default value is None.
        '''
        if fs_method not in ['BorutaShap','FeatureWiz']:
            number_to_select = trial.suggest_int('number_features',1,30)
        else:
            number_to_select = None
        trial.set_user_attr("number_features", number_to_select)
        pipeline.steps.append(
            ('featureselection',FeatureSelectionTransformer(fs_method, clf, scaling_indicator = scaling_indicator, cluster_indicator = cluster_indicator, damping=damping, number = number_to_select)))


    def pipeline_setup(pipeline, trial, clf):
        '''
            Method Name: pipeline_setup
            Description: This method configures pipeline for model training, which varies depending on model class and preprocessing related parameters selected by Optuna.
            Output: None
    
            Parameters:
            - pipeline: imblearn pipeline object
            - trial: Optuna trial object
            - clf: Model object
        '''
        balancing_indicator = trial.suggest_categorical(
            'balancing',['smoteen','no'])
        if balancing_indicator == 'smoteen':
            pipeline.steps.append(
                ('smote',SMOTEN(random_state=random_state,n_jobs=2)))
        pipeline.steps.append(['feature_engine',FeatureEngineTransformer()])
        pipeline.steps.append(['interval_encoding',IntervalDataTransformer()])
        pipeline.steps.append(['binary_encoding',BinaryDataTransformer()])
        pipeline.steps.append(['ordinal_encoding',OrdinalDataTransformer()])
        contrast_encoding_method = trial.suggest_categorical(
            'contrast_method',['Helmert', 'Polynomial', 'Sum', 'Backward Difference'])
        contrast_columns = ['Fruitveg_ytd','Number_people_household','Sports_in_week','Internet_in_week','Tired_in_week','Concentrate_in_week','Softdrink_in_week','Sugarsnack_in_week','Takeawayfood_in_week']
        if contrast_encoding_method == 'Helmert':
            pipeline.steps.append(
                ['contrast_data',HelmertEncoder(cols=contrast_columns,drop_invariant=True)])
        elif contrast_encoding_method == 'Polynomial':
            pipeline.steps.append(
                ['contrast_data',PolynomialEncoder(cols=contrast_columns,drop_invariant=True)])
        elif contrast_encoding_method == 'Sum':
            pipeline.steps.append(
                ['contrast_data',SumEncoder(cols=contrast_columns,drop_invariant=True)])
        elif contrast_encoding_method == 'Backward Difference':
            pipeline.steps.append(
                ['contrast_data',BackwardDifferenceEncoder(cols=contrast_columns,drop_invariant=True)])
        pipeline.steps.append(['rare_data',RareLabelEncoder()])
        nominal_columns = ['Doingwell_schoolwork','Lots_of_choices_important','Lots_of_things_good_at','Feel_partof_community','Outdoorplay_freq','Enoughtime_toplay','Play_inall_places','Gender','Going_school','Homespace_relax','Method_of_keepintouch','Breakfast_ytd','Type_of_play_places']
        cyclic_columns = ['Sleeptime_ytd','Awaketime_today','Sleeptime_ytd_hour','Awaketime_today_hour','Timestamp_month','Timestamp_quarter','Timestamp_week','Timestamp_day_of_week','Timestamp_day_of_month','Timestamp_day_of_year','Birth_Date_month','Birth_Date_quarter','Birth_Date_week','Birth_Date_day_of_week','Birth_Date_day_of_month','Birth_Date_day_of_year']
        if type(clf).__name__ in ['LogisticRegression','LinearSVC','KNeighborsClassifier']:
            pipeline.steps.append(
                ['nominal_encoding',OneHotEncoder(variables = nominal_columns)])
            pipeline.steps.append(
                ['Cyclic_encoding',CyclicalFeatures(variables = cyclic_columns, drop_original=True)])
            scaling_indicator = 'yes'
            pipeline.steps.append(('scaling',ScalingTransformer()))
        else:
            pipeline.steps.append(
                ['nominal_encoding',CatBoostEncoder(cols = nominal_columns, drop_invariant=True)])
            pipeline.steps.append(
                ['Cyclic_encoding',CatBoostEncoder(cols = cyclic_columns, drop_invariant=True)])
            scaling_indicator = 'no'
        fs_method = trial.suggest_categorical(
            'feature_selection',['BorutaShap','Lasso','FeatureImportance_ET','MutualInformation','ANOVA','FeatureWiz'])
        cluster_indicator = trial.suggest_categorical('cluster_indicator',['yes','no']) if type(clf).__name__ in ['LogisticRegression','LinearSVC'] else 'no'
        damping = trial.suggest_float('damping',0.5,0.95,log=True) if cluster_indicator == 'yes' else None
        model_trainer.pipeline_feature_selection_step(
            pipeline, trial, fs_method, clf,scaling_indicator=scaling_indicator, cluster_indicator=cluster_indicator, damping = damping)
        trial.set_user_attr("balancing_indicator", balancing_indicator)
        trial.set_user_attr(
            "contrast_encoding_method", contrast_encoding_method)
        trial.set_user_attr("scaling_indicator", scaling_indicator)
        trial.set_user_attr("feature_selection", fs_method)
        trial.set_user_attr("Pipeline", pipeline) 
        trial.set_user_attr("cluster_indicator", cluster_indicator)
        trial.set_user_attr("damping", damping)


    def lr_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: lr_objective
            Description: This method sets the objective function for logistic regression model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        C = trial.suggest_float('C',0.0001,1,log=True)
        class_weight = trial.suggest_categorical(
            'class_weight',['balanced','None'])
        class_weight = None if class_weight == 'None' else class_weight
        penalty = trial.suggest_categorical('penalty',['l1','l2'])
        max_iter = trial.suggest_categorical('max_iter',[100000])
        solver = trial.suggest_categorical('solver',['saga'])
        dual = trial.suggest_categorical('dual',[False])
        n_jobs = trial.suggest_categorical('n_jobs',[2])
        clf = LogisticRegression(
            C=C, max_iter=max_iter, random_state=random_state, 
            class_weight = class_weight, penalty=penalty,dual=dual, solver=solver, n_jobs=n_jobs)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data,cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def svc_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: svc_objective
            Description: This method sets the objective function for linear support vector classifier model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        C = trial.suggest_float('C',0.0001,1,log=True)
        class_weight = trial.suggest_categorical(
            'class_weight',['balanced','None'])
        class_weight = None if class_weight == 'None' else class_weight
        penalty = trial.suggest_categorical('penalty',['l1','l2'])
        dual = trial.suggest_categorical('dual',[False])
        clf = LinearSVC(
            C=C, random_state=random_state, dual=dual,penalty=penalty, class_weight=class_weight)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data,cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def knn_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: knn_objective
            Description: This method sets the objective function for K-neighbors classifier model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        n_neighbors = trial.suggest_categorical('n_neighbors', [3, 5, 7, 9, 11])
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        leaf_size = trial.suggest_int('leaf_size',10,50)
        p = trial.suggest_int('p',1,4)
        n_jobs = trial.suggest_categorical('n_jobs', [2])
        clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,weights=weights,leaf_size=leaf_size,p=p,n_jobs=n_jobs)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data,cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def dt_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: dt_objective
            Description: This method sets the objective function for Decision Tree classifier model by setting various hyperparameters, including pipeline steps for different Optuna trials using post pruning.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation
    
            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        ccp_alpha = trial.suggest_float('ccp_alpha',0,0.1)
        class_weight = trial.suggest_categorical(
            'class_weight',['balanced','None'])
        class_weight = None if class_weight == 'None' else class_weight
        clf = DecisionTreeClassifier(
            random_state=random_state, class_weight=class_weight, ccp_alpha=ccp_alpha)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data,cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def rf_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: rf_objective
            Description: This method sets the objective function for Random Forest classifier model by setting various hyperparameters, including pipeline steps for different Optuna trials using post pruning.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        ccp_alpha = trial.suggest_float('ccp_alpha',0,0.1)
        class_weight = trial.suggest_categorical(
            'class_weight', ['balanced', 'balanced_subsample','None'])
        class_weight = None if class_weight == 'None' else class_weight
        n_jobs = trial.suggest_categorical('n_jobs',[2])
        n_estimators = trial.suggest_categorical('n_estimators',[100])
        clf = RandomForestClassifier(
            random_state=random_state, class_weight=class_weight, ccp_alpha=ccp_alpha, n_jobs=n_jobs, n_estimators=n_estimators)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data,cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])

    def et_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: et_objective
            Description: This method sets the objective function for Extra Trees classifier model by setting various hyperparameters, including pipeline steps for different Optuna trials using post pruning.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        ccp_alpha = trial.suggest_float('ccp_alpha',0,0.1)
        class_weight = trial.suggest_categorical('class_weight', 
            ['balanced', 'balanced_subsample','None'])
        class_weight = None if class_weight == 'None' else class_weight
        n_jobs = trial.suggest_categorical('n_jobs',[2])
        n_estimators = trial.suggest_categorical('n_estimators',[100])
        clf = ExtraTreesClassifier(
            random_state=random_state, class_weight=class_weight, ccp_alpha=ccp_alpha, n_jobs=n_jobs, n_estimators=n_estimators)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data,cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def gaussiannb_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: gaussiannb_objective
            Description: This method sets the objective function for Gaussian Naive Bayes model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        var_smoothing = trial.suggest_float(
            'var_smoothing',0.000000001,1,log=True)
        clf = GaussianNB(var_smoothing=var_smoothing)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data,cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def adaboost_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: adaboost_objective
            Description: This method sets the objective function for AdaBoost model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        learning_rate = trial.suggest_float('learning_rate',0.01,1,log=True)
        n_estimators = trial.suggest_categorical('n_estimators',[100])
        clf = AdaBoostClassifier(
            learning_rate=learning_rate, random_state=random_state, n_estimators=n_estimators)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data,cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def gradientboost_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: gradientboost_objective
            Description: This method sets the objective function for Gradient Boosting classifier model by setting various hyperparameters, including pipeline steps for different Optuna trials using post pruning.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        ccp_alpha = trial.suggest_float('ccp_alpha',0,0.1)
        loss = trial.suggest_categorical('loss',['log_loss'])
        learning_rate = trial.suggest_float('learning_rate',0.01,0.3,log=True)
        n_estimators = trial.suggest_categorical('n_estimators',[100])
        subsample = trial.suggest_float('subsample',0.5,1,log=True)
        max_features = trial.suggest_categorical('max_features',['sqrt'])      
        clf = GradientBoostingClassifier(
            random_state=random_state, loss=loss, ccp_alpha=ccp_alpha,
            n_estimators=n_estimators, subsample=subsample, learning_rate=learning_rate, max_features=max_features)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, clf)
        if 'smote' not in pipeline.named_steps.keys():
            sample_weights = compute_sample_weight(
                class_weight='balanced',y= y_train_data)
            fit_params = {'clf__sample_weight':sample_weights}
        else:
            fit_params=None
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data,cv_jobs=3,fit_params=fit_params)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def xgboost_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: xgboost_objective
            Description: This method sets the objective function for XGBoost model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        booster = trial.suggest_categorical('booster',['gbtree','dart'])
        rate_drop = trial.suggest_float('rate_drop',0.0001,1,log=True) if booster == 'dart' else None
        eta = trial.suggest_float('eta',0.1,0.5,log=True)
        gamma = trial.suggest_float('gamma',0.1,20,log=True)
        min_child_weight = trial.suggest_float(
            'min_child_weight',0.1,1000,log=True)
        max_depth = trial.suggest_int('max_depth',1,10)
        lambdas = trial.suggest_float('lambda',0.1,1000,log=True)
        alpha = trial.suggest_float('alpha',0.1,100,log=True)
        subsample = trial.suggest_float('subsample',0.5,1,log=True)
        colsample_bytree = trial.suggest_float(
            'colsample_bytree',0.5,1,log=True)
        num_round = trial.suggest_categorical('num_round',[100])
        objective = trial.suggest_categorical('objective',['binary:logistic'])
        eval_metric = trial.suggest_categorical('eval_metric',['aucpr'])
        verbosity = trial.suggest_categorical('verbosity',[0])
        tree_method = trial.suggest_categorical('tree_method',['gpu_hist'])
        single_precision_histogram = trial.suggest_categorical(
            'single_precision_histogram',[True])
        clf = XGBClassifier(
            objective=objective, eval_metric=eval_metric, verbosity=verbosity,tree_method = tree_method, booster=booster, eta=eta, gamma=gamma,single_precision_histogram=single_precision_histogram,  min_child_weight=min_child_weight, max_depth=max_depth,subsample=subsample,colsample_bytree=colsample_bytree, lambdas=lambdas, alpha=alpha, random_state=random_state, num_round=num_round, rate_drop=rate_drop)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, clf)
        if 'smote' not in pipeline.named_steps.keys():
            sample_weights = compute_sample_weight(
                class_weight='balanced',y= y_train_data)
            fit_params = {'clf__sample_weight':sample_weights}
        else:
            fit_params=None
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data,cv_jobs=3,fit_params=fit_params)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def lightgbm_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: lightgbm_objective
            Description: This method sets the objective function for LightGBM model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        is_unbalance = trial.suggest_categorical(
            'is_unbalance',['true', 'false'])
        learning_rate = trial.suggest_float('learning_rate',0.01,0.3,log=True)
        max_depth = trial.suggest_int('max_depth',3,12)
        num_leaves = trial.suggest_int('num_leaves',8,4096)
        min_child_samples = trial.suggest_int('min_child_samples',5,100)
        boosting_type = trial.suggest_categorical(
            'boosting_type',['gbdt','dart'])
        drop_rate = trial.suggest_float('drop_rate',0.0001,1,log=True) if boosting_type == 'dart' else None
        subsample = trial.suggest_float('subsample',0.5,1,log=True)
        subsample_freq = trial.suggest_int('subsample_freq',1,10)
        reg_alpha = trial.suggest_float('reg_alpha',0.1,100,log=True)
        reg_lambda = trial.suggest_float('reg_lambda',0.1,100,log=True)
        min_split_gain = trial.suggest_float('min_split_gain',0.1,15,log=True)
        max_bin = trial.suggest_categorical("max_bin", [63])
        n_estimators = trial.suggest_categorical('n_estimators',[100])
        device_type = trial.suggest_categorical('device_type',['gpu'])
        gpu_use_dp = trial.suggest_categorical('gpu_use_dp',[False])
        clf = LGBMClassifier(
            num_leaves=num_leaves, learning_rate=learning_rate, is_unbalance = is_unbalance, boosting_type=boosting_type, max_depth=max_depth, min_child_samples = min_child_samples, max_bin=max_bin, reg_alpha=reg_alpha, reg_lambda=reg_lambda, subsample = subsample, subsample_freq = subsample_freq, min_split_gain=min_split_gain, random_state=random_state, n_estimators=n_estimators, device_type=device_type,gpu_use_dp=gpu_use_dp, drop_rate=drop_rate, drop_seed = random_state)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data,cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def catboost_objective(trial,X_train_data,y_train_data):
        '''
            Method Name: catboost_objective
            Description: This method sets the objective function for CatBoost model by setting various hyperparameters, including pipeline steps for different Optuna trials.
            Output: Single floating point value that represents f1 score of given model on validation set from using 3 fold cross validation

            Parameters:
            - trial: Optuna trial object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
        '''
        max_depth = trial.suggest_int('max_depth',4,10)
        l2_leaf_reg = trial.suggest_int('l2_leaf_reg',2,10)
        random_strength = trial.suggest_float('random_strength',0.1,10,log=True)
        auto_class_weights = trial.suggest_categorical(
            'auto_class_weights',['None', 'Balanced', 'SqrtBalanced'])
        auto_class_weights = None if auto_class_weights == 'None' else auto_class_weights
        learning_rate = trial.suggest_float('learning_rate',0.01,0.3,log=True)
        boosting_type = trial.suggest_categorical('boosting_type',['Plain'])
        loss_function = trial.suggest_categorical('loss_function',['MultiClass'])
        nan_mode = trial.suggest_categorical('nan_mode',['Min'])
        task_type = trial.suggest_categorical('task_type',['GPU'])
        iterations = trial.suggest_categorical('iterations',[100])
        verbose = trial.suggest_categorical('verbose',[False])
        clf = CatBoostClassifier(
            max_depth = max_depth, l2_leaf_reg = l2_leaf_reg, learning_rate=learning_rate, random_strength=random_strength,  auto_class_weights=auto_class_weights, boosting_type = boosting_type, loss_function=loss_function,nan_mode=nan_mode,random_state=random_state,task_type=task_type, iterations=iterations, verbose=verbose)
        pipeline = Pipeline(steps=[], memory='Caching')
        model_trainer.pipeline_setup(pipeline, trial, clf)
        cv_results = model_trainer.classification_metrics(
            clf,pipeline,X_train_data,y_train_data,cv_jobs=3)
        model_trainer.setting_attributes(trial,cv_results)
        return np.nanmean(cv_results['test_f1_score'])


    def classification_metrics(
            clf,pipeline,X_train_data,y_train_data, cv_jobs, fit_params=None):
        '''
            Method Name: classification_metrics
            Description: This method performs 3-fold cross validation on the training set and performs model evaluation on the validation set.
            Output: Dictionary of metric scores from 3-fold cross validation.

            Parameters:
            - clf: Model object
            - pipeline: imblearn pipeline object
            - X_train_data: Features from dataset
            - y_train_data: Target column from dataset
            - cv_jobs: Number of cross validation jobs to run in parallel
            - fit_params: Additional parameters passed to fit method of cross_validate function in the form of dictionary
        '''
        pipeline_copy = clone(pipeline)
        pipeline_copy.steps.append(('clf',clf))
        y_train_data = y_train_data.map({'normal':0,'emotional_significant':1,'behaviour_significant':2,'emotional_and_behaviour_significant':3})
        cv_results = cross_validate(
            pipeline_copy, X_train_data, y_train_data, cv=3,
            scoring={"balanced_accuracy": make_scorer(balanced_accuracy_score),
            "precision_score": make_scorer(precision_score, average='macro'),
            "recall_score": make_scorer(recall_score, average='macro'),
            "f1_score": make_scorer(f1_score, average='macro'),
            "matthews_corrcoef": make_scorer(matthews_corrcoef)},
            n_jobs=cv_jobs,return_train_score=True,error_score='raise',fit_params=fit_params)
        return cv_results


    def optuna_optimizer(self, obj, n_trials, fold):
        '''
            Method Name: optuna_optimizer
            Description: This method creates a new Optuna study object if the given Optuna study object doesn't exist or otherwise using existing Optuna study object and optimizes the given objective function. In addition, the following plots and results are also created and saved:
            1. Hyperparameter Importance Plot
            2. Optimization History Plot
            3. Optuna study object
            4. Optimization Results (csv format)
            
            Output: Single best trial object
            On Failure: Logging error and raise exception

            Parameters:
            - obj: Optuna objective function
            - n_trials: Number of trials for Optuna hyperparameter tuning
            - fold: Fold number from nested cross-validation in outer loop
        '''
        try:
            if f"OptStudy_{obj.__name__}_Fold_{fold}.pkl" in os.listdir(self.folderpath+obj.__name__):
                study = joblib.load(
                    self.folderpath+obj.__name__+f"/OptStudy_{obj.__name__}_Fold_{fold}.pkl")
            else:
                sampler = optuna.samplers.TPESampler(
                    multivariate=True, seed=random_state)
                study = optuna.create_study(
                    direction='maximize',sampler=sampler)
            study.optimize(
                obj, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)
            trial = study.best_trial
            if trial.number !=0:
                param_imp_fig = optuna.visualization.plot_param_importances(study)
                opt_fig = optuna.visualization.plot_optimization_history(study)
                param_imp_fig.write_image(
                    self.folderpath+ obj.__name__ +f'/HP_Importances_{obj.__name__}_Fold_{fold}.png')
                opt_fig.write_image(
                    self.folderpath+ obj.__name__ +f'/Optimization_History_{obj.__name__}_Fold_{fold}.png')
            joblib.dump(
                study, self.folderpath + obj.__name__ + f'/OptStudy_{obj.__name__}_Fold_{fold}.pkl')
            study.trials_dataframe().to_csv(
                self.folderpath + obj.__name__ + f"/Hyperparameter_Tuning_Results_{obj.__name__}_Fold_{fold}.csv",index=False)
            del study
        except Exception as e:
            self.log_writer.log(
                self.file_object, f'Performing optuna hyperparameter tuning for {obj.__name__} model failed with the following error: {e}')
            raise Exception(
                f'Performing optuna hyperparameter tuning for {obj.__name__} model failed with the following error: {e}')
        return trial

    
    def confusion_matrix_plot(
            self, clf, figtitle, plotname, actual_labels, pred_labels):
        '''
            Method Name: confusion_matrix_plot
            Description: This method plots confusion matrix and saves plot within the given model class folder.
            Output: None

            Parameters:
            - clf: Model object
            - figtitle: String that represents part of title figure
            - plotname: String that represents part of image name
            - actual_labels: Actual target labels from dataset
            - pred_labels: Predicted target labels from model
        '''
        cmd = ConfusionMatrixDisplay.from_predictions(
            actual_labels, pred_labels)
        cmd.ax_.set_title(f"{type(clf).__name__} {figtitle}")
        cmd.ax_.set_xticklabels(cmd.ax_.get_xticklabels(), rotation = 45)
        plt.grid(False)
        cmd.figure_.savefig(
            self.folderpath+type(clf).__name__+f'/Confusion_Matrix_{type(clf).__name__}_{plotname}.png', bbox_inches='tight', pad_inches=0.2)
        plt.clf()


    def classification_report_plot(
            self, clf, figtitle, plotname, actual_labels, pred_labels):
        '''
            Method Name: classification_report_plot
            Description: This method plots classification report in heatmap form and saves plot within the given model class folder.
            Output: None

            Parameters:
            - clf: Model object
            - figtitle: String that represents part of title figure
            - plotname: String that represents part of image name
            - actual_labels: Actual target labels from dataset
            - pred_labels: Predicted target labels from model
        '''
        clf_report = classification_report(
            actual_labels,pred_labels,output_dict=True,digits=4)
        fig = plt.figure()
        sns.heatmap(
            pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, fmt=".4f")
        plt.title(f"{type(clf).__name__} {figtitle}")
        fig.savefig(
            self.folderpath+type(clf).__name__+f'/Classification_Report_{type(clf).__name__}_{plotname}.png', bbox_inches='tight', pad_inches=0.2)
        plt.clf()


    def precision_recall_plot(
            self, clf, figtitle, plotname, actual_labels, pred_proba):
        '''
            Method Name: precision_recall_plot
            Description: This method plots precision recall curve and saves plot within the given model class folder.
            Output: None

            Parameters:
            - clf: Model object
            - figtitle: String that represents part of title figure
            - plotname: String that represents part of image name
            - actual_labels: Actual target labels from dataset
            - pred_proba: Predicted probability of target being positive (1) from model
        '''
        classes=['normal', 'emotional_significant','emotional_and_behaviour_significant', 'behaviour_significant']
        # Use label_binarize to be multi-label like settings
        Y = label_binarize(actual_labels, classes=classes)
        # For each class
        precision, recall, average_precision = dict(), dict(), dict()
        for i in range(len(classes)):
            precision[i], recall[i], _ = precision_recall_curve(Y[:,i], np.array(pred_proba)[:,i])
            average_precision[i] = average_precision_score(Y[:,i], np.array(pred_proba)[:,i])
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y.ravel(), np.array(pred_proba).ravel())
        average_precision["micro"] = average_precision_score(Y, pred_proba, average="micro")
        _, ax = plt.subplots(figsize=(8, 7))
        for i, color in zip(range(len(classes)), ['red','blue','green','pink']):
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=average_precision[i],
            )
            display.plot(ax=ax, name=f"Precision-recall for {classes[i]} class", color=color)
        PrecisionRecallDisplay(
            recall=recall["micro"],precision=precision["micro"],average_precision=average_precision["micro"]
        ).plot(ax=ax, name=f"Precision-recall micro average", color='orange')
        plt.legend(loc='best',fontsize=8)
        plt.title(f"{type(clf).__name__} {figtitle}")
        plt.savefig(
            self.folderpath+type(clf).__name__+f'/PrecisionRecall_Curve_{type(clf).__name__}_{plotname}.png')
        plt.clf()


    def learning_curve_plot(self, clf, input_data, output_data):
        '''
            Method Name: learning_curve_plot
            Description: This method plots learning curve of 5 fold cross validation and saves plot within the given model class folder.
            Output: None

            Parameters:
            - clf: Model object
            - input_data: Features from dataset
            - output_data: Target column from dataset
        '''
        if type(clf).__name__ == 'CatBoostClassifier':
            train_sizes, train_scores, validation_scores = learning_curve(estimator = clf, X = input_data, y = output_data, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state), scoring='f1_macro', train_sizes=np.linspace(0.3, 1.0, 10))
            plt.style.use('seaborn-whitegrid')
            plt.grid(True)
            plt.fill_between(train_sizes, train_scores.mean(axis = 1) - train_scores.std(axis = 1), train_scores.mean(axis = 1) + train_scores.std(axis = 1), alpha=0.25, color='blue')
            plt.plot(train_sizes, train_scores.mean(axis = 1), label = 'Training Score', marker='.',markersize=14)
            plt.fill_between(train_sizes, validation_scores.mean(axis = 1) - validation_scores.std(axis = 1), validation_scores.mean(axis = 1) + validation_scores.std(axis = 1), alpha=0.25, color='green')
            plt.plot(train_sizes, validation_scores.mean(axis = 1), label = 'Cross Validation Score', marker='.',markersize=14)
            plt.ylabel('Score')
            plt.xlabel('Training instances')
            plt.title(f'Learning Curve for {type(clf).__name__}')
            plt.legend(frameon=True, loc='best')
            plt.savefig(
                self.folderpath+type(clf).__name__+f'/LearningCurve_{type(clf).__name__}.png',bbox_inches='tight')
            plt.clf()
        else:
            visualizer = LearningCurve(
                clf, cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state), scoring='f1_macro', train_sizes=np.linspace(0.3, 1.0, 10))
            visualizer.fit(input_data,output_data)
            visualizer.show(
                outpath=self.folderpath+type(clf).__name__+f'/LearningCurve_{type(clf).__name__}.png',clear_figure=True)


    def shap_plot(self, clf, input_data):
        '''
            Method Name: shap_plot
            Description: This method plots feature importance and its summary using shap values and saves plot within the given model class folder. Note that this function will not work specifically for XGBoost models that use 'dart' booster. In addition, shap plots for KNeighbors and GaussianNB require use of shap's Kernel explainer that involves high computational time. Thus, this function excludes both KNeighbors and GaussianNB.
            Output: None

            Parameters:
            - clf: Model object
            - input_data: Features from dataset
        '''
        if (type(clf).__name__ not in ['KNeighborsClassifier','GaussianNB']):
            if type(clf).__name__ in ['LogisticRegression','LinearSVC']:
                explainer = shap.LinearExplainer(clf, input_data)
                explainer_obj = explainer(input_data)
                shap_values = explainer.shap_values(input_data)
            else:
                if ('dart' in clf.get_params().values()) and (type(clf).__name__ == 'XGBClassifier'):
                    return
                explainer = shap.TreeExplainer(clf)
                explainer_obj = explainer(input_data)
                shap_values = explainer.shap_values(input_data)
            classes=['normal', 'emotional_significant','emotional_and_behaviour_significant', 'behaviour_significant']
            for index in range(len(classes)):
                plt.figure()
                shap.summary_plot(
                    shap_values[index], input_data, plot_type="bar", show=False, max_display=40)
                plt.title(
                    f'Shap Feature Importances for {type(clf).__name__} from {classes[index]} class')
                plt.savefig(
                    self.folderpath+type(clf).__name__+f'/Shap_Feature_Importances_{type(clf).__name__}_{classes[index]}.png',bbox_inches='tight')
                plt.clf()
                plt.figure()
                shap.plots.beeswarm(
                    explainer_obj[:,:,index], show=False, max_display=40)
                plt.title(
                    f'Shap Summary Plot for {type(clf).__name__} from {classes[index]} class')
                plt.savefig(
                    self.folderpath+type(clf).__name__+f'/Shap_Summary_Plot_{type(clf).__name__}_{classes[index]}.png',bbox_inches='tight')
                plt.clf()


    def model_training(
            self, clf, obj, input_data, output_data, n_trials, fold_num):
        '''
            Method Name: model_training
            Description: This method performs Optuna hyperparameter tuning using 3 fold cross validation on given dataset. The best hyperparameters with the best pipeline identified is used for model training.
            
            Output: 
            - model_copy: Trained model object
            - best_trial: Optuna's best trial object from hyperparameter tuning
            - input_data_transformed: Transformed features from dataset
            - output_data_transformed: Transformed target column from dataset
            - best_pipeline: imblearn pipeline object

            On Failure: Logging error and raise exception

            Parameters:
            - clf: Model object
            - obj: Optuna objective function
            - input_data: Features from dataset
            - output_data: Target column from dataset
            - n_trials: Number of trials for Optuna hyperparameter tuning
            - fold_num: Indication of fold number for model training (can be integer or string "overall")
        '''
        func = lambda trial: obj(trial, input_data, output_data)
        func.__name__ = type(clf).__name__
        self.log_writer.log(
            self.file_object, f"Start hyperparameter tuning for {type(clf).__name__} for fold {fold_num}")
        best_trial = self.optuna_optimizer(func, n_trials, fold_num)
        self.log_writer.log(
            self.file_object, f"Hyperparameter tuning for {type(clf).__name__} completed for fold {fold_num}")
        self.log_writer.log(
            self.file_object, f"Start using best pipeline for {type(clf).__name__} for transforming training and validation data for fold {fold_num}")
        best_pipeline = best_trial.user_attrs['Pipeline']
        output_data = output_data.map({'normal':0,'emotional_significant':1,'behaviour_significant':2,'emotional_and_behaviour_significant':3})
        input_data_transformed = best_pipeline.fit_transform(
            input_data, output_data)
        if 'smote' in best_pipeline.named_steps.keys():
            output_data_transformed = best_pipeline.steps[0][1].fit_resample(input_data, output_data)[1]
        else:
            output_data_transformed = output_data
        self.log_writer.log(
            self.file_object, f"Finish using best pipeline for {type(clf).__name__} for transforming training and validation data for fold {fold_num}")
        for parameter in ['missing','balancing','contrast_method','scaling','feature_selection','number_features','feature_selection_missing','damping','cluster_indicator']:
            if parameter in best_trial.params.keys():
                best_trial.params.pop(parameter)
        for weight_param in ['class_weight','auto_class_weights']:
            if weight_param in best_trial.params.keys():
                if best_trial.params[weight_param] == 'None':
                    best_trial.params.pop(weight_param)
        self.log_writer.log(
            self.file_object, f"Start evaluating model performance for {type(clf).__name__} on validation set for fold {fold_num}")
        model_copy = clone(clf)
        model_copy = model_copy.set_params(**best_trial.params)
        model_copy.fit(input_data_transformed, output_data_transformed)
        return model_copy, best_trial, input_data_transformed, output_data_transformed, best_pipeline


    def hyperparameter_tuning(
            self, obj, clf, n_trials, input_data, output_data):
        '''
            Method Name: hyperparameter_tuning
            Description: This method performs Stratified Nested 3 Fold Cross Validation on the entire dataset, where the inner loop (3-fold) performs Optuna hyperparameter tuning and the outer loop (5-fold) performs model evaluation to obtain overall generalization error of model. The best hyperparameters with the best pipeline identified from inner loop is used for model training on the entire training set and model evaluation on the test set for the outer loop.
            In addition, the following intermediate results are saved for a given model class:
            1. Model_Performance_Results_by_Fold (csv file)
            2. Overall_Model_Performance_Results (csv file)
            3. Confusion Matrix image
            4. Classification Report heatmap image
            5. Precision-Recall curve image
            
            Output: None
            On Failure: Logging error and raise exception

            Parameters:
            - obj: Optuna objective function
            - clf: Model object
            - n_trials: Number of trials for Optuna hyperparameter tuning
            - input_data: Features from dataset
            - output_data: Target column from dataset
        '''
        try:
            num_folds = 5
            skfold = StratifiedKFold(
                n_splits=num_folds, shuffle=True, random_state=random_state)
            bal_accuracy_train_cv, precision_train_cv, recall_train_cv, f1_train_cv, mc_train_cv = [], [], [], [], []
            bal_accuracy_val_cv, precision_val_cv, recall_val_cv, mc_val_cv, f1_val_cv = [], [], [], [], []
            bal_accuracy_test_cv, precision_test_cv, recall_test_cv, mc_test_cv, f1_test_cv = [], [], [], [], []
            actual_labels, pred_labels, pred_proba = [], [], []
            for fold, (outer_train_idx, outer_valid_idx) in enumerate(skfold.split(input_data, output_data)):
                input_sub_train_data = input_data.iloc[outer_train_idx,:].reset_index(drop=True)
                output_sub_train_data = output_data.iloc[outer_train_idx].reset_index(drop=True)
                model_copy, best_trial, input_train_data_transformed, output_train_data_transformed, best_pipeline = self.model_training(clf, obj, input_sub_train_data, output_sub_train_data, n_trials, fold+1)
                input_val_data = input_data.iloc[outer_valid_idx,:].reset_index(drop=True)
                input_val_data_transformed = best_pipeline.transform(input_val_data)
                val_pred = model_copy.predict(input_val_data_transformed)
                dict_class = {0:'normal',1:'emotional_significant',2:'behaviour_significant',3:'emotional_and_behaviour_significant'}
                val_pred = [dict_class[k] for k in val_pred.ravel()]
                actual_labels.extend(output_data.iloc[outer_valid_idx].tolist())
                pred_labels.extend(val_pred)
                y_score = model_copy.predict_proba(input_val_data_transformed) if type(clf).__name__ != 'LinearSVC' else model_copy._predict_proba_lr(input_val_data_transformed)
                pred_proba.extend(y_score)
                bal_acc_outer_val_value = balanced_accuracy_score(
                    np.array(output_data.iloc[outer_valid_idx]),val_pred)
                precision_outer_val_value = precision_score(
                    np.array(output_data.iloc[outer_valid_idx]),val_pred, average='macro')
                f1_outer_val_value = f1_score(
                    np.array(output_data.iloc[outer_valid_idx]),val_pred, average='macro')
                mc_outer_val_value = matthews_corrcoef(
                    np.array(output_data.iloc[outer_valid_idx]),val_pred)
                recall_outer_val_value = recall_score(
                    np.array(output_data.iloc[outer_valid_idx]),val_pred, average='macro')
                cv_lists = [bal_accuracy_train_cv, precision_train_cv, f1_train_cv, mc_train_cv, recall_train_cv, bal_accuracy_val_cv, precision_val_cv, f1_val_cv, mc_val_cv, recall_val_cv, bal_accuracy_test_cv, precision_test_cv, f1_test_cv, mc_test_cv, recall_test_cv]
                metric_values = [best_trial.user_attrs['train_balanced_accuracy'], best_trial.user_attrs['train_precision_score'], best_trial.user_attrs['train_f1_score'], best_trial.user_attrs['train_matthews_corrcoef'], best_trial.user_attrs['train_recall_score'], best_trial.user_attrs['val_balanced_accuracy'], best_trial.user_attrs['val_precision_score'], best_trial.user_attrs['val_f1_score'], best_trial.user_attrs['val_matthews_corrcoef'], best_trial.user_attrs['val_recall_score'], bal_acc_outer_val_value, precision_outer_val_value, f1_outer_val_value, mc_outer_val_value, recall_outer_val_value]
                for cv_list, metric in zip(cv_lists, metric_values):
                    cv_list.append(metric)
                self.log_writer.log(
                    self.file_object, f"Evaluating model performance for {type(clf).__name__} on validation set completed for fold {fold+1}")
                optimized_results = pd.DataFrame({
                    'Feature_selector':best_trial.user_attrs['feature_selection'], 'Contrast_encoding_method': best_trial.user_attrs['contrast_encoding_method'], 'Models': type(model_copy).__name__, 'Best_params': str(model_copy.get_params()), 'Cluster_Indicator': best_trial.user_attrs['cluster_indicator'], 'Damping_cluster_value': best_trial.user_attrs['damping'], 'Number_features': [len(input_train_data_transformed.columns.tolist())], 'Features': [input_train_data_transformed.columns.tolist()], 'Balancing_handled': best_trial.user_attrs['balancing_indicator'], 'Feature_scaling_handled': best_trial.user_attrs['scaling_indicator'], 'Outer_fold': fold+1,'bal_acc_inner_train_cv': best_trial.user_attrs['train_balanced_accuracy'],'bal_acc_inner_val_cv': best_trial.user_attrs['val_balanced_accuracy'],'bal_acc_outer_val_cv': [bal_acc_outer_val_value],'precision_inner_train_cv': best_trial.user_attrs['train_precision_score'],'precision_inner_val_cv': best_trial.user_attrs['val_precision_score'],'precision_outer_val_cv': [precision_outer_val_value],'recall_inner_train_cv': best_trial.user_attrs['train_recall_score'],'recall_inner_val_cv': best_trial.user_attrs['val_recall_score'],'recall_outer_val_cv': [recall_outer_val_value],'f1_inner_train_cv': best_trial.user_attrs['train_f1_score'],'f1_inner_val_cv': best_trial.user_attrs['val_f1_score'],'f1_outer_val_cv': [f1_outer_val_value],'mc_inner_train_cv': best_trial.user_attrs['train_matthews_corrcoef'],'mc_inner_val_cv': best_trial.user_attrs['val_matthews_corrcoef'],'mc_outer_val_cv': [mc_outer_val_value]})
                optimized_results.to_csv(
                    self.folderpath+'Model_Performance_Results_by_Fold.csv', mode='a', index=False, header=not os.path.exists(self.folderpath+'Model_Performance_Results_by_Fold.csv'))
                self.log_writer.log(
                    self.file_object, f"Optimized results for {type(clf).__name__} model saved for fold {fold+1}")
                time.sleep(10)
            average_results = pd.DataFrame({
                'Models': type(model_copy).__name__, 'bal_acc_train_cv_avg': np.mean(bal_accuracy_train_cv), 'bal_acc_train_cv_std': np.std(bal_accuracy_train_cv), 'bal_acc_val_cv_avg': np.mean(bal_accuracy_val_cv), 'bal_acc_val_cv_std': np.std(bal_accuracy_val_cv), 'bal_acc_test_cv_avg': np.mean(bal_accuracy_test_cv), 'bal_acc_test_cv_std': np.std(bal_accuracy_test_cv), 'precision_train_cv_avg': np.mean(precision_train_cv), 'precision_train_cv_std': np.std(precision_train_cv), 'precision_val_cv_avg': np.mean(precision_val_cv), 'precision_val_cv_std': np.std(precision_val_cv), 'precision_test_cv_avg': np.mean(precision_test_cv), 'precision_test_cv_std': np.std(precision_test_cv), 'recall_train_cv_avg': np.mean(recall_train_cv), 'recall_train_cv_std': np.std(recall_train_cv), 'recall_val_cv_avg': np.mean(recall_val_cv),'recall_val_cv_std': np.std(recall_val_cv),'recall_test_cv_avg': np.mean(recall_test_cv),'recall_test_cv_std': np.std(recall_test_cv), 'f1_train_cv_avg': np.mean(f1_train_cv), 'f1_train_cv_std': np.std(f1_train_cv), 'f1_val_cv_avg': np.mean(f1_val_cv),'f1_val_cv_std': np.std(f1_val_cv), 'f1_test_cv_avg': np.mean(f1_test_cv), 'f1_test_cv_std': np.std(f1_test_cv),'mc_train_cv_avg': np.mean(mc_train_cv), 'mc_train_cv_std': np.std(mc_train_cv), 'mc_val_cv_avg': np.mean(mc_val_cv),'mc_val_cv_std': np.std(mc_val_cv), 'mc_test_cv_avg': np.mean(mc_test_cv), 'mc_test_cv_std': np.std(mc_test_cv)}, index=[0])
            average_results.to_csv(
                self.folderpath+'Overall_Model_Performance_Results.csv', mode='a', index=False, header=not os.path.exists(self.folderpath+'Overall_Model_Performance_Results.csv'))
            self.log_writer.log(
                self.file_object, f"Average optimized results for {type(clf).__name__} model saved")                
            self.confusion_matrix_plot(
                clf, 'Confusion Matrix', 'CV', actual_labels, pred_labels)
            self.classification_report_plot(
                clf, 'Classification Report', 'CV', actual_labels, pred_labels)
            self.precision_recall_plot(
                clf, 'Precision Recall Curve', 'CV', actual_labels, pred_proba)
        except Exception as e:
            self.log_writer.log(
                self.file_object, f'Hyperparameter tuning on {type(clf).__name__} model failed with the following error: {e}')
            raise Exception(
                f'Hyperparameter tuning on {type(clf).__name__} model failed with the following error: {e}')


    def final_overall_model(self, obj, clf, input_data, output_data, n_trials):
        '''
            Method Name: final_overall_model
            Description: This method performs hyperparameter tuning on best model algorithm identified using stratified 3 fold cross validation on entire dataset. The best hyperparameters identified are then used to train the entire dataset before saving model for deployment.
            In addition, the following intermediate results are saved for a given model class:
            1. Confusion Matrix image
            2. Classification Report heatmap image
            3. Precision Recall Curve image
            4. Learning Curve image
            5. Shap Feature Importances for every class (barplot image)
            6. Shap Summary Plot for every class (beeswarm plot image)
            
            Output: None

            Parameters:
            - obj: Optuna objective function
            - clf: Model object
            - input_data: Features from dataset
            - output_data: Target column from dataset
            - n_trials: Number of trials for Optuna hyperparameter tuning
        '''
        self.log_writer.log(
            self.file_object, f"Start final model training on all data for {type(clf).__name__}")
        overall_model, best_trial, input_data_transformed, output_data_transformed, best_pipeline = self.model_training(
            clf, obj, input_data, output_data, n_trials, 'overall')
        joblib.dump(best_pipeline,'Saved_Models/Preprocessing_Pipeline.pkl')
        joblib.dump(overall_model,'Saved_Models/FinalModel.pkl')
        dict_class = {0:'normal',1:'emotional_significant',2:'behaviour_significant',3:'emotional_and_behaviour_significant'}
        actual_labels = [dict_class[k] for k in output_data_transformed]
        pred_labels = overall_model.predict(input_data_transformed)
        pred_labels = [dict_class[k] for k in pred_labels]
        pred_proba = overall_model.predict_proba(input_data_transformed) if type(overall_model).__name__ != 'LinearSVC' else overall_model._predict_proba_lr(input_data_transformed)
        self.confusion_matrix_plot(
            clf, 'Confusion Matrix - Final Model', 'Final_Model', actual_labels, pred_labels)
        self.classification_report_plot(
            clf, 'Classification Report - Final Model', 'Final_Model', actual_labels, pred_labels)
        self.precision_recall_plot(
            clf, 'Precision Recall Curve - Final Model', 'Final_Model', actual_labels, pred_proba)
        self.learning_curve_plot(
            overall_model, input_data_transformed, output_data_transformed)
        self.shap_plot(overall_model, input_data_transformed)
        self.log_writer.log(
            self.file_object, f"Finish final model training on all data for {type(clf).__name__}")
        

    def model_selection(self, input, output, num_trials, folderpath):
        '''
            Method Name: model_selection
            Description: This method performs model algorithm selection using Stratified Nested Cross Validation (5-fold cv outer loop for model evaluation and 3-fold cv inner loop for hyperparameter tuning)
            Output: None

            Parameters:
            - input: Features from dataset
            - output: Target column from dataset
            - num_trials: Number of Optuna trials for hyperparameter tuning
            - folderpath: String path name where all results generated from model training are stored.
        '''
        self.log_writer.log(
            self.file_object, 'Start process of model selection')
        self.input = input
        self.output = output
        self.num_trials = num_trials
        self.folderpath = folderpath
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        input_data = self.input.astype('object').copy()
        output_data = self.output['Wellbeing_Category_WMS'].copy()
        for selector in self.optuna_selectors.values():
            obj = selector['obj']
            clf = selector['clf']
            path = os.path.join(self.folderpath, type(clf).__name__)
            if not os.path.exists(path):
                os.mkdir(path)
            self.hyperparameter_tuning(
                obj = obj, clf = clf, n_trials = self.num_trials, input_data = input_data, output_data = output_data)
            time.sleep(10)
        overall_results = pd.read_csv(
            self.folderpath + 'Overall_Model_Performance_Results.csv')
        self.log_writer.log(
            self.file_object, f"Best model identified based on balanced accuracy score is {overall_results.iloc[overall_results['bal_acc_test_cv_avg'].idxmax()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['bal_acc_test_cv_avg'].idxmax()]['bal_acc_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['bal_acc_test_cv_avg'].idxmax()]['bal_acc_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, f"Best model identified based on precision score is {overall_results.iloc[overall_results['precision_test_cv_avg'].idxmax()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['precision_test_cv_avg'].idxmax()]['precision_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['precision_test_cv_avg'].idxmax()]['precision_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, f"Best model identified based on recall score is {overall_results.iloc[overall_results['recall_test_cv_avg'].idxmax()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['recall_test_cv_avg'].idxmax()]['recall_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['recall_test_cv_avg'].idxmax()]['recall_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, f"Best model identified based on f1 score is {overall_results.iloc[overall_results['f1_test_cv_avg'].idxmax()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['f1_test_cv_avg'].idxmax()]['f1_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['f1_test_cv_avg'].idxmax()]['f1_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, f"Best model identified based on matthews correlation coefficient is {overall_results.iloc[overall_results['mc_test_cv_avg'].idxmax()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['mc_test_cv_avg'].idxmax()]['mc_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['mc_test_cv_avg'].idxmax()]['mc_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, 'Finish process of model selection')


    def final_model_tuning(
            self, input_data, output_data, num_trials, folderpath):
        '''
            Method Name: final_model_tuning
            Description: This method performs final model training from best model algorithm identified on entire dataset using Stratified 3-fold cross validation.
            Output: None

            Parameters:
            - input_data: Features from dataset
            - output_data: Target column from dataset
            - num_trials: Number of Optuna trials for hyperparameter tuning
            - folderpath: String path name where all results generated from model training are stored.
        '''
        self.input_data = input_data
        self.output_data = output_data
        self.num_trials = num_trials
        self.folderpath = folderpath
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        try:
            model_number = int(input("""
    Select one of the following models to use for model deployment: 
    [1] Logistic Regression
    [2] Linear SVC
    [3] K Neighbors Classifier
    [4] Gaussian Naive Bayes
    [5] Decision Tree Classifier
    [6] Random Forest Classifier
    [7] Extra Trees Classifier
    [8] Ada Boost Classifier
    [9] Gradient Boost Classifier
    [10] XGBoost Classifier
    [11] LGBM Classifier
    [12] CatBoost Classifier
            """))
            model_options = {1: 'LogisticRegression', 2: 'LinearSVC', 3: 'KNeighborsClassifier', 4: 'GaussianNB', 5: 'DecisionTreeClassifier', 6: 'RandomForestClassifier', 7: 'ExtraTreesClassifier', 8: 'AdaBoostClassifier', 9: 'GradientBoostingClassifier', 10: 'XGBClassifier', 11: 'LGBMClassifier', 12: 'CatBoostClassifier'}
            best_model_name = model_options[model_number]
        except:
            print(
                'Please insert a valid number of choice for model deployment.')
            return
        self.log_writer.log(
            self.file_object, f"Start performing hyperparameter tuning on best model identified overall: {best_model_name}")
        obj = self.optuna_selectors[best_model_name]['obj']
        clf = self.optuna_selectors[best_model_name]['clf']
        input_data = self.input_data.copy()
        output_data = self.output_data['Wellbeing_Category_WMS'].copy()
        self.final_overall_model(
            obj = obj, clf = clf, input_data = input_data, output_data = output_data, n_trials = self.num_trials)
        self.log_writer.log(
            self.file_object, f"Finish performing hyperparameter tuning on best model identified overall: {best_model_name}")


class FeatureEngineTransformer(BaseEstimator, TransformerMixin):
    
    
    def __init__(self):
        '''
            Method Name: __init__
            Description: This method initializes instance of FeatureEngineTransformer class
            Output: None
        '''
        pass

    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method simply passes the fit method of transformer without execution.
            Output: self
        '''
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs customized feature engineering on features of this specific dataset.
            Output: Additional features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        X_['Timestamp'] = X_['Timestamp'].apply(
            lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d'))
        X_['Birth_Date'] = X_['Birth_Date'].apply(
            lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d'))
        X_['age'] = (X_['Timestamp']-X_['Birth_Date']).astype('timedelta64[Y]').astype('int')
        X_['Sleeptime_ytd'] = X_['Sleeptime_ytd'].apply(
            lambda x: datetime.datetime.strptime(x[11:], '%H:%M:%S') + datetime.timedelta(days=1) if datetime.datetime.strptime(x[11:], '%H:%M:%S').hour < 13 else datetime.datetime.strptime(x[11:], '%H:%M:%S'))
        X_['Awaketime_today'] = X_['Awaketime_today'].apply(
            lambda x: datetime.datetime.strptime(x[11:], '%H:%M:%S') + datetime.timedelta(days=1))
        X_['Hours_slept'] = (X_['Awaketime_today']-X_['Sleeptime_ytd']).astype('timedelta64[m]')/60
        feat_extract_timestamp = DatetimeFeatures(
            variables=['Timestamp'],missing_values='ignore',features_to_extract=['month','quarter','week','day_of_week','day_of_month','day_of_year'])
        X_ = feat_extract_timestamp.fit_transform(X_)
        feat_extract_birthdate = DatetimeFeatures(
            variables=['Birth_Date'],missing_values='ignore',features_to_extract=['year','month','quarter','week','day_of_week','day_of_month','day_of_year'])
        X_ = feat_extract_birthdate.fit_transform(X_)
        feat_extract_time = DatetimeFeatures(
            drop_original=False,variables=['Sleeptime_ytd','Awaketime_today'],missing_values='ignore',features_to_extract=['hour','minute'])
        X_ = feat_extract_time.fit_transform(X_)
        X_['Brush_teeth_ytd'] = X_['Brush_teeth_ytd'].astype('int')
        X_['Num_Method_of_keepintouch'] = X_['Method_of_keepintouch'].apply(lambda x: len(x.split(';')))
        X_['Num_Type_of_play_places'] = X_['Type_of_play_places'].apply(
            lambda x: len(x.split(';')))
        X_['Num_Breakfast_ytd'] = X_['Breakfast_ytd'].apply(
            lambda x: len(x.split(';')))
        dict_time = {}
        for value in enumerate(sorted(X_['Sleeptime_ytd'].unique())):
            dict_time[value[1]] = value[0]
        X_['Sleeptime_ytd'] = X_['Sleeptime_ytd'].map(dict_time)
        dict_time = {}
        for value in enumerate(sorted(X_['Awaketime_today'].unique())):
            dict_time[value[1]] = value[0]
        X_['Awaketime_today'] = X_['Awaketime_today'].map(dict_time)
        return X_


class IntervalDataTransformer(BaseEstimator, TransformerMixin):
    
    
    def __init__(self):
        '''
            Method Name: __init__
            Description: This method initializes instance of IntervalDataTransformer class
            Output: None
        '''
        pass

    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method simply passes the fit method of transformer without execution.
            Output: self
        '''
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs ordinal encoding on interval features.
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        X_['Study_Year'] = X_['Study_Year'].map(
            {'Year 3': 0, 'Year 4': 1, 'Year 5': 2, 'Year 6': 3})
        X_['Safety_toplay_scale'] = X_['Safety_toplay_scale'].apply(
            lambda x: int(float(x)))
        col_scale = ['Health_scale','School_scale','Family_scale','Friends_scale','Looks_scale','Life_scale']
        X_[col_scale] = X_[col_scale].astype('int')
        X_['WIMD_2019_Decile'] = X_['WIMD_2019_Rank'].apply(
            lambda x: IntervalDataTransformer.decile_ranking(int(x)))
        X_['WIMD_2019_Quintile'] = X_['WIMD_2019_Rank'].apply(
            lambda x: IntervalDataTransformer.quintile_ranking(int(x)))
        X_['WIMD_2019_Quartile'] = X_['WIMD_2019_Rank'].apply(
            lambda x: IntervalDataTransformer.quartile_ranking(int(x)))
        X_['WIMD_2019_Rank'] = X_['WIMD_2019_Rank'].astype('int')
        X_['Birth_Date_year'] = X_['Birth_Date_year'] - 2007
        return X_
    

    def decile_ranking(most_ranked):
        '''
            Method Name: decile_ranking
            Description: This method assigns values to WIMD rank feature on decile scale (1 to 10)
            Output: Integer value between 1 and 10
            
            Parameters:
            - most_ranked: Integer value that represents the highest frequency of WIMD rank
        '''
        if most_ranked in range(1,192):
            return 1
        elif most_ranked in range(192,383):
            return 2
        elif most_ranked in range(383,574):
            return 3
        elif most_ranked in range(574,765):
            return 4
        elif most_ranked in range(765,956):
            return 5
        elif most_ranked in range(956,1147):
            return 6
        elif most_ranked in range(1147,1338):
            return 7
        elif most_ranked in range(1338,1529):
            return 8
        elif most_ranked in range(1529,1720):
            return 9
        elif most_ranked in range(1720,1910):
            return 10


    def quintile_ranking(most_ranked):
        '''
            Method Name: quintile_ranking
            Description: This method assigns values to WIMD rank feature on quintile scale (1 to 5)
            Output: Integer value between 1 and 5
            
            Parameters:
            - most_ranked: Integer value that represents the highest frequency of WIMD rank
        '''
        if most_ranked in range(1,383):
            return 1
        elif most_ranked in range(383,765):
            return 2
        elif most_ranked in range(765,1147):
            return 3
        elif most_ranked in range(1147,1529):
            return 4
        elif most_ranked in range(1529,1910):
            return 5


    def quartile_ranking(most_ranked):
        '''
            Method Name: quartile_ranking
            Description: This method assigns values to WIMD rank feature on quartile scale (1 to 4)
            Output: Integer value between 1 and 4
            
            Parameters:
            - most_ranked: Integer value that represents the highest frequency of WIMD rank
        '''
        if most_ranked in range(1,479):
            return 1
        elif most_ranked in range(479,956):
            return 2
        elif most_ranked in range(956,1433):
            return 3
        elif most_ranked in range(1433,1910):
            return 4


class BinaryDataTransformer(BaseEstimator, TransformerMixin):
    
    
    def __init__(self):
        '''
            Method Name: __init__
            Description: This method initializes instance of BinaryDataTransformer class
            Output: None
        '''
        pass

    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method simply passes the fit method of transformer without execution.
            Output: self
        '''
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs categorical encoding on binary features.
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        for col in ['Read_Info_Sheet','School_Health_Records','Other_children_inhouse','Easywalk_topark','Easywalk_somewhere','Garden','Keep_in_touch_family_outside_household','Keep_in_touch_friends']:
            X_[col] = X_[col].map({'Yes':1,'No':0})
        X_['Sleeptime_ytd_halfhour'] = X_['Sleeptime_ytd_minute'].map(
            {30:1,0:0})
        X_['Awaketime_today_halfhour'] = X_['Awaketime_today_minute'].map(
            {30:1,0:0})
        X_.drop(
            ['Sleeptime_ytd_minute','Awaketime_today_minute'],axis=1,inplace=True)
        for contact_str, search_key in zip(['Contact_by_phone','Contact_by_visit','Contact_by_social_media','Contact_by_game'],['phone','social distance','social media','games']):
            X_[contact_str] = X_['Method_of_keepintouch'].apply(
                lambda x: 1 if x.find(search_key)!=-1 else 0)
        X_['Play_in_house'] = X_['Type_of_play_places'].apply(
            lambda x: 1 if "In my house" in x.strip().split(";") else 0)
        X_['Play_in_garden'] = X_['Type_of_play_places'].apply(
            lambda x: 1 if "In my garden" in x.strip().split(";") else 0)
        X_['Play_in_grass_area'] = X_['Type_of_play_places'].apply(
            lambda x: 1 if "On a local grassy area" in x.strip().split(";") else 0)
        X_['Play_in_bushes'] = X_['Type_of_play_places'].apply(
            lambda x: 1 if "In a place with bushes, trees and flowers" in x.strip().split(";") else 0)
        X_['Play_in_woods'] = X_['Type_of_play_places'].apply(
            lambda x: 1 if x.lower().find("woods")!=-1 else 0)
        X_['Play_in_field'] = X_['Type_of_play_places'].apply(
            lambda x: 1 if x.lower().find("field")!=-1 else 0)
        X_['Play_in_street'] = X_['Type_of_play_places'].apply(
            lambda x: 1 if "In the street" in x.strip().split(";") or "on my street" in x.strip().split(";") else 0)
        X_['Play_in_playground'] = X_['Type_of_play_places'].apply(
            lambda x: 1 if 'In my school playground' in x.strip().split(";") else 0)
        X_['Play_in_bike_or_park'] = X_['Type_of_play_places'].apply(
            lambda x: 1 if any(sub_x in x.lower() for sub_x in ['bike', 'park']) else 0)
        X_['Play_near_water'] = X_['Type_of_play_places'].apply(
            lambda x: 1 if any(sub_x in x.lower() for sub_x in ['beach', 'river', 'stream', 'lake', 'riverfront', 'canal', 'water']) else 0)
        X_['Bread_Brk'] = X_['Breakfast_ytd'].apply(
            lambda x: 1 if any(sub_x in x.lower() for sub_x in ['toast', 'sandwich', 'bread', 'bagel', 'au choc', 'pano', 'croissant', 'brioch', 'crumpet', 'bun', 'danish', 'roll', 'muffins', 'bacon bap']) else 0)
        X_['Sugary_Cereal_Brk'] = X_['Breakfast_ytd'].apply(
            lambda x: 1 if x.find("Sugary cereal")!=-1 else 0)
        X_['Healthy_Cereal_Brk'] = X_['Breakfast_ytd'].apply(
            lambda x: 1 if any(sub_x in x.strip().split(";") for sub_x in ['Healthy cereal e.g. porridge, weetabix, readybrek, muesli, branflakes, cornflakes', 'Shreddies','GRANOLA','cereal']) else 0)
        X_['Fruits_Brk'] = X_['Breakfast_ytd'].apply(
            lambda x: 1 if any(sub_x in x.lower() for sub_x in ['fruit', 'strawberries', 'strawberreis', 'apple', 'Healthy Shake', 'smoothy']) else 0)
        X_['Yogurt_Brk'] = X_['Breakfast_ytd'].apply(
            lambda x: 1 if any(sub_x in x.lower() for sub_x in ['yoghurt', 'yogurt']) else 0)
        X_['Nothing_Brk'] = X_['Breakfast_ytd'].apply(
            lambda x: 1 if x == "Nothing" else 0)
        X_['Cooked_Breakfast_Brk'] = X_['Breakfast_ytd'].apply(
            lambda x: 1 if any(sub_x in x.lower() for sub_x in ['cooked', 'omlet', 'boiled', 'rice', 'beans', 'porige', 'egg', 'portage', 'pancake', 'pankaces', 'waffle']) else 0)
        X_['Snacks_Brk'] = X_['Breakfast_ytd'].apply(
            lambda x: 1 if any(sub_x in x.lower() for sub_x in ['snacks', 'finger', 'pop tarts', 'biscuits', 'marshmallow', 'nutella', 'crackerbread']) else 0)
        return X_


class OrdinalDataTransformer(BaseEstimator, TransformerMixin):
    
    
    def __init__(self):
        '''
            Method Name: __init__
            Description: This method initializes instance of OrdinalDataTransformer class
            Output: None
        '''
        pass

    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method simply passes the fit method of transformer without execution.
            Output: self
        '''
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs categorical encoding on ordinal features as intermediate step for one hot encoding or catboost encoding.
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        X_['Fruitveg_ytd'] = X_['Fruitveg_ytd'].map(
            {'2 Or More Fruit and Veg': '2', '1 Piece':'1', 'No':'0', '2':'2', '1':'1'})        
        agreescale_col = ['Doingwell_schoolwork','Lots_of_choices_important','Lots_of_things_good_at','Feel_partof_community']
        for col in agreescale_col:
            X_[col] = X_[col].map(
                {'Strongly disagree':'0', 'Disagree':'1', "Don't agree or disagree":'2', 'Agree':'3','Strongly agree':'4'})
        X_['Outdoorplay_freq'] = X_['Outdoorplay_freq'].map(
            {"I don't play":'0', 'Hardly ever':'1', 'A few days each week':'2', 'Most days':'3'})
        X_['Enoughtime_toplay'] = X_['Enoughtime_toplay'].map(
            {'Yes, I have loads':'3', "Yes, it's just about enough":'2','No, I would like to have a bit more':'1','No, I need a lot more':'0'})
        X_['Play_inall_places'] = X_['Play_inall_places'].map(
            {'I can play in some of the places I would like to': '2','I can only play in a few places I would like to': '1','I can play in all the places I would like to': '3','I can hardly play in any of the places I would like to': '0'})
        return X_


class ScalingTransformer(BaseEstimator, TransformerMixin):
    
    
    def __init__(self):
        '''
            Method Name: __init__
            Description: This method initializes instance of ScalingTransformer class
            Output: None
        '''
        pass


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method fits dataset onto MinMax scaler.
            Output: self

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_)
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method performs transformation on features using MinMax scaler.
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        X_ = pd.DataFrame(self.scaler.transform(X_), columns = X_.columns)
        return X_


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    
    
    def __init__(
            self, method, model, scaling_indicator= 'no', cluster_indicator= 'no', damping = None, number=None):
        '''
            Method Name: __init__
            Description: This method initializes instance of MissingTransformer class
            Output: None

            Parameters:
            - method: String that represents method of feature selection (Accepted values are 'BorutaShap', 'Lasso', 'FeatureImportance_ET', 'FeatureImportance_self', 'MutualInformation', 'ANOVA', 'FeatureWiz')
            - model: Model object
            - scaling_indicator: String that represents method of performing feature scaling. (Accepted values are 'MinMax' and 'no'). Default value is 'no'
            - cluster_indicator: String indicator of including cluster-related feature (yes or no). Default value is 'no'
            - damping: Float value (range from 0.5 to 1 not inclusive) as an additional hyperparameter for Affinity Propagation clustering algorithm. Default value is None.
            - number: Integer that represents number of features to select. Minimum value required is 1. Default value is None.

        '''
        self.method = method
        self.model = model
        self.scaling_indicator = scaling_indicator
        self.cluster_indicator = cluster_indicator
        self.damping = damping
        self.number = number


    def fit(self, X, y=None):
        '''
            Method Name: fit
            Description: This method removes columns with zero variance, identifies subset of columns from respective feature selection techniques and fits clustering from features using Affinity Propagation if clustering indicator is 'yes'.
            Output: self

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        y = y.reset_index(drop=True)
        self.dropconstant = fes.DropConstantFeatures(missing_values='ignore')
        self.dropconstant.fit(X_)
        X_ = self.dropconstant.transform(X_)
        if self.method == 'BorutaShap':
            borutashap = BorutaShap(
                importance_measure = 'shap', classification = True)
            borutashap.fit(
                X = X_, y = y, verbose = False, stratify = y)
            self.sub_columns = borutashap.Subset().columns.to_list()
        elif self.method == 'Lasso':
            imp_model = LogisticRegression(
                random_state=random_state,penalty='l1',max_iter=1000, solver='saga')
            imp_model.fit(X_,y)
            if self.scaling_indicator == 'no':
                result = pd.DataFrame(
                    [pd.Series(X_.columns),pd.Series(np.abs(imp_model.coef_[0])*np.array(X_).std(axis=0))], index=['Variable','Value']).T
            else:
                result = pd.DataFrame(
                    [pd.Series(X_.columns),pd.Series(np.abs(imp_model.coef_[0]))], index=['Variable','Value']).T
            result['Value'] = result['Value'].astype('float64')
            self.sub_columns =  result.loc[result['Value'].nlargest(self.number).index.tolist()]['Variable'].tolist()
        elif self.method == 'FeatureImportance_ET':
            fimp_model = ExtraTreesClassifier(random_state=random_state)
            fimportance_selector = SelectFromModel(
                fimp_model,max_features=self.number,threshold=0.0)
            fimportance_selector.fit(X_,y)
            self.sub_columns = X_.columns[fimportance_selector.get_support()].to_list()
        elif self.method == 'FeatureImportance_self':
            fimp_model = clone(self.model)
            fimportance_selector = SelectFromModel(
                fimp_model,max_features=self.number,threshold=0.0)
            fimportance_selector.fit(X_,y)
            self.sub_columns = X_.columns[fimportance_selector.get_support()].to_list()
        elif self.method == 'MutualInformation':
            values = mutual_info_classif(X_,y,random_state=random_state)
            result = pd.DataFrame(
                [pd.Series(X_.columns),pd.Series(values)], index=['Variable','Value']).T
            result['Value'] = result['Value'].astype('float64')
            self.sub_columns =  result.loc[result['Value'].nlargest(self.number).index.tolist()]['Variable'].tolist()
        elif self.method == 'ANOVA':
            fclassif_selector = SelectKBest(f_classif,k=self.number)
            fclassif_selector.fit(X_,y)
            self.sub_columns =  X_.columns[fclassif_selector.get_support()].to_list()
        elif self.method == 'FeatureWiz':
            selector = FeatureWiz(verbose=0)
            selector.fit(X_, y)
            self.sub_columns = selector.features
        if self.cluster_indicator == 'yes':
            self.affinitycluster = AffinityPropagation(random_state=random_state, damping = self.damping)
            if self.sub_columns != []:
                self.affinitycluster.fit(X_[self.sub_columns])
                dist = pairwise_distances(
                    X_[self.sub_columns], self.affinitycluster.cluster_centers_).min(axis=1)
            else:
                self.affinitycluster.fit(X_)
                dist = pairwise_distances(
                    X_, self.affinitycluster.cluster_centers_).min(axis=1)
            self.scaler = MinMaxScaler()
            self.scaler.fit(dist.reshape(-1, 1))
        return self
    

    def transform(self, X, y=None):
        '''
            Method Name: transform
            Description: This method removes columns with constant variance, followed by subset of columns identified from feature selection and add additional feature named "cluster_distance" if cluster indicator is "yes" 
            Output: Transformed features from dataset in dataframe format.

            Parameters:
            - X: Features from dataset
        '''
        X_ = X.copy()
        X_ = self.dropconstant.transform(X_)
        if self.sub_columns != []:
            X_ = X_[self.sub_columns]
        if self.cluster_indicator == 'yes':
            X_['cluster_distance'] = pairwise_distances(X_, self.affinitycluster.cluster_centers_).min(axis=1)
            X_['cluster_distance'] = self.scaler.transform(np.array(X_['cluster_distance']).reshape(-1, 1))
        return X_