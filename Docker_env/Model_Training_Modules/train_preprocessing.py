'''
Author: Liaw Yi Xian
Last Modified: 20th October 2022
'''

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from Application_Logger.logger import App_Logger
import numpy as np
import joblib
from feature_engine.imputation import CategoricalImputer
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px
from tqdm import tqdm
from collections import Counter
import datetime

random_state=120

class train_Preprocessor:


    def __init__(self, file_object, datapath, result_dir):
        '''
            Method Name: __init__
            Description: This method initializes instance of train_Preprocessor class
            Output: None

            Parameters:
            - file_object: String path of logging text file
            - datapath: String path where compiled data is located
            - result_dir: String path for storing intermediate results from running this class
        '''
        self.file_object = file_object
        self.datapath = datapath
        self.result_dir = result_dir
        self.log_writer = App_Logger()


    def extract_compiled_data(self):
        '''
            Method Name: extract_compiled_data
            Description: This method extracts data from a csv file and converts it into a pandas dataframe.
            Output: A pandas dataframe
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start reading compiled data from database")
        try:
            data = pd.read_csv(self.datapath)
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Fail to read compiled data from database with the following error: {e}")
            raise Exception(
                f"Fail to read compiled data from database with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish reading compiled data from database")
        return data


    def remove_irrelevant_columns(self, data, cols):
        '''
            Method Name: remove_irrelevant_columns
            Description: This method removes columns from a pandas dataframe, which are not relevant for analysis.
            Output: A pandas DataFrame after removing the specified columns. In addition, columns that are removed will be stored in a separate csv file.
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
            - cols: List of irrelevant columns to remove from dataset
        '''
        self.log_writer.log(
            self.file_object, "Start removing irrelevant columns from the dataset")
        try:
            data = data.drop(cols, axis=1)
            result = pd.concat(
                [pd.Series(cols, name='Columns_Removed'), pd.Series(["Irrelevant column"]*len(cols), name='Reason')], axis=1)
            result.to_csv(self.result_dir+self.col_drop_path, index=False)
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Irrelevant columns could not be removed from the dataset with the following error: {e}")
            raise Exception(
                f"Irrelevant columns could not be removed from the dataset with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish removing irrelevant columns from the dataset")
        return data


    def remove_duplicated_rows(self, data):
        '''
            Method Name: remove_duplicated_rows
            Description: This method removes duplicated rows from a pandas dataframe.
            Output: A pandas DataFrame after removing duplicated rows. In addition, duplicated records that are removed will be stored in a separate csv file labeled "Duplicated_Records_Removed.csv"
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
        '''
        self.log_writer.log(
            self.file_object, "Start handling duplicated rows in the dataset")
        if len(data[data.duplicated()]) == 0:
            self.log_writer.log(
                self.file_object, "No duplicated rows found in the dataset")
        else:
            try:
                data[data.duplicated()].to_csv(
                    self.result_dir+'Duplicated_Records_Removed.csv', index=False)
                data = data.drop_duplicates(ignore_index=True)
            except Exception as e:
                self.log_writer.log(
                    self.file_object, f"Fail to remove duplicated rows with the following error: {e}")
                raise Exception(
                    f"Fail to remove duplicated rows with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish handling duplicated rows in the dataset")
        return data
    

    def features_and_labels(self,data,target_col):
        '''
            Method Name: features_and_labels
            Description: This method splits a pandas dataframe into two pandas objects, consist of features and target labels.
            Output: Two pandas/series objects consist of features and labels separately.
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
            - target_col: Name of target column
        '''
        self.log_writer.log(
            self.file_object, "Start separating the data into features and labels")
        try:
            X = data.drop(target_col, axis=1)
            y = data[target_col]
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Fail to separate features and labels with the following error: {e}")
            raise Exception(
                f"Fail to separate features and labels with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish separating the data into features and labels")
        return X, y


    def derive_target(self, data):
        '''
            Method Name: derive_target
            Description: This custom method derives target variable known as "Wellbeing_Category_WMS" based on existing features related to wellbeing of children.
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
        '''
        self.log_writer.log(
            self.file_object, 'Start deriving target values on the dataset.')
        try:
            emotional_cols, behaviour_cols = [], []
            for column in data.columns:
                if column.find('Wellbeing')!=-1:
                    if column!='Calm(Behaviour_Wellbeing)':
                        data[column] = data[column].map(
                            {'Never':0,'Sometimes':1,'Always':2})
                    else:
                        data[column] = data[column].map(
                            {'Never':2,'Sometimes':1,'Always':0})
                if column.find('Emotional_Wellbeing')!=-1:
                    emotional_cols.append(column)
                elif column.find('Behaviour_Wellbeing')!=-1:
                    behaviour_cols.append(column)
            data['Emotional_Wellbeing_Score'] = data[emotional_cols].sum(axis=1)
            data['Behaviour_Wellbeing_Score'] = data[behaviour_cols].sum(axis=1)
            data['Emotional_Wellbeing_Category_WMS'] = data['Emotional_Wellbeing_Score'].apply(
                lambda x: 'normal' if x<10 else 'significant')
            data['Behaviour_Wellbeing_Category_WMS'] = data['Behaviour_Wellbeing_Score'].apply(
                lambda x: 'normal' if x<6 else 'significant')
            data['Wellbeing_Category_WMS'] = (data['Emotional_Wellbeing_Category_WMS'] + data['Behaviour_Wellbeing_Category_WMS']).map(
            {'normalnormal':'normal','significantnormal':'emotional_significant','normalsignificant':'behaviour_significant','significantsignificant':'emotional_and_behaviour_significant'})
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Fail to derive target values on the dataset with the following error: {e}")
            raise Exception(
                f"Fail to derive target values on the dataset with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish deriving target values on the dataset.")
        return data, emotional_cols, behaviour_cols


    def reformat_time_features(self, data):
        '''
            Method Name: reformat_time_features
            Description: This custom method reformats certain time features of the dataset for further data preprocessing.
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
        '''
        self.log_writer.log(
            self.file_object, 'Start reformatting certain time features on the dataset.')
        try:
            data['Month'] = data['Month'].map(
                {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12})
            data['Birth_Date'] = pd.to_datetime(data[['Year','Month','Day']])
            data['Timestamp'] = data['Timestamp'].apply(
                lambda x: datetime.datetime.strptime(x[:18], '%Y/%m/%d %I:%M:%S'))
            data['Sleeptime_ytd'] = data['Sleeptime_ytd'].apply(
                lambda x: datetime.datetime.strptime(x, '%I:%M%p') + datetime.timedelta(days=1) if datetime.datetime.strptime(x, '%I:%M%p').hour < 13 else datetime.datetime.strptime(x, '%I:%M%p'))
            data['Awaketime_today'] = data['Awaketime_today'].apply(
                lambda x: datetime.datetime.strptime(x, '%I:%M%p') + datetime.timedelta(days=1))
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Fail to reformat certain time features on the dataset with the following error: {e}")
            raise Exception(
                f"Fail to reformat certain time features on the dataset with the following error: {e}")
        self.log_writer.log(
            self.file_object, 'Finish reformatting certain time features on the dataset.')
        return data


    def category_imputing(self, data):
        '''
            Method Name: category_imputing
            Description: This custom method performs missing data imputation for categorical variables based on most frequent category for every feature with missing values.
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
        '''
        self.log_writer.log(
            self.file_object, 'Start imputing missing categorical data by most frequent category')
        try:
            data['WIMD_2019_Rank'] = data['WIMD_2019_Rank'].apply(
                lambda x: np.nan if x=='not found' else x)
            imputer_missing = CategoricalImputer(
                imputation_method='frequent',ignore_format=True,variables=X.columns[data.isnull().sum() > 0].tolist())
            data = imputer_missing.fit_transform(data)
            joblib.dump(
                imputer_missing,open(self.result_dir + f'CategoryImputer.pkl','wb'))
            most_ranked = int(data[data['WIMD_2019_Quartile'] == 'not found']['WIMD_2019_Rank'].unique()[0])
            data['WIMD_2019_Decile'] = data['WIMD_2019_Decile'].apply(
                lambda x: str(train_Preprocessor.decile_ranking(most_ranked)) if x=='not found' else x)
            data['WIMD_2019_Quintile'] = data['WIMD_2019_Quintile'].apply(
                lambda x: str(train_Preprocessor.quintile_ranking(most_ranked)) if x=='not found' else x)
            data['WIMD_2019_Quartile'] = data['WIMD_2019_Quartile'].apply(
                lambda x: str(train_Preprocessor.quartile_ranking(most_ranked)) if x=='not found' else x)
        except Exception as e:
            self.log_writer.log(
                self.file_object, f"Fail to impute missing categorical data by most frequent category with the following error: {e}")
            raise Exception(
                f"Fail to impute missing categorical data by most frequent category with the following error: {e}")
        self.log_writer.log(
            self.file_object, "Finish imputing missing categorical data by most frequent category")
        return data


    def eda(self):
        '''
            Method Name: eda
            Description: This method performs exploratory data analysis on the entire dataset, while generating various plots/csv files for reference.
            Output: None
        '''
        self.log_writer.log(
            self.file_object, 'Start performing exploratory data analysis')
        path = os.path.join(self.result_dir, 'EDA')
        if not os.path.exists(path):
            os.mkdir(path)
        scat_path = os.path.join(path, 'High_Correlation_Scatterplots')
        if not os.path.exists(scat_path):
            os.mkdir(scat_path)
        data = self.extract_compiled_data()
        data = data[data['Use_questionnaire'] == 'Yes']
        data.drop('Use_questionnaire',axis=1, inplace=True)
        # Extract basic information about dataset
        pd.DataFrame({"name": data.columns, "non-nulls": len(data)-data.isnull().sum().values, "type": data.dtypes.values}).to_csv(self.result_dir + "EDA/Data_Info.csv",index=False)
        # Extract summary statistics about dataset
        data.describe().T.to_csv(
            self.result_dir + "EDA/Data_Summary_Statistics.csv")
        # Plotting proportion of null values of dataset
        null_prop = []
        for col in data.columns:
            null_prop.append(data[col].isnull().sum())
        null_results = pd.DataFrame(
            [data.columns, null_prop], index=['Variable','Number']).T
        null_results = null_results[null_results['Number']>0].sort_values(by='Number',ascending=False)
        plt.figure(figsize=(24, 16),dpi=100)
        barplot = sns.barplot(
            data=null_results,y='Variable',x='Number',palette='flare_r')
        for rect in barplot.patches:
            width = rect.get_width()
            plt.text(
                0.5+rect.get_width(), rect.get_y()+0.5*rect.get_height(),'%.1d' % width, ha='center', va='center')
        plt.title("Number of null values", fontdict={'fontsize':24})
        plt.savefig(
            self.result_dir+"EDA/Proportion of null values",bbox_inches='tight', pad_inches=0.2)
        plt.clf()
        # Extract information related to number of unique values for every feature of dataset
        nunique_values = []
        for col in data.columns:
            nunique_values.append(data[col].nunique())
        pd.DataFrame(
            [data.columns, nunique_values], index=['Variable','Number']).T.to_csv(self.result_dir + "EDA/Number_Unique_Values.csv", index=False)
        for col in tqdm(data.columns):
            col_path = os.path.join(path, col)
            if not os.path.exists(col_path):
                os.mkdir(col_path)
            data[col].value_counts().to_csv(
                self.result_dir+f'EDA/{col}/{col}_nunique.csv')
            if data[col].nunique() < 50:
                countplot = sns.countplot(data = data, y = col)
                for rect in countplot.patches:
                    width = rect.get_width()/len(data)*100
                    plt.text(
                        rect.get_width()+8, rect.get_y()+0.5*rect.get_height(), '%.2f' % width + '%', ha='center', va='center')
                plt.title(f'{col} distribution')
                plt.savefig(self.result_dir+f'EDA/{col}/{col}_nunique.png', bbox_inches='tight', pad_inches=0.2)
                plt.clf()
            if col in ['Breakfast_ytd','Method_of_keepintouch','Type_of_play_places']:
                splitted_list = []
                for i in list(data[col]):
                    if type(i)!=float:
                        splitted_list.extend(i.split(';'))
                pd.DataFrame(Counter(splitted_list).most_common(),columns=[col,'Frequency']).to_csv(
                    self.result_dir + f"EDA/{col}/{col}_Unique_Values_Split.csv", index=False)            
        # Deriving target variable
        data, emotional_cols, behaviour_cols = self.derive_target(data)
        # Plotting target class distribution
        target_dist = data['Wellbeing_Category_WMS'].value_counts().reset_index().rename(columns={'index': 'Wellbeing_Category_WMS','Wellbeing_Category_WMS':'Number_Obs'})
        fig6 = px.bar(
            target_dist,x='Wellbeing_Category_WMS',y='Number_Obs',title=f"Target Class Distribution",text_auto=True)
        fig6.write_image(self.result_dir + f"EDA/Target_Class_Distribution.png")
        #  Plotting distribution of features by target class 
        for col in tqdm(data.columns):
            if (data[col].nunique() < 50) & (col.find('Wellbeing')==-1):
                countplot = sns.countplot(
                    data = data, y = col, hue='Wellbeing_Category_WMS')
                for rect in countplot.patches:
                    width = rect.get_width()/len(data)*100
                    plt.text(
                        rect.get_width()+8, rect.get_y()+0.5*rect.get_height(), '%.2f' % width + '%', ha='center', va='center')
                plt.title(f'{col} distribution by wellbeing category', fontdict={'fontsize':24})
                plt.legend(loc="best")
                plt.savefig(
                    self.result_dir+f'EDA/{col}/{col}_nunique_by_target.png', bbox_inches='tight', pad_inches=0.2)
                plt.clf()
            if data[col].isnull().sum() != 0:
                # Plotting histogram of number of missing values of features by target class
                fig4 = px.histogram(
                    data,x=data[col].isnull(),color='Wellbeing_Category_WMS',title=f"{col} Number of Missing Values by Target",text_auto=True, height=800, width=800)
                fig4.write_image(
                    self.result_dir + f"EDA/{col}/{col}_Count_Missing_By_Target.png")
        self.log_writer.log(
            self.file_object, 'Finish performing exploratory data analysis')


    def data_preprocessing(self, col_drop_path):
        '''
            Method Name: data_preprocessing
            Description: This method performs all the data preprocessing tasks for the data.
            Output: None
            
            Parameters:
            - col_drop_path: String path that stores list of columns that are removed from the data
        '''
        self.log_writer.log(self.file_object, 'Start of data preprocessing')
        self.col_drop_path = col_drop_path
        data = self.extract_compiled_data()
        # Only use data where respondent provides permission to use questionnaire
        data = data[data['Use_questionnaire'] == 'Yes'].reset_index(drop=True)
        data, emotional_cols, behaviour_cols = self.derive_target(data)
        data = self.reformat_time_features(data)
        cols_to_remove = ['_id','ID.1','LSOA_Code','LSOA_Name(English)','LSOA_Name(Cymraeg)','Use_questionnaire','Emotional_Wellbeing_Score','Behaviour_Wellbeing_Score','Emotional_Wellbeing_Category_WMS','Behaviour_Wellbeing_Category_WMS','Year','Month','Day'] + emotional_cols + behaviour_cols
        data = self.remove_irrelevant_columns(
            data = data, cols = cols_to_remove)
        data = self.remove_duplicated_rows(data = data)
        X, y = self.features_and_labels(
            data = data, target_col='Wellbeing_Category_WMS')
        X = self.category_imputing(X)
        X.to_csv(self.result_dir+'X.csv',index=False)
        y.to_csv(self.result_dir+'y.csv',index=False)
        self.log_writer.log(self.file_object, 'End of data preprocessing')


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
