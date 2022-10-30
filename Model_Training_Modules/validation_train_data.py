'''
Author: Liaw Yi Xian
Last Modified: 30th October 2022
'''

import json
import pandas as pd
from pymongo import MongoClient
from Application_Logger.logger import App_Logger
from Application_Logger.exception import CustomException
import os, shutil, sys

MONGO_DB_URL = os.getenv('MONGO_DB_URL')

class rawtraindatavalidation:


    def __init__(self, dbname, collectionname, file_object):
        '''
            Method Name: __init__
            Description: This method initializes instance of rawtraindatavalidation class
            Output: None

            Parameters:
            - dbname: String name of MongoDB database
            - collectionname: String name of collection within a given MongoDB database
            - file_object: String path of logging text file
        '''
        self.db_name = dbname
        self.collection_name = collectionname
        self.file_object = file_object
        self.log_writer = App_Logger()
        self.client = MongoClient(MONGO_DB_URL)
        self.collection = self.client[self.db_name][self.collection_name]


    def newDB(self):
        '''
            Method Name: newDB
            Description: This method creates a new database and table in MongoDB database based on a given schema object.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, f"Start creating new collection ({self.collection_name}) in MongoDB database ({self.db_name})")
        try:
            data = json.load(open(self.datafilepath + '/data.json'))
            for row in data:
                row['_id'] = row['ID']
                del(row['ID'])
            self.collection.insert_many(data)
        except ConnectionError:
            self.log_writer.log(self.file_object, "Error connecting to MongoDB database")
            raise Exception("Error connecting to MongoDB database")
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        self.log_writer.log(
            self.file_object, f"Finish creating new collection ({self.collection_name}) in MongoDB database ({self.db_name})")
    

    def compile_data_from_DB(self):
        '''
            Method Name: compile_data_from_DB
            Description: This method compiles data from MongoDB database into csv file for further data preprocessing.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start writing compiled good training data into a new CSV file")
        try:
            result = self.collection.find()
            data = pd.DataFrame(list(result))
            data.to_csv(self.compiledir, index=False)
        except ConnectionError:
            self.log_writer.log(self.file_object, "Error connecting to MongoDB database")
            raise Exception("Error connecting to MongoDB database")
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        self.log_writer.log(
            self.file_object, "Finish writing compiled good training data into a new CSV file")


    def file_initialize(self):
        '''
            Method Name: file_initialize
            Description: This method creates the list of folders mentioned in the filelist if not exist. If exist, this method deletes the existing folders and creates new ones. Note that manual archiving will be required if backup of existing files is required.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start initializing folder structure")
        try:
            for folder in self.folders:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                os.makedirs(os.path.dirname(folder), exist_ok=True)
                self.log_writer.log(
                    self.file_object, f"Folder {folder} has been initialized")
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        self.log_writer.log(
            self.file_object, "Finish initializing folder structure")


    def initial_data_preparation(self, folders, datafilepath, compiledir):
        '''
            Method Name: initial_data_preparation
            Description: This method performs all the preparation tasks for the data to be ingested into MongoDB database.
            Output: None

            Parameters:
            - folders: List of string file paths for initializing folder structure
            - datafilepath: String file path for specified folder where original data is located
            - compiledir: String path where good quality data is compiled from database
        '''
        self.folders = folders
        self.datafilepath = datafilepath
        self.compiledir = compiledir
        self.log_writer.log(self.file_object, "Start initial data preparation")
        self.file_initialize()
        self.newDB()
        self.compile_data_from_DB()
        self.log_writer.log(self.file_object, "Finish initial data preparation")