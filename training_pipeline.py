from Model_Training_Modules.validation_train_data import rawtraindatavalidation
from Model_Training_Modules.train_preprocessing import train_Preprocessor
from Model_Training_Modules.model_training import model_trainer
import pandas as pd
import streamlit as st

DATABASE_LOG = "Training_Logs/Training_Main_Log.txt"
DATA_SOURCE = 'Training_Data_FromDB/Training_Data.csv'
PREPROCESSING_LOG = "Training_Logs/Training_Preprocessing_Log.txt"
RESULT_DIR = 'Intermediate_Train_Results/'
TRAINING_LOG = "Training_Logs/Training_Model_Log.txt"


def main():
    st.title("Mental Health Status Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Mental Health Status Prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    if st.button("Training Data Validation"):
        trainvalidator = rawtraindatavalidation(
            dbname = 'mentalhealth', collectionname = 'mentalhealthdata', file_object = DATABASE_LOG)
        folders = ['Training_Data_FromDB/','Intermediate_Train_Results/','Caching/','Saved_Models/']
        trainvalidator.initial_data_preparation(
            folders = folders, datafilepath = "Training_Batch_Files", compiledir= DATA_SOURCE)
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Exploratory Data Analysis"):
        preprocessor = train_Preprocessor(
            file_object= PREPROCESSING_LOG, datapath = DATA_SOURCE, result_dir= RESULT_DIR)
        preprocessor.eda()
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Training Data Preprocessing"):
        preprocessor = train_Preprocessor(
            file_object= PREPROCESSING_LOG, datapath = DATA_SOURCE, result_dir= RESULT_DIR)
        preprocessor.data_preprocessing(
            col_drop_path= 'Columns_Drop_from_Original.csv')
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Model Selection"):
        trainer = model_trainer(
            file_object= TRAINING_LOG)
        X = pd.read_csv(RESULT_DIR + 'X.csv')
        y = pd.read_csv(RESULT_DIR + 'y.csv')
        trainer.model_selection(
            input = X, output = y, num_trials = 40, folderpath = RESULT_DIR)
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Final Model Training"):
        trainer = model_trainer(
            file_object= TRAINING_LOG)
        X = pd.read_csv(RESULT_DIR + 'X.csv')
        y = pd.read_csv(RESULT_DIR + 'y.csv')
        trainer.final_model_tuning(
            input_data = X, output_data = y, num_trials = 40, folderpath = RESULT_DIR)
        st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")

        
if __name__=='__main__':
    main()