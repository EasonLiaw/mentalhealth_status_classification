import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import time, timedelta, date, datetime
from Model_Training_Modules.validation_train_data import rawtraindatavalidation
from Model_Training_Modules.train_preprocessing import train_Preprocessor
from Model_Training_Modules.model_training import model_trainer
import os

DATABASE_LOG = "Training_Logs/Training_Main_Log.txt"
DATA_SOURCE = 'Training_Data_FromDB/Training_Data.csv'
PREPROCESSING_LOG = "Training_Logs/Training_Preprocessing_Log.txt"
RESULT_DIR = 'Intermediate_Train_Results/'
TRAINING_LOG = "Training_Logs/Training_Model_Log.txt"

def main():
    st.title("Mental Health Status Classification")
    html_temp = """
    <div style="background-color:purple;padding:10px">
    <h2 style="color:white;text-align:center;">Mental Health Status Classification App </h2>
    <p></p>
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
        if 'Training_Data_FromDB' not in os.listdir(os.getcwd()):
            st.error("Database has not yet inserted. Have u skipped Training Data Validation step?")
        else:
            preprocessor = train_Preprocessor(
                file_object= PREPROCESSING_LOG, datapath = DATA_SOURCE, result_dir= RESULT_DIR)
            preprocessor.eda()
            st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    if st.button("Training Data Preprocessing"):
        if 'Training_Data_FromDB' not in os.listdir(os.getcwd()):
            st.error("Database has not yet inserted. Have u skipped Training Data Validation step?")
        else:
            preprocessor = train_Preprocessor(
                file_object= PREPROCESSING_LOG, datapath = DATA_SOURCE, result_dir= RESULT_DIR)
            preprocessor.data_preprocessing(
                col_drop_path= 'Columns_Drop_from_Original.csv')
            st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    model_names = st.multiselect(
        "Select the following model you would like to train for model selection", options=['LogisticRegression', 'LinearSVC','KNeighborsClassifier', 'GaussianNB', 'DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier'])
    if st.button("Model Selection"):
        if not os.path.isdir(RESULT_DIR) or 'X.csv' not in os.listdir(RESULT_DIR):
            st.error("Data has not yet been preprocessed. Have u skipped Training Data Preprocessing step?")
        else:
            trainer = model_trainer(file_object= TRAINING_LOG)
            X = pd.read_csv(RESULT_DIR + 'X.csv')
            y = pd.read_csv(RESULT_DIR + 'y.csv')
            trainer.model_selection(
                input = X, output = y, num_trials = 40, folderpath = RESULT_DIR, model_names = model_names)
            st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    choices = dict()
    def format_func(option):
          return choices[option]
    choices = {1: 'LogisticRegression', 2: 'LinearSVC', 3: 'KNeighborsClassifier', 4: 'GaussianNB', 5: 'DecisionTreeClassifier', 6: 'RandomForestClassifier', 7: 'ExtraTreesClassifier', 8: 'AdaBoostClassifier', 9: 'GradientBoostingClassifier', 10: 'XGBClassifier', 11: 'LGBMClassifier', 12: 'CatBoostClassifier'}
    model_list = choices.copy()
    model_number = st.selectbox(
        "Select the following model you would like to train for final model deployment", options=list(choices.keys()), format_func=format_func)
    if st.button("Final Model Training"):
        if not os.path.isdir(RESULT_DIR) or 'X.csv' not in os.listdir(RESULT_DIR):
            st.error("Data has not yet been preprocessed. Have u skipped Training Data Preprocessing step?")
        elif not os.path.isdir(RESULT_DIR + model_list[model_number]):
            st.error("Model algorithm selection has not been done. Have u skipped model selection step?")
        else:
            trainer = model_trainer(file_object= TRAINING_LOG)
            X = pd.read_csv(RESULT_DIR + 'X.csv')
            y = pd.read_csv(RESULT_DIR + 'y.csv')
            trainer.final_model_tuning(
                input_data = X, output_data = y, num_trials = 40, folderpath = RESULT_DIR, model_number = model_number)
            st.success("This step of the pipeline has been completed successfully. Check the local files for more details.")
    with st.expander("Model Prediction"):
        Read_Info_Sheet = st.selectbox(
            'Have u read the information sheet?', options=['Yes','No'])
        School_Health_Records = st.selectbox(
            'Do u provide permission for accessing your school health records?', options=['Yes','No'])
        Birth_date = st.date_input(
            "Select your birth date", value=date(2010,1,1), min_value=date(2007,1,1), max_value = date(2012,12,31)).strftime("%Y-%m-%d")
        choices ={349:'Aberbargod 1', 177:'Aberbargod 2', 718:'Abercarn 2', 1174:'Abercynffig', 735:'Aberdulais', 936:'Abergwaun Gogledd Orllewin', 
        1398:'Abergwili', 311:'Abersychan 1', 1048:'Abersychan 3', 837:'Abersychan 5', 72:'Alway 2', 1426:'Alway 3', 393:'Alway 5', 834:'Alway 6', 1079:'Amroth', 1523:'Arberth Gwledig', 1802:'Baglan 2', 538:'Bargod 1', 
        704:'Bargod 2', 456:'Bargod 3', 
        29:'Bargod 4', 772:'Beechwood 1', 1039:'Beechwood 2', 226:'Beechwood 3', 817:'Beechwood 4', 1353:'Beechwood 5', 47:'Betws (Casnewydd) 1', 204:'Betws (Casnewydd) 2', 77:'Betws (Casnewydd) 3', 289:'Betws (Casnewydd) 4', 644:'Betws (Casnewydd) 6', 143:'Betws (Pen-y-bont ar Ogwr)', 1126:'Betws yn Rhos', 84:'Bigyn 4', 81:'Bonymaen 1', 355:'Bonymaen 4', 1116:'Botffordd', 934:'Bracla 2', 60:'Bracla 3', 1639:'Bracla 5', 
        1507:'Bracla 6', 1444:'Bronington 1', 581:'Bryn-coch', 1135:'Brynceithin', 1371:'Brynna 2', 1874:'Bryntirion Laleston a Merthyr Mawr 5', 1585:'Bryntirion Laleston a Merthyr Mawr 6', 1450:'Burton', 1505:'Caer-went', 27:'Caerau (Caerdydd) 3', 5:'Caerau (Pen-y-bont ar Ogwr) 1', 232:'Caerau (Pen-y-bont ar Ogwr) 4', 632:'Caerau (Pen-y-bont ar Ogwr) 5', 1526:'Caerllion 1', 1698:'Caerllion 2', 1413:'Caerllion 4', 1762:'Caerllion 5', 556:'Caerllion 6', 1314:'Cas-wis', 1330:'Casllwchwr Uchaf 2', 
        23:'Castell 1', 1034:'Castell 2 De', 321:'Castell 4', 
        660:'Castell 8', 1613:'Castell Cil-y-coed 1', 1008:'Catwg Sant 3', 1565:'Catwg Sant 4', 1153:'Cefn 1', 562:'Cefn 2', 816:'Cefn Cribwr', 420:'Cefn Fforest 2', 235:'Cendl 2', 910:'Cil-y-cwm', 1897:'Cilâ Gogledd 2', 1549:'Cimla 2', 1714:'Clydach 2', 297:'Clydach 3', 1238:'Coed Efa', 692:'Coed-ffranc Canol 2', 1305:'Coed-ffranc Gorllewin', 
        1412:'Coety', 1335:'Corneli 2', 723:'Corneli 3', 272:'Corneli 4', 1551:'Croesyceiliog Gogledd 2', 1231:'Crucornau Fawr', 1448:'Crymlyn 1', 251:'Crymlyn 3', 929:'Crymlyn 4', 912:'Cwm Ogwr 1', 943:'Cynwyl Elfed 1', 1440:'Cynwyl Elfed 2', 1610:'Deganwy 2', 1499:'Devauden', 62:'Doc Penfro: Llanion 1', 770:'Doc Penfro: Llanion 2', 746:'Doc Penfro: Marchnad', 1608:'Drenewydd Gelli-farch', 1501:'Dynfant 1', 616:'Eirias 1', 
        1852:'Ewlo 1', 511:'Fairwater (Tor-faen) 1', 371:'Fairwater (Tor-faen) 2', 1306:'Fairwater (Tor-faen) 3', 1743:'Fairwater (Tor-faen) 4', 1718:'Fairwood 1', 1516:'Fairwood 2', 1480:'Felin-fâch', 555:'Felindre 1', 1561:'Ffordun', 1019:'Gabalfa 2', 647:'Gele 1', 1739:'Gele 3', 482:'Gilfach', 230:'Glandwr 2', 1246:'Glanyrafon 6', 1066:'Glyn (Sir Gaerfyrddin)', 978:'Glyn-nedd 2', 365:'Glynebwy Gogledd 1', 86:'Glynebwy Gogledd 2', 
        1597:'Greenmeadow 2', 271:'Greenmeadow 3', 600:'Gwersyllt Dwyrain a De 1', 1611:'Gwyr (Abertawe) 2', 1590:'Hafren 1', 932:'Hafren 2', 1709:'Hendre (Pen-y-bont ar Ogwr) 1', 822:'Hendre (Pen-y-bont ar Ogwr) 2', 1680:'Hendy 1', 1173:'Hendy 2', 326:'Hengastell 1', 1734:'Hengastell 2', 123:'Hengoed (Caerffili) 2', 677:'Hengoed (Caerffili) 3', 1043:'Hengoed (Sir Gaerfyrddin) 1', 908:'Hengoed 2', 1702:'Langstone 1', 1595:'Langstone 2', 1660:'Langstone 3', 1490:'Larkfield', 
        373:'Liswerry 1', 171:'Liswerry 2', 778:'Liswerry 3', 620:'Liswerry 4', 134:'Liswerry 5', 740:'Liswerry 6', 750:'Liswerry 7', 1385:'Llan-non 3', 913:'Llan-wern 1', 760:'Llan-wern 2', 1380:'Llanandras 1', 1438:'Llanandras 2', 1288:'Llanbadarn Fawr', 1391:'Llanbadog', 987:'Llanbradach 2', 1815:'Llandaf 1', 1531:'Llandaf 3', 1278:'Llanddarog', 998:'Llanddewi', 797:'Llanddewi a Green Lane 2', 
        1277:'Llanddulas', 1895:'Llandeilo Ferwallt 1', 1825:'Llandeilo Ferwallt 2', 1269:'Llandeilo Gresynni', 1263:'Llandinam', 769:'Llandrindod Gogledd', 1368:'Llandrinio', 1303:'Llandybïe 1', 1399:'Llandysilio', 1291:'Llanfihangel Troddi', 1300:'Llanfihangel Ysgeifiog', 673:'Llanfihangel-ar-Arth 1', 625:'Llangollen 1', 1058:'Llangollen 2', 1318:'Llangollen 3', 1337:'Llangollen Wledig', 1459:'Llangrallo Isaf', 1765:'Llangyfelach 3', 1605:'Llangynidr', 939:'Llangynnwr 1',
        721:'Llangynwyd 2', 727:'Llannerch-y-medd', 1664:'Llansamlet 7', 1313:'Llansanffraid', 1064:'Llansilin', 1282:'Llanyrafon Gogledd', 1309:'Lliedi 2', 366:'London Road', 1563:'Maerun 2', 1522:'Maerun 3', 1241:'Maes-car/Llywel', 322:'Maesteg Dwyrain 2', 1150:'Maesteg Gorllewin 1', 79:'Malpas 2', 1441:'Marchwiel 1', 1357:'Marchwiel 2', 1487:'Margam 2', 1906:'Mayals 2', 1104:'Meifod a Llanfihangel', 1614:'Mill 2', 
        397:'Morfa 1', 299:'Morfa 3', 237:'Morgan Jones 2', 926:'Morgan Jones 4', 391:'Moria 1', 1905:'Newton (Abertawe) 1', 1801:'Newton (Abertawe) 2', 1603:'Newton (Pen-y-bont ar Ogwr) 1', 1142:'Neyland Dwyrain', 852:'Neyland Gorllewin', 1767:'Notais 1', 1669:'Notais 2', 1128:'Parc 2', 1586:'Pembre 1', 1255:'Pen-clawdd 2', 924:'Pen-dre', 877:'Pen-twyn 1', 1299:'Pen-y-groes (Sir Gaerfyrddin) 1', 1833:'Pen-yr-heol (Caerffili) 2', 536:'Pen-yr-heol (Caerffili) 3', 
        1374:'Pen-yr-heol (Caerffili) 5', 31:'Penderi 1', 279:'Penderi 7', 399:'Penfro: Santes Fair Gogledd', 419: "Penlle'r-gaer 2", 1781:'Pennard 1', 1312:'Penprysg 2', 1554:'Penycae a De Rhiwabon 1', 1405:'Penyrheol (Abertawe) 3', 241:'Pilgwenlli 2', 88:'Pilgwenlli 3', 99:'Plas Madoc', 1884:'Pont-y-clun 3', 1171:'Pont-y-clun 4', 744:'Pontardawe 1', 589:'Pontardawe 2', 1653:'Pontarddulais 1', 820:'Pontarddulais 2', 964:'Pontarddulais 3', 315:'Pontlotyn', 
        93:'Pontnewydd 1', 975:'Pontnewydd 2', 661:'Pontnewynydd', 1851:'Pontybrenin 2', 1560:'Porth Sgiwed', 510:'Porthcawl Dwyrain Canol 2', 524:'Porthcawl Gorllewin Canol 1', 1364:'Porthcawl Gorllewin Canol 2', 1025:'Prestatyn De Orllewin 2', 389:'Prestatyn Dwyrain 1', 592:'Prestatyn Gogledd 1', 836:'Prestatyn Gogledd 2', 1706:'Rest Bay 1', 1779:'Rest Bay 2', 1068:'Rhaeadr Gwy', 1463:'Rhaglan', 447:'Rhisga Dwyrain 2', 1387:'Rhisga Dwyrain 3', 844:'Rhisga Gorllewin 1', 1769:'Rhiw 1',
        754:'Rhiwabon 1', 1304:'Rhiwcynon', 1333:'Rhos 1', 764:'Rhosyr', 329:'Ringland 1', 645:'Ringland 3', 102:'Ringland 4', 1295:'Rogiet', 865:'Sain Silian 1', 330:'Sain Silian 2', 1276:'Sain Silian 3', 421:'Sain Silian 4', 1386:'Sain Silian 5', 956:'Sain Silian 6', 1200:'Sanclêr 1', 1245:'Sanclêr 2', 1092:'Sandfields Dwyrain 3', 3:'Sant Iago 3', 1878:'Sant Martin 5', 188:'Sarn 1', 
        949:'Sarn 2', 1751:'Sgeti 1', 1838:'Sgeti 2', 382:'Sgeti 4', 1784:'Sgeti 6', 1881:'Sgeti 8', 53:'Sirhywi 2', 381:'Snatchwood', 1579:'St John 1', 515:'St John 2', 1382:'St. Arvans', 1252: "St. Christopher's", 302:'St. Dials 1', 584:'St. Dials 2', 1903:'St. Kingsmark 1', 1834:'St. Kingsmark 2', 1004: "St. Mary's", 269:'St. Thomas 1', 749:'St. Thomas 4', 135:'Stow Hill 3', 
        1404:'Tal-y-bont ar Wysg', 1723:'Tonyrefail Gorllewin 2', 18:'Townhill 1', 32:'Townhill 3', 362:'Townhill 4', 58:'Townhill 6', 1119:'Trawsfynydd', 1406:'Tre Caerfyrddin De 2', 369:'Tre Caerfyrddin Gogledd 2', 743:'Tre Caerfyrddin Gogledd 3', 971:'Tre Caerfyrddin Gorllewin 1', 1464:'Tre Caerfyrddin Gorllewin 2', 1256:'Tre Caerfyrddin Gorllewin 3', 1110:'Trecelyn 3', 798:'Trecelyn 4', 35:'Tredegar Canol a Gorllewin 2', 217:'Tredegar Canol a Gorllewin 3', 203:'Tredegar Canol a Gorllewin 4', 246:'Tredegar Newydd 2', 1417:'Tref-y-clawdd 2', 
        641:'Treflan Talacharn 1', 1114:'Trefnant', 1088:'Treforys 1', 686:'Treforys 4', 309:'Treforys 6', 154:'Treforys 7', 205:'Treforys 9', 1722:'Tregwyr 1', 1525:'Tregwyr 2', 1147:'Tregwyr 3', 1185:'Treletert', 1594:'Trelái 9', 543:'Trimsaran 2', 965:'Trowbridge 2', 283:'Two Locks 1', 1814:'Two Locks 2', 1060:'Two Locks 3', 8:'Twyn Carno 1', 473:'Twyn Carno 2', 804:'Ty Du 3', 
        1127:'Ty Du 6', 868:'Uplands 2', 1541:'Uplands 3', 1790:'Uplands 5', 1298:'Uplands 6', 914:'Uwch Conwy', 317:'Victoria 1', 161:'Victoria 2', 178:'Victoria 3', 112:'Victoria 4', 1893:'West Cross 1', 1750:'West Cross 2', 591:'West Cross 4', 656:'West End', 986:'Wyesham', 861:'Y Cocyd 1', 229:'Y Cocyd 2', 775:'Y Cocyd 3', 1163:'Y Cocyd 4', 1539:'Y Cocyd 5',
        207:'Y Cocyd 8', 948:'Y Dref 4', 284:'Y Drenewydd De', 588:'Y Gaer 6', 1774:'Y Graig (Casnewydd) 2', 1261:'Y Maerdy 2', 1592:'Y Mynydd Bychan 5', 479:'Y Pîl 3', 771:'Y Pîl 4', 1093:'Y Trallwng Gungrog 2', 887:'Ynys-ddu 1', 1083:'Ynys-ddu 2', 431:'Ynysawdre 1', 1381:'Ynysawdre 2', 1793:'Yr Eglwys Newydd a Thongwynlais 10', 1805:'Yr Orsedd 1', 1415:'Yr Orsedd 2', 1264:'Ysgir', 1719:'Ystumllwynarth 1', 1760:'Ystumllwynarth 2', 1826:'Ystumllwynarth 3'}
        WIMD_2019_Rank = st.selectbox(
            "What is the name of the area you live in?", options=list(choices.keys()), format_func=format_func)
        WIMD_2019_Decile = decile_ranking(WIMD_2019_Rank)
        WIMD_2019_Quintile = quintile_ranking(WIMD_2019_Rank)
        WIMD_2019_Quartile = quartile_ranking(WIMD_2019_Rank)
        Study_Year = st.selectbox(
            'What is your current year of study?', options=['Year 3','Year 4', 'Year 5', 'Year 6'])
        Gender = st.selectbox(
            'Select your gender',['Boy','Girl','Prefer not to say'])
        breakfast = st.multiselect(
            'Select all the types of foods you have for breakfast yesterday (if applicable)',['Bread','Fruits','Healthy cereal','Sugary cereal','Yogurt','Cooked Breakfast','Nothing','Snack'])
        breakfast_str = ';'.join(breakfast)
        Fruitveg_ytd = st.slider(
            'How many pieces of fruit or vegetable did you eat yesterday?', 0, 2, value=2)
        Brush_teeth_ytd = st.slider('How many times did you brush your teeth yesterday?', 0, 3, value=2)
        Sleeptime_ytd = st.slider(
            "What time did u sleep last night?", value=time(23, 0), step=timedelta(minutes=30) ).strftime("%Y-%m-%d %H:%M:%S")
        Awaketime_today = st.slider(
            "What time did u wake up today?", value=time(7, 0), step=timedelta(minutes=30) ).strftime("%Y-%m-%d %H:%M:%S")
        Going_school = st.selectbox('Are you currently going to school during lockdown?',['No','Yes, most days of the week','Yes, sometimes'])
        Other_children_inhouse = st.selectbox(
            'Do you live with other children in the house?', options=['Yes','No'])
        Number_people_household = st.select_slider(
            'How many people are you living with?', options=[1,2,3,4,5,6], value = 2)
        Easywalk_somewhere = st.selectbox(
            'From your home, is it easy for you to walk to somewhere that you can play?', options=['Yes','No'])
        Easywalk_topark = st.selectbox(
            'From your home, is it easy for you to walk to a park?', options=['Yes','No'])
        Garden = st.selectbox(
            'Do you have a garden at the house?', options=['Yes','No'])
        Outdoorplay_freq = st.selectbox(
            'How often did you usually play outside?', options=["I don't play", "Hardly ever", "A few days each week", "Most days"])
        Play_inall_places = st.selectbox(
            'Can you play in all of the places you would like to?', options=["I can hardly play in any of the places I would like to", "I can only play in a few places I would like to", "I can play in some of the places I would like to", "I can play in all the places I would like to"])
        Enoughtime_toplay =  st.selectbox(
            'Do you usually have enough time to play?', options=["No, I need a lot more", "No, I would like to have a bit more", "Yes, it's just about enough", "Yes, I have loads"])
        play_places = st.multiselect('Select all the types of places that you usually play at',['In the house','In the woods','In grass area','In the field','In the bushes','Near the water','On bike or at park','In the garden','On the street','At the playground'])
        play_places_str = ';'.join(play_places)
        homespace_relax = st.selectbox('Do you feel that you have a home space to relax in?', options=['Yes','Sometimes but not all the time','No'])
        Keep_in_touch_family_outside_household = st.selectbox(
            'Are you able to keep in touch with family members outside of your household?', options=['Yes','No'])
        Keep_in_touch_friends = st.selectbox(
            'Are you able to keep in touch with your friends?', options=['Yes','No'])
        contact_methods = st.multiselect(
            'Select all the methods you use to keep in touch with your family and friends',['Visit','Phone calls','Gaming','Social media'])
        contact_methods_str = ';'.join(contact_methods)
        st.write('---')
        st.write(
            "On a scale of 0 to 10 (0 being not very safe and 10 being very safe):")
        Safety_toplay_scale = st.slider(
            "How safe do you feel playing in your area?",0,10,5)
        st.write(
            "On a scale of 0 to 10 (0 being very unhappy and 10 being very happy):")
        School_scale = st.slider("How do you feel about your school?",0,10,5)
        Life_scale = st.slider("How do you feel about your life?",0,10,5)
        Looks_scale = st.slider("How do you feel about your looks?",0,10,5)
        Health_scale = st.slider("How do you feel about your health?",0,10,5)
        Family_scale = st.slider("How do you feel about your family?",0,10,5)
        Friends_scale = st.slider("How do you feel about your friends?",0,10,5)
        st.write("---")
        st.write("Do you agree or disagree that:")
        Lots_of_things_good_at = st.select_slider(
            'There are lots of things that you are good at', options=['Strongly disagree', 'Disagree', "Don't agree or disagree", 'Agree', 'Strongly agree'], value = 'Agree')
        Doingwell_schoolwork = st.select_slider(
            'You are doing well at school work', options=['Strongly disagree', 'Disagree', "Don't agree or disagree", 'Agree', 'Strongly agree'], value = 'Agree')
        Lots_of_choices_important = st.select_slider(
            'You have choice of things that are important to you', options=['Strongly disagree', 'Disagree', "Don't agree or disagree", 'Agree', 'Strongly agree'], value = 'Agree')
        Feel_partof_community = st.select_slider(
            'You are part of your school community', options=['Strongly disagree', 'Disagree', "Don't agree or disagree", 'Agree', 'Strongly agree'], value = 'Agree')
        st.write('---')
        st.write("In the last 7 days:")
        Concentrate_in_week = st.select_slider(
            'How many days did you feel you could concentrate on?', options=["O days", "1-2 days", "3-4 days", "5-6 days", "7 days"], value="3-4 days")
        Sugarsnack_in_week = st.select_slider(
            'How many days did you eat a sugar snack?', options=["O days", "1-2 days", "3-4 days", "5-6 days", "7 days"], value="3-4 days")
        Sports_in_week = st.select_slider(
            'How many days did you do sports or exercise for at least 1 hour in total?', options=["O days", "1-2 days", "3-4 days", "5-6 days", "7 days"], value="3-4 days")
        Tired_in_week= st.select_slider(
            'How many days did you feel tired?', options=["O days", "1-2 days", "3-4 days", "5-6 days", "7 days"], value="3-4 days")
        Internet_in_week = st.select_slider(
            'How many days did you watch TV/play online games/use the internet etc. for 2 or more hours a day (in total)?', options=["O days", "1-2 days", "3-4 days", "5-6 days", "7 days"], value="3-4 days")
        Softdrink_in_week = st.select_slider(
            'How many days did you drink soft drinks?', options=["O days", "1-2 days", "3-4 days", "5-6 days", "7 days"], value="3-4 days")
        Takeawayfood_in_week = st.select_slider(
            'How many days did you have takeaway food?', options=["O days", "1-2 days", "3-4 days", "5-6 days", "7 days"], value="3-4 days")
        if st.button('Predict your wellbeing status'):
            if not os.path.isdir('Saved_Models/') or not os.listdir('Saved_Models/'):
                st.error("No model has been saved yet. Have u skipped Final Model Training step?")
            else:
                Timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                column_names = ['Timestamp', 'Read_Info_Sheet', 'School_Health_Records', 'Study_Year', 'Gender', 'Breakfast_ytd', 'Fruitveg_ytd', 'Brush_teeth_ytd', 'Sleeptime_ytd', 'Awaketime_today', 'Safety_toplay_scale', 'Doingwell_schoolwork', 'Lots_of_choices_important', 'Lots_of_things_good_at', 'Health_scale', 'School_scale', 'Family_scale', 'Friends_scale', 'Looks_scale', 'Life_scale', 'WIMD_2019_Rank', 'WIMD_2019_Decile', 'WIMD_2019_Quintile', 'WIMD_2019_Quartile', 'Going_school', 'Other_children_inhouse', 'Number_people_household', 'Easywalk_topark','Easywalk_somewhere', 'Garden', 'Outdoorplay_freq', 'Enoughtime_toplay', 'Type_of_play_places', 'Play_inall_places', 'Homespace_relax', 'Feel_partof_community', 'Keep_in_touch_family_outside_household', 'Keep_in_touch_friends', 'Method_of_keepintouch', 'Sports_in_week', 'Internet_in_week', 'Tired_in_week', 'Concentrate_in_week', 'Softdrink_in_week', 'Sugarsnack_in_week', 'Takeawayfood_in_week', 'Birth_Date']
                model = joblib.load('Saved_Models/FinalModel.pkl')
                pipeline = joblib.load('Saved_Models/Preprocessing_Pipeline.pkl')
                inputs = pd.DataFrame(np.array([Timestamp,Read_Info_Sheet,School_Health_Records,Study_Year,Gender,breakfast_str,Fruitveg_ytd,Brush_teeth_ytd,Sleeptime_ytd,Awaketime_today,Safety_toplay_scale,Doingwell_schoolwork,Lots_of_choices_important,Lots_of_things_good_at,Health_scale,School_scale,Family_scale,Friends_scale,Looks_scale,Life_scale,WIMD_2019_Rank,WIMD_2019_Decile,WIMD_2019_Quintile,WIMD_2019_Quartile,Going_school,Other_children_inhouse,Number_people_household,Easywalk_topark,Easywalk_somewhere,Garden,Outdoorplay_freq,Enoughtime_toplay,play_places_str,Play_inall_places,homespace_relax,Feel_partof_community,Keep_in_touch_family_outside_household,Keep_in_touch_friends,contact_methods_str,Sports_in_week,Internet_in_week,Tired_in_week,Concentrate_in_week,Softdrink_in_week,Sugarsnack_in_week,Takeawayfood_in_week,Birth_date]),index=column_names).T
                inputs['Timestamp'] = inputs['Timestamp'].astype('object')
                inputs_transformed = pipeline.transform(inputs)
                predicted_value = model.predict(inputs_transformed)
                dict_class = {0:'normal',1:'emotional_significant',2:'behaviour_significant',3:'emotional_and_behaviour_significant'}
                st.write("Your wellbeing is predicted to be ",str(dict_class[predicted_value[0]]))


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

if __name__=='__main__':
    main()