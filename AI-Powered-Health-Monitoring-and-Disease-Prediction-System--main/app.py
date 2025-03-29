from flask import Flask, render_template, url_for, request
import sqlite3
import numpy as np
import pandas as pd
import pickle
import requests
import warnings
import google.generativeai as genai
from fuzzywuzzy import process


connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()
command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

command = """CREATE TABLE IF NOT EXISTS sessions(name TEXT, password TEXT, timestamp TEXT)"""
cursor.execute(command)


warnings.filterwarnings('ignore')
model = pickle.load(open('LL.pkl', 'rb'))

# Load Gemini AI model
genai.configure(api_key='AIzaSyBL2LtSwzQZy4VzdRzlm5YtRW-R7YIcjkM')
gemini_model = genai.GenerativeModel('gemini-1.5-pro')
chat = gemini_model.start_chat(history=[])

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------

# Loading the datasets from Kaggle website

sym_des = pd.read_csv("kaggle_dataset/symptoms_df.csv")
precautions = pd.read_csv("kaggle_dataset/precautions_df.csv")
workout = pd.read_csv("kaggle_dataset/workout_df.csv")
description = pd.read_csv("kaggle_dataset/description.csv")
medications = pd.read_csv('kaggle_dataset/medications.csv')
diets = pd.read_csv("kaggle_dataset/diets.csv")

Rf = pickle.load(open('model/RandomForest.pkl','rb'))


# Here we make a dictionary of symptoms and diseases and preprocess it

symptoms_list = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

symptoms_list_processed = {symptom.replace('_', ' ').lower(): value for symptom, value in symptoms_list.items()}

# Here we created a function (information) to extract information from all the datasets

def information(predicted_dis):
    disease_desciption = description[description['Disease'] == predicted_dis]['Description']
    disease_desciption = " ".join([w for w in disease_desciption])

    disease_precautions = precautions[precautions['Disease'] == predicted_dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    disease_precautions = [col for col in disease_precautions.values]

    disease_medications = medications[medications['Disease'] == predicted_dis]['Medication']
    disease_medications = [med for med in disease_medications.values]

    disease_diet = diets[diets['Disease'] == predicted_dis]['Diet']
    disease_diet = [die for die in disease_diet.values]

    disease_workout = workout[workout['disease'] == predicted_dis] ['workout']


    return disease_desciption, disease_precautions, disease_medications, disease_diet, disease_workout


# This is the function that passes the user input symptoms to our Model
def predicted_value(patient_symptoms):
    i_vector = np.zeros(len(symptoms_list_processed))
    for i in patient_symptoms:
        i_vector[symptoms_list_processed[i]] = 1
    return diseases_list[Rf.predict([i_vector])[0]]

# Function to correct the spellings of the symptom (if any)
def correct_spelling(symptom):
    closest_match, score = process.extractOne(symptom, symptoms_list_processed.keys())
    # If the similarity score is above a certain threshold, consider it a match
    if score >= 80:
        return closest_match
    else:
        return None

# gui_stuff------------------------------------------------------------------------------------

# connection = sqlite3.connect('user_data.db')
# cursor = connection.cursor()

# command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
# cursor.execute(command)

app = Flask(__name__)
chat_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('result.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/graph')
def graph():
    return render_template('graph.html')




@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT * FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchone()

        if result:
            
            from datetime import datetime
            now = datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
            cursor.execute("INSERT INTO sessions VALUES ('"+name+"', '"+password+"', '"+str(date_time)+"')")
            connection.commit()
           
            return render_template('result.html')  #,bp=bp)
        else:
            return render_template('index.html', msg='Sorry , Incorrect Credentials Provided,  Try Again')

    return render_template('index.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        query = "SELECT * FROM user WHERE mobile = '"+mobile+"'"
        cursor.execute(query)

        result = cursor.fetchone()
        if result:
            return render_template('index.html', msg='Phone number already exists')
        else:

            cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
            connection.commit()

            return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')




@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        name = request.form['Name']
        sm1 = request.form['system1']
        sm2 = request.form['system2']
        sm3 = request.form['system3']
        sm4 = request.form['system4']
        sm5 = request.form['system5']

        psymptoms = [sm1, sm2, sm3 ,sm4, sm5]
        symptoms = ''
        counts = 0
        for d in psymptoms:
            if d != 'no':
                symptoms += d+','
                counts += 1
        if counts > 1:
            symptoms = symptoms[:-1]
            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            patient_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            patient_symptoms = [symptom.strip("[]' ") for symptom in patient_symptoms]

            # Correct the spelling of symptoms
            corrected_symptoms = []
            for symptom in patient_symptoms:
                corrected_symptom = correct_spelling(symptom)
                if corrected_symptom:
                    corrected_symptoms.append(corrected_symptom)
                else:
                    message = f"Symptom '{symptom}' not found in the database."
                    return render_template('index.html', message=message)

            # Predict the disease using corrected symptoms
            predicted_disease = predicted_value(corrected_symptoms)


            gemini_response = chat.send_message(predicted_disease+" this is the disease detected throught the algorithm so you need to provide the {1) drug with dosage recommendatoin ,2) atleast 4 to 5 diet plan fr veg and non veg seperate seperately,3) write some do's and dont's ,4) if given disease is dangeoruos give doctors list near bangalore } (in exact html format)")
            recommendatoin = gemini_response.text
            recommendatoin = recommendatoin.replace("```html", "")
            recommendatoin = recommendatoin.replace("```", "")

            return render_template('result.html', name=name,recommendatoin=recommendatoin,msg3=predicted_disease)
        else:

        # msg1 = DecisionTree(psymptoms)
        # msg2 = NaiveBayes(psymptoms)
        # msg3 = randomforest(psymptoms)

            return render_template('result.html',msg="Provide Atleast Two Symptoms")


@app.route('/health', methods=['GET', 'POST'])

def health():
    if request.method == 'POST':

        name = request.form['name']
        Age = request.form['age']
        sex = request.form['sex']
        if sex==1:
            sex="MALE"
        else:
            sex="FEMALE"
        bp = request.form['bp']
        oxy = request.form['oxy']
        print(oxy)
        hb = request.form['heart']
        ecg = request.form['ecg']
        Temperature = request.form['Temperature']
        to_predict_list = np.array([[bp,oxy,hb,ecg,Temperature]])
        print(to_predict_list)
        prediction = model.predict(to_predict_list)
        output = prediction[0]
        print("Prediction is {}  :  ".format(output))
        print(output)
        
        # Check the output values and retrive the result with html tag based on the value
        
        if output == 1:
            result="Healthy   !!!!!!" 
            med=""
        if output == 2:
            result="Fever" 
            med="Diagnosis Drugs for Fever  \n  Paracetamol \n acetaminophen \n Tylenol  \n aspirin \n  Acephen  \n Ecpirin \n"
        if output == 3:
            result="Chest Pain" 
            med="Diagnosis Drugs for chest_pain \n Exercise ,\n Avoid carbonated beverages , \n Hydration ,\n Acupuncture or Acupressure, \n Herbal Remedies ,\n Diltiazem \n"
        if output == 4:
            result="Critical" 
            med="You are critical \n concern the doctor nearby"
            print("Patient  :  "+str(name)+ "\n Age  :  "+str(Age)+ "\n Gender  :  "+str(sex)+ "\n Status  :  "+str(result)+" \n  Medicine Provided  :  "+str(med)+ "LOCATION https://maps.app.goo.gl/ftDTGkrZBpgXdNGs9")


        gemini_response = chat.send_message(result+" this is the disease detected throught the algorithm so you need to provide the {1) drug with dosage recommendatoin ,2) atleast 4 to 5 diet plan fr veg and non veg seperate seperately,3) write some do's and dont's ,4) if given disease is dangeoruos give doctors list near bangalore } (in exact  html format)")
        recommendatoin = gemini_response.text
        recommendatoin = recommendatoin.replace("```html", "")
        recommendatoin = recommendatoin.replace("```", "")
        
        return render_template('health.html',recommendatoin=recommendatoin, hb=hb,bp=bp,ecg=ecg,temp=Temperature,oxy=oxy, result = result,out=output,name =name,med=med )

   
    
    return render_template('health.html')

@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':
        user_input = request.form['query']
        # Get response from Gemini AI model
        gemini_response = chat.send_message(user_input)
        data = gemini_response.text
        chat_history.append([user_input, data])

        return render_template('chatbot.html', chat_history=chat_history)
    return render_template('chatbot.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
