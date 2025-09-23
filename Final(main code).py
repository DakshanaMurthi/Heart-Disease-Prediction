###
"""import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('heart(final_data).csv')
print(df.head())

# Create a pie chart for HeartDisease distribution
labels = ['Yes', 'No']
values = df['HeartDisease'].value_counts().values
plt.pie(values, labels=labels, autopct='%1.0f%%')
plt.title('Heart Disease Distribution')
plt.show()

# Bar chart of Chest Pain Type vs Heart Disease
pd.crosstab(df.ChestPainType, df.HeartDisease).plot(kind="bar", figsize=(8, 6))
plt.title('Heart Disease Frequency by Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(np.arange(4), ('typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'), rotation=0)
plt.ylabel('Frequency')
plt.show()

# Age stats
print('Min age:', df['Age'].min())
print('Max age:', df['Age'].max())

# Age distribution based on Heart Disease
sns.histplot(df[df['HeartDisease'] == 1]['Age'], label='Have heart disease', kde=True, color='red')
sns.histplot(df[df['HeartDisease'] == 0]['Age'], label='Do not have heart disease', kde=True, color='green')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution Based on Heart Disease')
plt.legend()
plt.show()

# Age ranges based on Heart Disease presence
print('Min age (No heart disease):', df[df['HeartDisease'] == 0]['Age'].min())
print('Max age (Heart disease present):', df[df['HeartDisease'] == 1]['Age'].max())

# Label encoding
le = LabelEncoder()
df['Age'] = le.fit_transform(df['Age'])
df['Sex'] = le.fit_transform(df['Sex'])
df['ChestPainType'] = le.fit_transform(df['ChestPainType'])
df['FastingBS_Num'] = le.fit_transform(df['FastingBS_Num'])
df['RestingECG'] = le.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])
df['HeartDisease'] = le.fit_transform(df['HeartDisease'])

# Splitting data
x = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=55)

# Train Gaussian Naive Bayes
NB = GaussianNB()
NB.fit(x_train, y_train)

# Prediction & accuracy
y_pred = NB.predict(x_test)
print('Naive Bayes ACCURACY is', accuracy_score(y_test, y_pred))

# Function to evaluate patient
def evaluate_patient(patient_data):
    prediction = NB.predict([patient_data])[0]
    bp = patient_data[3]
    chol = patient_data[4]
    sex = patient_data[1]
    
    if prediction == 1:
        return "Prediction: The Patient **has heart disease**. Please consult a doctor."
    else:
        advice = "Prediction: The Patient **does NOT have heart disease**."
        if chol > 200:
            if bp > 120:
                print(" But blood pressure and cholesterol are high. Please consult a doctor.")
            elif bp < 120:
                print(" Blood pressure is low, but cholesterol is high. Please consult a doctor.")
            elif bp == 120:
                print(" Blood pressure is normal but cholesterol is high. Please consult a doctor.")
        elif bp > 120:
              print(" Blood pressure is high. Please consult a doctor.")
        else:
            print(" All readings normal.")
        return 0

# Predict for new patient
# Format: [Age, Sex, ChestPainType, RestingBP, Cholesterol, RestingECG, MaxHR, ExerciseAngina, FastingBS_Num]
patient_data = [30, 0, 2, 130, 280, 1, 150, 0, 90]  # Example encoded values
print(evaluate_patient(patient_data))

# Optional: Save model using Pickle
import pickle
pickle.dump(NB, open('model.pkl', 'wb'))
"""
###
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('heart(final_data).csv')    #to read the file
print(df.head())

# Create a plot to display the percentage of the positive and negative heart disease
labels = ['yes', 'No']
values = df['HeartDisease'].value_counts().values

plt.pie(values, labels=labels, autopct='%1.0f%%')
plt.title('HeartDisease')
plt.show()

# Display chest pain types based on the Heart Disease
pd.crosstab(df.ChestPainType,df.HeartDisease).plot(kind = "bar", figsize = (8, 6))
plt.title('Heart Disease Frequency According to Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(np.arange(4), ('typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'), rotation = 0)
plt.ylabel('Frequency')
plt.show()


# Get min, max and average of the age
print('Min age: ', min(df['Age']))
print('Max age: ', max(df['Age']))



# Display age distribution based on heart disease
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('heart(final_data).csv')

# Display age distribution based on heart disease
sns.histplot(df[df['HeartDisease'] == 'PRESENCE']['Age'], label='Have heart disease', kde=True, color='red')
sns.histplot(df[df['HeartDisease'] == 'ABSENCE']['Age'], label = 'Do not have heart disease', kde=True, color='green') # Corrected to 0 for 'Do not have heart disease'
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution based on Heart Disease')
plt.legend() # Added legend to show labels
plt.show()

# Get min, max and average of the age of the people do not have heart diseas
print('Min age of people who do not have heart disease: ', min(df[df['HeartDisease'] == 'ABSENCE']['Age']))
print('Max age of people who do not have heart disease: ', max(df[df['HeartDisease'] == 'PRESENCE']['Age']))



# Apply label encoding after data analysis and visualization
le=LabelEncoder()
df['Age'] = le.fit_transform(df['Age'])
df['Sex'] = le.fit_transform(df['Sex'])
df['ChestPainType'] = le.fit_transform(df['ChestPainType'])
df['FastingBS_Num'] = le.fit_transform(df['FastingBS_Num'])
df['RestingECG'] = le.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])
df['HeartDisease'] = le.fit_transform(df['HeartDisease']) # Encode DiseaseOutcome last








NB = GaussianNB()

x=df.drop(columns=['HeartDisease'])
y=df['HeartDisease']      #to create the variable
print(x)
print(y)

# First split: Train (70%) + Temp (30%)
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.30, random_state=42
)

# Second split: Temp (30%) → Test (20%) + Eval (10%)
# Since temp is 30%, we need to take 2/3 for test and 1/3 for eval to match 20% and 10% overall
x_test, x_eval, y_test, y_eval = train_test_split(
    x_temp, y_temp, test_size=(1/3), random_state=42
)

print("Train size:", len(x_train))
print("Test size:", len(x_test))
print("Evaluation size:", len(x_eval))


NB.fit(x_train, y_train)  #train the data

y_pred=NB.predict(x_test)
print('Naive Bayes ACCURACY is', accuracy_score(y_test,y_pred))


##from sklearn.neighbors import KNeighborsClassifier
##knn = KNeighborsClassifier()
##knn.fit(x_train, y_train)
##y_pred_knn=knn.predict(x_test)
##print('KNN ACCURACY is', accuracy_score(y_test,y_pred_knn))





# Test the model with new patient data
# Example input: Age=30, Sex=FEMALE (0), ChestPainType=Non-Anginal Pain (2), RestingBP=120, Cholesterol=180, RestingECG=Normal (1), MaxHR=150, ExerciseAngina=NO (0), FastingBS_Num=90
patient_data = [[40, 0, 2, 140, 289, 1, 172, 0, 108]]
testPrediction = NB.predict(patient_data)

# Extract values from the input data (Note: Applying manual rules after model prediction is generally not standard practice)
Sex = patient_data[0][1]
RestingBP = patient_data[0][3]
Cholesterol = patient_data[0][4]

if testPrediction[0] == 1:
    print(testPrediction, "The Patient Has Heart Disease, please consult the Doctor")

elif testPrediction[0] == 0:
    print(testPrediction, "The Patient Normal")
    if Sex == 0:
        if (RestingBP > 120 and Cholesterol > 200):
            print("your BP and Cholesterol is high please consult the Doctor")
        elif (RestingBP < 120 and Cholesterol > 200):
            print("your BP low and Cholesterol is high please consult the Doctor")
        elif (RestingBP == 120 and Cholesterol > 200):
            print("your BP is good but Cholesterol is high please consult the Doctor")
        elif (RestingBP > 120 and Cholesterol < 200):
            print("your BP high and Cholesterol is low please consult the Doctor")
        else:
            print("The Patient Normal")
    if Sex == 1:
        if (RestingBP > 120 and Cholesterol > 200):
            print("your BP and Cholesterol is high please consult the Doctor")
        elif (RestingBP < 120 and Cholesterol > 200):
            print("your BP low and Cholesterol is high please consult the Doctor")
        elif (RestingBP == 120 and Cholesterol > 200):
            print("your BP is good but Cholesterol is high please consult the Doctor")
        elif (RestingBP > 120 and Cholesterol < 200):
            print("your BP high and Cholesterol is low please consult the Doctor")
        else:
            print("The Patient Normal")
else:
    print(testPrediction, "The Patient Normal")

import pickle
pickle.dump(NB,open('model.pkl','wb'))

"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler # Import StandardScaler for KNN
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- Existing Data Loading and Preprocessing ---
df = pd.read_csv('heart(final_data).csv')

# Create separate LabelEncoder instances for each categorical column
le_sex = LabelEncoder()
le_chestpaintype = LabelEncoder()
le_restingecg = LabelEncoder()
le_exerciseangina = LabelEncoder()
le_heartdisease = LabelEncoder()

# Apply encoding using their respective encoders
df['Sex'] = le_sex.fit_transform(df['Sex'])
df['ChestPainType'] = le_chestpaintype.fit_transform(df['ChestPainType'])
df['RestingECG'] = le_restingecg.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = le_exerciseangina.fit_transform(df['ExerciseAngina'])
df['HeartDisease'] = le_heartdisease.fit_transform(df['HeartDisease'])

# IMPORTANT: Store the numerical codes for 'PRESENCE' and 'ABSENCE'
PRESENCE_CODE = le_heartdisease.transform(['PRESENCE'])[0]
ABSENCE_CODE = le_heartdisease.transform(['ABSENCE'])[0]

print(df.head())

# --- Visualizations (unchanged, will run in the script) ---
# Create a plot to display the percentage of the positive and negative heart disease
labels = le_heartdisease.classes_
values = df['HeartDisease'].value_counts().values
plt.pie(values, labels=labels, autopct='%1.0f%%')
plt.title('Heart Disease Distribution')
plt.show()

# Display chest pain types based on the Heart Disease
plt.figure(figsize=(8, 6))
df_plot_cp = df.copy()
df_plot_cp['ChestPainType_Decoded'] = le_chestpaintype.inverse_transform(df_plot_cp['ChestPainType'])
df_plot_cp['HeartDisease_Decoded'] = le_heartdisease.inverse_transform(df_plot_cp['HeartDisease'])
pd.crosstab(df_plot_cp.ChestPainType_Decoded, df_plot_cp.HeartDisease_Decoded).plot(kind="bar", figsize=(8, 6))
plt.title('Heart Disease Frequency According to Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.show()

# Get min, max and average of the age
print('Min age: ', min(df['Age']))
print('Max age: ', max(df['Age']))

# Display age distribution based on heart disease
df_original_for_age_plot = pd.read_csv('heart(final_data).csv')
sns.histplot(df_original_for_age_plot[df_original_for_age_plot['HeartDisease'] == 'PRESENCE']['Age'], label='Have heart disease', kde=True, color='red')
sns.histplot(df_original_for_age_plot[df_original_for_age_plot['HeartDisease'] == 'ABSENCE']['Age'], label = 'Do not have heart disease', kde=True, color='green')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution based on Heart Disease')
plt.legend()
plt.show()

# Get min, max and average of the age of the people do not have heart disease
print('Min age of people who do not have heart disease: ', min(df_original_for_age_plot[df_original_for_age_plot['HeartDisease'] == 'ABSENCE']['Age']))
print('Max age of people who do not have heart disease: ', max(df_original_for_age_plot[df_original_for_age_plot['HeartDisease'] == 'PRESENCE']['Age']))


# --- Data Splitting ---
x=df.drop(columns=['HeartDisease'])
y=df['HeartDisease']
print("\nFeatures used for training (x.columns):", x.columns.tolist())

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.30, random_state=42)
x_test, x_eval, y_test, y_eval = train_test_split(x_temp, y_temp, test_size=(1/3), random_state=42)

print("Train size:", len(x_train))
print("Test size:", len(x_test))
print("Evaluation size:", len(x_eval))


# --- Train and Evaluate Naive Bayes ---
NB = GaussianNB()
NB.fit(x_train, y_train)
y_pred_nb = NB.predict(x_test)
print('\nNaive Bayes ACCURACY is', accuracy_score(y_test,y_pred_nb))


# --- NEW CODE FOR KNN ---
from sklearn.neighbors import KNeighborsClassifier

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_eval_scaled = scaler.transform(x_eval)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_scaled, y_train)
y_pred_knn = knn.predict(x_test_scaled)
print('KNN ACCURACY is', accuracy_score(y_test,y_pred_knn))
# --- End NEW CODE FOR KNN ---


# --- Updated evaluate_patient function to handle both models ---
def evaluate_patient_corrected(model, scaler, model_name,
                               age, sex_str, chest_pain_type_str, resting_bp, cholesterol,
                               resting_ecg_str, max_hr, exercise_angina_str, fasting_bs_num_val):
    
    # Use the fitted LabelEncoders to transform string inputs to numerical
    try:
        encoded_sex = le_sex.transform([sex_str])[0]
        encoded_chest_pain_type = le_chestpaintype.transform([chest_pain_type_str])[0]
        encoded_resting_ecg = le_restingecg.transform([resting_ecg_str])[0]
        encoded_exercise_angina = le_exerciseangina.transform([exercise_angina_str])[0]
    except ValueError as e:
        return f"Error encoding categorical feature: {e}. Check input values and ensure they match categories in original data."

    # Construct the patient data array in the EXACT ORDER of x.columns
    patient_features = [
        age,
        encoded_sex,
        encoded_chest_pain_type,
        resting_bp,
        cholesterol,
        encoded_resting_ecg,
        max_hr,
        encoded_exercise_angina,
        fasting_bs_num_val # Direct numerical value
    ]

    patient_data_array = np.array(patient_features).reshape(1, -1)
    
    # Apply scaling if the model expects scaled data (like KNN)
    if scaler:
        patient_data_array = scaler.transform(patient_data_array)

    prediction = model.predict(patient_data_array)[0]

    result_message = f"Prediction ({model_name}): "
    if prediction == PRESENCE_CODE:
        result_message += f"The Patient **has heart disease** ({le_heartdisease.inverse_transform([prediction])[0]}). Please consult a doctor."
    else:
        result_message += f"The Patient **does NOT have heart disease** ({le_heartdisease.inverse_transform([prediction])[0]})."
        
        # Additional advice logic
        if cholesterol > 200:
            if resting_bp > 120:
                result_message += "\nBut blood pressure and cholesterol are high. Please consult a doctor."
            elif resting_bp < 120:
                result_message += "\nBlood pressure is low, but cholesterol is high. Please consult a doctor."
            elif resting_bp == 120:
                result_message += "\nBlood pressure is normal but cholesterol is high. Please consult a doctor."
        elif resting_bp > 120:
              result_message += "\nBlood pressure is high. Please consult a doctor."
        else:
            result_message += "\nAll readings normal."
            
    return result_message

# --- Test with the data from your image using both models ---
print("\n--- Testing with provided image data using corrected function ---")

# Patient 1 data
patient1_args = {
    'age': 40, 'sex_str': 'MALE', 'chest_pain_type_str': 'Atypical Angina',
    'resting_bp': 140, 'cholesterol': 289, 'resting_ecg_str': 'Normal',
    'max_hr': 172, 'exercise_angina_str': 'NO', 'fasting_bs_num_val': 108
}
# Patient 2 data
patient2_args = {
    'age': 49, 'sex_str': 'FEMALE', 'chest_pain_type_str': 'Non-Anginal Pain',
    'resting_bp': 160, 'cholesterol': 180, 'resting_ecg_str': 'Normal',
    'max_hr': 156, 'exercise_angina_str': 'NO', 'fasting_bs_num_val': 98
}

# --- NEW PATIENT 3 DATA ---
patient3_args = {
    'age': 37, 'sex_str': 'MALE', 'chest_pain_type_str': 'Asymptomatic', # Assuming 'Asymptom' maps to 'Asymptomatic'
    'resting_bp': 140, 'cholesterol': 207, 'resting_ecg_str': 'Normal',
    'max_hr': 130, 'exercise_angina_str': 'YES', 'fasting_bs_num_val': 92
}

print("\n--- Naive Bayes Predictions ---")
print("Patient 1 Result (Expected: ABSENCE):", evaluate_patient_corrected(NB, None, "Naive Bayes", **patient1_args))
print("Patient 2 Result (Expected: PRESENCE):", evaluate_patient_corrected(NB, None, "Naive Bayes", **patient2_args))
print("Patient 3 Result (Expected: PRESENCE):", evaluate_patient_corrected(NB, None, "Naive Bayes", **patient3_args))

print("\n--- KNN Predictions ---")
print("Patient 1 Result (Expected: ABSENCE):", evaluate_patient_corrected(knn, scaler, "KNN", **patient1_args))
print("Patient 2 Result (Expected: PRESENCE):", evaluate_patient_corrected(knn, scaler, "KNN", **patient2_args))
print("Patient 3 Result (Expected: PRESENCE):", evaluate_patient_corrected(knn, scaler, "KNN", **patient3_args))


# Optional: Save models using Pickle
import pickle
pickle.dump(NB,open('naive_bayes_model.pkl','wb'))
pickle.dump(knn,open('knn_model.pkl','wb'))
pickle.dump(scaler,open('scaler.pkl','wb')) # Save the scaler too if you use KNN in deployment
