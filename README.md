# Heart Disease Prediction Using Naive Bayes Classifier

## Project Overview
This project implements a heart disease prediction system using a Naive Bayes classifier. The system leverages key clinical and demographic health attributes from patient data to predict the presence or absence of heart disease with an accuracy of approximately 83%. It also provides alerts for abnormal cholesterol and blood pressure values to aid preventive healthcare.

## System Description
- The dataset consists of 918 patient records with features such as age, gender, chest pain type, blood pressure, cholesterol, ECG results, exercise-induced angina, fasting blood sugar, and more.
- Data preprocessing includes label encoding of categorical variables, handling missing values, and normalization/scaling of numeric values.
- The Naive Bayes classifier is trained on 70% of the data, with 20% reserved for testing and 10% for validation.
- The application uses Streamlit for a user-friendly interface where users input patient details and receive real-time predictions.
- MongoDB is used as a backend database to store patient input and prediction results.
- The system highlights key indicators with alerts for high blood pressure and cholesterol.

## Technologies Used
- Python (pandas, numpy, scikit-learn)
- Naive Bayes Classifier (GaussianNB)
- Streamlit for UI
- MongoDB for data storage
- Pickle for model serialization

## How to Use
1. Clone the repository.
2. Install required Python packages:
## pip install -r requirements.txt:
3. Run the Streamlit app:
## streamlit run app.py
4. Input patient information in the provided form and select the model for prediction.
5. View the prediction results and alerts on the interface.

## Project Structure
- `app.py` - Streamlit application script
- `model_training.py` - Script for training Naive Bayes and KNN models
- `naivebayesmodel.pkl` - Serialized Naive Bayes model
- `knnmodel.pkl` - Serialized KNN model
- `scaler.pkl` - Scaler object for KNN
- `data/` - Folder containing dataset CSV files
- `README.md` - This file

## References
1. Singh et al., Comparative analysis of heart disease prediction using multiple ML models, Nature Scientific Reports, 2025.
2. Ahmed et al., Optimizing heart disease diagnosis with advanced machine learning, BMC Cardiovascular Disorders, 2025.
3. Sharma et al., Advancements in Heart Disease Prediction, arXiv preprint, 2024.

