import pickle
import streamlit as st
import numpy as np  
import pandas as pd
from sklearn.metrics import accuracy_score
import os

st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")

# Custom CSS for modern transparent pink & blue design
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to right, rgba(255, 192, 203, 0.7), rgba(173, 216, 230, 0.7)), 
                        url('https://www.transparenttextures.com/patterns/inspiration-geometry.png');
            color: white;
        }
        .stButton>button {
            display: block;
            margin: 20px auto;
            background: rgba(255, 105, 180, 0.8);
            color: white;
            font-size: 110%;
            padding: 12px 24px;
            border-radius: 12px;
            transition: all 0.3s ease-in-out;
            border: none;
            cursor: pointer;
            font-weight: bold;
            box-shadow: 3px 3px 15px rgba(255, 20, 147, 0.5);
        }
        .stButton>button:hover {
            background: rgba(255, 20, 147, 0.9);
            transform: scale(1.08);
        }
        .result-container {
            background: rgba(173, 216, 230, 0.85);
            color: black;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            margin-top: 15px;
            box-shadow: 3px 3px 15px rgba(0, 0, 0, 0.3);
        }
        input {
            border-radius: 8px;
            padding: 10px;
            border: none;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            font-size: 16px;
        }
        input::placeholder {
            color: rgba(255, 255, 255, 0.7);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model
diabetes_model_path = "diabites_model.sav"
try:
    with open(diabetes_model_path, 'rb') as model_file:
        diabetes_model = pickle.load(model_file)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

st.title('💖 Diabetes Prediction using ML')

# Layout for input fields
col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.text_input('Number of Pregnancies', value="0")

with col2:
    Glucose = st.text_input('Glucose Level', value="0")

with col3:
    BloodPressure = st.text_input('Blood Pressure value', value="0")

with col1:
    SkinThickness = st.text_input('Skin Thickness value', value="0")

with col2:
    Insulin = st.text_input('Insulin Level', value="0")

with col3:
    BMI = st.text_input('BMI value', value="0")

with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', value="0")

with col2:
    Age = st.text_input('Age of the Person', value="0")

# Check Model Accuracy
if st.button('Show Model Accuracy'):
    try:
        test_data = pd.read_csv(r"C:\workshop3\diabetes.csv")
        x_test = test_data.drop(columns=["Outcome"])
        y_test = test_data["Outcome"]
        y_pred = diabetes_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model Accuracy: {accuracy * 100:.2f}%")
    except Exception as e:
        st.error(f"Error calculating accuracy: {e}")

# Prediction Button
if st.button('Diabetes Test Result'):
    try:
        # Convert all inputs to float
        user_input = np.array([
            float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
            float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)
        ]).reshape(1, -1)  # Ensure it's a 2D array

        # Make prediction
        diab_prediction = diabetes_model.predict(user_input)

        # Display result
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

        st.markdown(f"<div class='result-container'>{diab_diagnosis}</div>", unsafe_allow_html=True)

    except ValueError:
        st.error("Please enter valid numerical values for all fields.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
