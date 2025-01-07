import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('Model')

class_labels = {
    0:'Average',
    1:'Excellent',
    2:'Good',
    3:'Poor',
    4:'Very Good',
    5:'Very Poor',
}

st.title("Acidity Calibre Check on Wine")
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.1)
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.1)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001, format="%.4f")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0, step=1)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0, step=1)
density = st.number_input("Density", min_value=0.0, step=0.0001, format="%.5f")
pH = st.number_input("pH", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

# Store the user inputs in a list
features = [
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
    pH, sulphates, alcohol
]

# Add a submit button
if st.button("Submit"):
    input_data = np.array(features).reshape(1, -1)
    quality = model.predict(input_data)
    st.write(f"The predicted wine quality is: {class_labels[quality[0]]}")
