import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Enhanced Laptop Price Predictor")

# Define available screen resolutions and their aspect ratios
RESOLUTIONS = {
    '1920x1080': (1920, 1080),
    '1366x768': (1366, 768),
    '1600x900': (1600, 900),
    '3840x2160': (3840, 2160),
    '3200x1800': (3200, 1800),
    '2880x1800': (2880, 1800),
    '2560x1600': (2560, 1600),
    '2560x1440': (2560, 1440),
    '2304x1440': (2304, 1440),
}

# Brand
company = st.selectbox('Select Brand', df['Company'].unique())

# Laptop Type
type_name = st.selectbox('Select Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Enter Weight of the Laptop (in kg)', min_value=0.5, max_value=5.0, step=0.1)

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS Display
ips = st.selectbox('IPS Display', ['No', 'Yes'])

# Screen Size
screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 15.6)

# Screen Resolution
resolution = st.selectbox('Select Screen Resolution', list(RESOLUTIONS.keys()))

# CPU
cpu = st.selectbox('Select CPU Brand', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('Select GPU Brand', df['Gpu brand'].unique())

# Operating System
os = st.selectbox('Select Operating System', df['os'].unique())

# Predict button
if st.button('Predict Price'):
    # Convert categorical inputs to numeric values
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Extract screen resolution details and calculate PPI
    X_res, Y_res = RESOLUTIONS[resolution]
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Prepare the input query for prediction
    query = np.array([company, type_name, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, -1)

    # Predict and display the price
    try:
        predicted_price = np.exp(pipe.predict(query)[0])
        st.success(f"The predicted price of this laptop configuration is: ${predicted_price:,.2f}")
        st.info(f"Details: {company} {type_name}, {ram}GB RAM, {weight}kg, "
                f"{'Touchscreen' if touchscreen else 'Non-Touchscreen'}, {'IPS' if ips else 'Non-IPS'}, "
                f"{screen_size}\" {resolution} ({ppi:.1f} PPI), {cpu}, {hdd}GB HDD, {ssd}GB SSD, {gpu}, {os}.")
    except Exception as e:
        st.error(f"Error predicting the price: {e}")
