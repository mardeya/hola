import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random 
from PIL import Image
import Doctors_dashboard
import Patient_dashboard
import Data_analysts
import altair as alt
import plotly.express as px

st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'> 🫀 Heart Attack Risk Prediction Dashboard 🫀</h1>",
    unsafe_allow_html=True
)
with st.sidebar:
    st.title("Menu")
    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

profile = st.sidebar.selectbox("Profile",["Patient","Doctor", "Data Analyst"])
if profile == "Doctor":
    Doctors_dashboard.show_dashboard()
elif profile =="Patient":
    Patient_dashboard.show_dashboard()
else: 
    Data_analysts.show_dashboard(selected_color_theme)


