import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random 
from PIL import Image
import joblib
import numpy
import shap
from streamlit_shap import st_shap
import dalex as dx
from sklearn.impute import KNNImputer
from lime.lime_tabular import LimeTabularExplainer

selected_variables = [
    'Age',
    'Gender', 
    'Diabetes',
    'Family History',
    'Smoking',
    'Obesity',
    'Alcohol Consumption',  
    'Exercise Hours Per Week',
    'Diet',
    'Previous Heart Problems',
    'Medication Use',
    'Stress Level',
    'Sedentary Hours Per Day',
    'Sleep Hours Per Day'    
]

modelo_cargado = joblib.load('Modelo/modelo_rf_patients.joblib')

def show_dashboard():
    st.header("What is your risk of heart attack?")
    st.subheader("Please complete the following fields:")
    pred = []
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        gender = st.segmented_control("Sex", ["M", "F"])
        gender = 0 if gender == "M" else 1
    with col2:
        obesity = st.radio('Have you obesity?', ["No", "Yes"])
        obesity = 1 if obesity == "Yes" else 0
    with col3:
        val = st.radio('Previous Heart Problems', ["No", "Yes"])
        val = 1 if val == "Yes" else 0
    with col4:
        diabetes = st.radio("Do you have diabetes?",("No", "Yes"))
        diabetes = 1 if diabetes == "Yes" else 0
       

    colx, col5, col6, col7 = st.columns(4)
    with colx:
        alcohol = st.radio("Do you drink alcohol?",("No", "Yes"))
        alcohol = 1 if alcohol == "Yes" else 0
    with col5:
        medication = st.radio("Do you take any medication?",("No", "Yes"))
        medication = 1 if medication == "Yes" else 0
    with col6:
        smoking = st.radio("Do you smoke?",("No", "Yes"))
        smoking = 1 if smoking == "Yes" else 0
    with col7:
        family_history = st.radio("Do you have any family history of heart attacks?",("No", "Yes"))
        family_history = 1 if family_history == "Yes" else 0

    col8, col9, col10= st.columns(3)
    with col8:
        diet = st.radio("How is your diet?", ("Unhealthy", "Normal", "Healthy"))
        diet_map = {"Unhealthy": 0, "Normal": 1, "Healthy": 2}
        diet = diet_map.get(diet, -1)
    with col9:
        age = st.number_input(
            label="Introduce your age",
            min_value=18,
            max_value=90
        )
    with col10:
        sleep = st.number_input(
            label="How many hours do you sleep per day?",
            min_value=0,
            max_value=12
        )

    col11, col12, col13 = st.columns(3)
    with col11:
        sedentary = st.number_input(
            label="How many hours do you sit?",
            min_value=0,
            max_value=12
        )
    with col12: 
        exercise = st.number_input(
            label="How many hours a week do you exercise?",
            min_value=0,
            max_value=20
        )
    with col13:
        stress = st.number_input(
            label="From 1 to 10, which is your strees level?",
            min_value=1,
            max_value=10
        )

    _, col14, _= st.columns(3) 
    with col14:
        b = st.button("CALCULATE!")
    pred = [age,gender,diabetes,family_history,smoking,obesity, alcohol, exercise, diet,val, medication, stress, sedentary, sleep]

    if b:
        risk_percent = get_prediction(pred)  
        risk_class = get_predicted_class(pred)
        if risk_class == 1:
            decision_text = "HIGH RISK"
            emoji = "🚨"
        else:
            decision_text = "LOW RISK"
            emoji = "💪"

        st.markdown(f"###  You have {emoji} **{decision_text}** of having a heart attack.")        
        if risk_percent < 50:
            color = "green"
            image_path = "images/SemaforoVerde.png"
        else:
            color = "red"
            image_path = "images/SemaforoRojo.png"
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(image_path)
            rotated_image = image.rotate(90, expand=True)
            st.image(rotated_image, width=250)
        with col2:
            st.markdown(
                f"""
                <div style='
                    background-color: {color};
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                    color: black;
                    width: 200px;
                    margin: auto;
                '>
                    {risk_percent} %
                </div>
                """,
                unsafe_allow_html=True
            )
        
        explainer = shap.TreeExplainer(modelo_cargado)
        pred = preprocess_input(pred)
        # Explain the prediction
        X = explain_dashboard()
        explainer = dx.Explainer(modelo_cargado, X)
        local_exp = explainer.predict_parts(pred)

        # Explanation Breakdown Plot
        st.header(f"Breakdown Explanation for Your Risk Profile")
        fig = local_exp.plot(show=False)  
        st.plotly_chart(fig)

        # Obtener contribuciones ordenadas
        top_features = local_exp.result[['variable', 'contribution']].sort_values(by='contribution', ascending=False)

        # Separar las que suman al riesgo y las que lo reducen
        mayores_contribuciones = top_features[top_features['contribution'] > 0].head(5)
        menores_contribuciones = top_features[top_features['contribution'] < 0].tail(5)

        # Generar explicación en texto
        st.subheader("Top Risk-Increasing and Risk-Reducing Factors:")
        explicacion = "Based on your data, the factors that increase your heart attack risk the most are:\n"
        for _, row in mayores_contribuciones.iterrows():
            if ((row['variable'] != 'prediction') & (row['variable'] != 'intercept')):
                explicacion += f"- **{row['variable']}** (contribution: +{row['contribution']:.2f})\n"

        explicacion += "\nThe factors that decrease your risk the most are:\n"
        for _, row in menores_contribuciones.iterrows():
            explicacion += f"- **{row['variable']}** (contribution: {row['contribution']:.2f})\n"

        st.markdown(explicacion)

        X_train = explain_dashboard().values

        # Crear el explicador de LIME
        lime_explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=selected_variables,
            class_names=['Low Risk', 'High Risk'],
            mode='classification',
            discretize_continuous=True
        )

        # Explicar la predicción individual
        explanation = lime_explainer.explain_instance(
            numpy.array(pred),  # input del usuario
            modelo_cargado.predict_proba,
            num_features=10
        )

        # Mostrar la explicación como texto y gráfico
        st.subheader("Personalized visual explanation of your result")
        st.pyplot(explanation.as_pyplot_figure())

        st.subheader("Factors that influenced your result the most")
        for feature, weight in explanation.as_list():
            st.markdown(f"- **{feature}**: {weight:+.2f}")

def preprocess_input(pred):
    new_pred = []
    for i,p in enumerate(pred): 
        if p == 'No':
            p = 0 
        elif p=='Yes':
            p = 1
        new_pred.append(p)
    return numpy.array(new_pred)

def get_predicted_class(pred):
    pred = preprocess_input(pred)
    pred = pred.reshape(1, -1)
    return int(modelo_cargado.predict(pred)[0])

def get_prediction(pred):
    pred = preprocess_input(pred)
    pred= pred.reshape(1, -1)
    probab = modelo_cargado.predict_proba(pred)
    return int(probab[0][1] * 100)


def explain_dashboard():
    df = pd.read_csv("Modelo/data/heart_attack_prediction_dataset.csv")
    df = df.rename(columns={'Sex': 'Gender'})
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    diet_map = {'Unhealthy': 0, 'Average': 1, 'Healthy': 2}
    df['Diet'] = df['Diet'].map(diet_map)
    df = df[selected_variables]
    imputer = KNNImputer(n_neighbors=2)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns) 
    X =  df[selected_variables]
    X= X.sample(n=200, random_state=42)
    return X