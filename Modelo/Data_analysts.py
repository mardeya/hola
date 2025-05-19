import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import shap
from sklearn.impute import KNNImputer
from streamlit_shap import st_shap
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import altair as alt
import pycountry
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




modelo_cargado = joblib.load("Modelo/modelo_rf.joblib")

columns = [
    "Age",
    "Gender",
    "Cholesterol",
    "Heart Rate",
    "Diabetes",
    "Family History",
    "Smoking",
    "Obesity",
    "Alcohol Consumption",
    "Exercise Hours Per Week",
    "Diet",
    "Previous Heart Problems",
    "Medication Use",
    "Stress Level",
    "Sedentary Hours Per Day",
    "BMI",
    "Triglycerides",
    "Sleep Hours Per Day",
    "Country",
    "Systolic blood pressure",
    "Diastolic blood pressure"
]
Feature_categories = {
    "Age": "Demographic",
    "Gender": "Demographic",
    "Country": "Demographic",

    "Cholesterol": "Clinical",
    "Heart Rate": "Clinical",
    "Diabetes": "Clinical",
    "Family History": "Clinical",
    "Previous Heart Problems": "Clinical",
    "Medication Use": "Clinical",
    "Systolic blood pressure": "Clinical",
    "Diastolic blood pressure": "Clinical",
    "BMI": "Clinical",
    "Triglycerides": "Clinical",

    "Smoking": "Lifestyle",
    "Obesity": "Lifestyle",
    "Alcohol Consumption": "Lifestyle",
    "Exercise Hours Per Week": "Lifestyle",
    "Diet": "Lifestyle",
    "Stress Level": "Lifestyle",
    "Sedentary Hours Per Day": "Lifestyle",
    "Sleep Hours Per Day": "Lifestyle"
}


color_theme_map = {
    'blues': px.colors.sequential.Blues,
    'cividis': px.colors.sequential.Cividis,
    'greens': px.colors.sequential.Greens,
    'inferno': px.colors.sequential.Inferno,
    'magma': px.colors.sequential.Magma,
    'plasma': px.colors.sequential.Plasma,
    'reds': px.colors.sequential.Reds,
    'viridis': px.colors.sequential.Viridis,
    'turbo': px.colors.sequential.Turbo,
    'rainbow': px.colors.qualitative.Bold 
}

def make_donut(input_response, input_text, input_color):
    # Define color schemes
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    elif input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    elif input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    elif input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']
    else:
        chart_color = ['gray', '#cccccc']

    # Data for filled and empty parts
    source = pd.DataFrame({
        'category': ['filled', 'empty'],
        'value': [input_response, 100 - input_response]
    })

    # Data for background circle
    source_bg = pd.DataFrame({
        'category': ['background'],
        'value': [100]
    })

    # Background chart
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=50, outerRadius=70).encode(
        theta=alt.Theta('value:Q'),
        color=alt.value(chart_color[1])  # darker / background color
    )

    # Foreground (main) donut chart
    plot = alt.Chart(source).mark_arc(innerRadius=50, outerRadius=70).encode(
        theta=alt.Theta('value:Q'),
        color=alt.Color('category:N',
                        scale=alt.Scale(
                            domain=['filled', 'empty'],
                            range=[chart_color[0], 'transparent']
                        ),
                        legend=None)
    )

    # Center text
    text = alt.Chart(pd.DataFrame({'label': [f'{input_response} %']})).mark_text(
        align='center',
        font='Lato',
        fontSize=24,
        fontWeight=700,
        fontStyle='italic',
        color=chart_color[0],
        dy=-5
    ).encode(
        text='label:N'
    )

    # Optional label under percentage
    sublabel = alt.Chart(pd.DataFrame({'label': [input_text]})).mark_text(
        align='center',
        font='Lato',
        fontSize=13,
        color=chart_color[0],
        dy=15
    ).encode(
        text='label:N'
    )

    # Combine all layers (background → main chart → text)
    return plot_bg + plot + text + sublabel

def get_iso3(country_name):
    try:
        return pycountry.countries.get(name=country_name).alpha_3
    except:
        return None


def make_choropleth(input_df, input_id, input_column, input_color_theme):
    choropleth = px.choropleth(
        input_df,
        locations=input_id,
        color=input_column,
        locationmode="ISO-3",
        color_discrete_sequence=color_theme_map[input_color_theme],
        hover_name = 'Country',
        hover_data={input_column:True, input_id: False},
        scope="world",
    )
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=500
    )
    return choropleth
def plot_by_variable(name, df, color):
    summary = []
    for group in df[name].unique():
        subset = df[df[name] == group]
        for risk in [0, 1]:
            real_count = (subset['True Risk'] == risk).sum()
            pred_count = (subset['Predicted Risk'] == risk).sum()

            summary.extend([
                {'Group': group, 'Risk': 'Low Risk' if risk == 0 else 'High Risk', 'Type': 'Real', 'Count': real_count},
                {'Group': group, 'Risk': 'Low Risk' if risk == 0 else 'High Risk', 'Type': 'Predicted', 'Count': pred_count},
            ])

    summary_df = pd.DataFrame(summary)
    fig = px.bar(
        summary_df,
        x='Group',
        y='Count',
        color='Type',
        barmode='group',
        facet_col='Risk',
        color_discrete_sequence=color_theme_map[color],
        labels={'Group': name, 'Count': 'Número de Casos'}
    )
    st.plotly_chart(fig, use_container_width=True)


def show_dashboard(selected_color_theme):
    ACCURACY, PRECISION, SPECIFITY, RECALL, F1SCORE, pct_high_risk, prob, y_test= get_metrics()
    st.header("Data Scientist Dashboard: Model Testing Interface")
    col = st.columns((1.5, 3, 3), gap='medium')
    with col[0]:
        st.markdown('#### Model Metrics ')
        st.markdown('##### Accuracy ')
        chart = make_donut(round(ACCURACY * 100, 2), 'Accuracy', 'green')
        st.altair_chart(chart,
                        use_container_width=True, on_select='ignore')
        st.markdown('##### Precision on High Risk class ')
        st.altair_chart(make_donut(round(PRECISION * 100,2), 'Precision', 'blue'),
                        use_container_width=True,on_select='ignore')
        st.markdown('##### Recall on High Risk class ')
        st.altair_chart(make_donut(round(SPECIFITY * 100,2), 'Recall', 'orange'),
                        use_container_width=True,on_select='ignore')
        st.markdown('##### Specificity on Low Risk class ')
        st.altair_chart(make_donut(round(RECALL * 100,2), 'Specificity', 'red'),
                        use_container_width=True,on_select='ignore')
        st.markdown('##### F1-SCORE ')
        st.altair_chart(make_donut(round(F1SCORE * 100,2), 'F1-Score', 'green'),
                        use_container_width=True,on_select='ignore')
        st.markdown('#### Dataset Statistics')
        st.metric(label="'%' of Predicted High Risk", value=f"{pct_high_risk}%", border=True)
        st.metric(label="Average Probability of class High Risk", value=f"{prob}%", border=True)

    with col[1]:

        st.markdown('#### Top Feature by Country')
        explainer, _, shap_values = explain_dashboard()
        df = get_df()
        X = df[columns]
        X = X.sample(n=200, random_state=42)

        shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
        shap_df['Country'] = X['Country'].values
        country_shap = (
            shap_df.drop(columns='Country')
            .abs()
            .groupby(shap_df['Country'])
            .mean()
            .reset_index()
        )

        # Convert to long format for ranking
        long_df = country_shap.melt(id_vars='Country', var_name='Feature', value_name='MeanAbsSHAP')

        # Get top feature per country (rank 1)
        long_df['Rank'] = long_df.groupby('Country')['MeanAbsSHAP'].rank(ascending=False, method='first')
        top_feature_df = long_df[long_df['Rank'] == 1].copy()

        # Add ISO3 country codes
        top_feature_df['country_code'] = top_feature_df['Country'].apply(get_iso3)

        # Now create choropleth colored by this top feature (categorical)
        choropleth = make_choropleth(
            input_df=top_feature_df,
            input_id='country_code',
            input_column='Feature',  # single top feature per country
            input_color_theme=selected_color_theme
        )
        st.plotly_chart(choropleth, use_container_width=True)

        st.subheader("How Features Influence Predictions")
        st.markdown("""
        <p style="margin-bottom:4px;">The plot below uses <b>SHAP values</b> to show how each feature affects the model's prediction.</p>
        <p style="margin-bottom:4px;">Each dot is a person in the dataset.</p>
        <p style="margin-bottom:4px;">The horizontal position shows whether the feature increases (right) or decreases (left) the risk.</p>
        <p style="margin-bottom:4px;">The color represents the actual value of the feature (e.g., high or low cholesterol).</p>
        <p style="margin-bottom:4px;">Features are sorted by overall importance.</p>
        """, unsafe_allow_html=True)
        st_shap(shap.plots.beeswarm(shap_values, max_display=15), height=500, width=600)
        

        with col[2]:
            st.markdown('#### Prediction vs Real')
            selected_categories2 = st.selectbox(
                label="Filter by Feature Category",
                options=['Gender','Previous Heart Problems', 'Medication Use', 'Smoking','Obesity','Alcohol Consumption', 'Diet'],
            )
            df = y_test[1].copy()
            df['True Risk'] = y_test[2]
            df['Predicted Risk'] = y_test[0]

            plot_by_variable(selected_categories2, df, selected_color_theme)

            feature_importance = modelo_cargado.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': columns,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)
            importance_df['Category'] = importance_df['Feature'].map(Feature_categories)
            with st.expander("🔧 Filter Options"):
                selected_categories = st.multiselect(
                    "Filter by Feature Category",
                    options=sorted(set(Feature_categories.values())),
                    default=sorted(set(Feature_categories.values()))
                )
                filtered_df = importance_df.copy()
                filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]
            if filtered_df.empty:
                st.warning("No features match your filters.")
            else:
                fig = px.bar(
                    filtered_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance (Random Forest)',
                    color='Importance',
                    color_continuous_scale=selected_color_theme
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)


def get_df():
    df = pd.read_csv("Modelo/data/heart_attack_prediction_dataset.csv")
    df = df.drop(columns=['Patient ID', 'Income',
                 'Physical Activity Days Per Week', 'Continent', 'Hemisphere'])
    df = df.rename(columns={'Sex': 'Gender'})
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    diet_map = {'Unhealthy': 0, 'Average': 1, 'Healthy': 2}
    df['Diet'] = df['Diet'].map(diet_map)
    bp_split = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic blood pressure'] = pd.to_numeric(bp_split[0])
    df['Diastolic blood pressure'] = pd.to_numeric(bp_split[1])
    df = df.drop(columns=['Blood Pressure'])
    return df

def get_metrics():
    df = get_df()
    le = LabelEncoder()
    df['Country'] = le.fit_transform(df['Country'])
    target_column = "Heart Attack Risk"
    X = df.drop(columns=[target_column, "Heart Attack Risk"])
    y = df[target_column]
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    y_pred = modelo_cargado.predict(X_test)
    y_proba = modelo_cargado.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=0, average='binary')
    specifity = precision_score(y_test, y_pred, pos_label=1, average='binary')
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    num_high_risk = sum(1 for p in y_pred if p == 1)
    pct_high_risk = round(100 * num_high_risk / len(y_pred), 2)
    high_risk_probs = [prob[1] for prob in y_proba]
    avg_high_risk_prob = round(100 * np.mean(high_risk_probs), 2)
    return accuracy, precision, specifity, recall, f1score, pct_high_risk, avg_high_risk_prob, [y_pred, X_test,y_test]

@st.cache_resource
def explain_dashboard():
    df = get_df()
    le = LabelEncoder()
    df['Country'] = le.fit_transform(df['Country'])
    imputer = KNNImputer(n_neighbors=2)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    X = df[columns]
    explainer = shap.Explainer(modelo_cargado)
    X = X.sample(n=200, random_state=42)
    shap_values = explainer(X)[:,:,1]
    return explainer, X, shap_values


def get_prediction(pred):
    pred = np.array(pred).reshape(1, -1)
    probab = modelo_cargado.predict_proba(pred)
    return int(probab[0][1] * 100)


def display_risk_indicator(value):
    if value < 30:
        color = "green"
        image_path = "images/SemaforoVerde.png"
    elif 30 <= value < 70:
        color = "yellow"
        image_path = "images/SemaforoAmarillo.png"
    else:
        color = "red"
        image_path = "images/SemaforoRojo.png"

    col1, col2 = st.columns([1, 1])
    with col1:
        image = Image.open(image_path)
        st.image(image, width=250)
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
                {value} %
            </div>
            """,
            unsafe_allow_html=True
        )
