#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import shap


# %%
df = pd.read_csv("Modelo/data/heart_attack_prediction_dataset.csv")
df.head()

# %%
print(df.columns)

# %%
print(df.describe())

# %%
df = df.drop(columns=['Patient ID', 'Income','Physical Activity Days Per Week','Continent', 'Hemisphere'])
df = df.rename(columns={'Sex': 'Gender'})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})


bp_split = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic blood pressure'] = pd.to_numeric(bp_split[0])
df['Diastolic blood pressure'] = pd.to_numeric(bp_split[1])
df = df.drop(columns=['Blood Pressure'])

print(df['Diet'].unique())
print(df['Country'].unique())

diet_map = {'Unhealthy': 0, 'Average': 1, 'Healthy': 2}
df['Diet'] = df['Diet'].map(diet_map)

le = LabelEncoder()
df['Country'] = le.fit_transform(df['Country'])
country_mapping = {i: country for i, country in enumerate(le.classes_)}
print(country_mapping)

# min_max = pd.DataFrame({
#     'Min': df.min(numeric_only=True),
#     'Max': df.max(numeric_only=True)
# })
# min_max.to_csv('min_max.txt', sep='\t')
# numeric_cols = df.select_dtypes(include='number').columns
# df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())

# Opcional: guardar resultado
#df.to_csv("newNormalized.csv", index=False)
df['Heart Attack Risk'].value_counts()
# %%
df.info()

# %%
df.isna().sum()

# %%
imputer = KNNImputer(n_neighbors=2)

df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print(df)
# %%
df.isna().sum()

#%%
df.head()
# %%
correlation_matrix = df.corr()
correlation_matrix['Heart Attack Risk']

# %%
target_corr = correlation_matrix['Heart Attack Risk'].drop('Heart Attack Risk')

target_corr_sorted = target_corr.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=target_corr_sorted.values, y=target_corr_sorted.index, palette='coolwarm')
plt.title("Correlación con la variable 'Hearth-attack'")
plt.xlabel("Coeficiente de correlación")
plt.ylabel("Variables")
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.show()

# %%
target_column = "Heart Attack Risk"
X = df.drop(columns=[target_column, "Heart Attack Risk"])
y = df[target_column]

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_train.shape, y_train.shape

# %%
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# %%
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión: {accuracy}')

# %%
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# %%
# Guardar el modelo
joblib.dump(rf, './Modelo/modelo_rf.joblib')

#%%
# Cargar el modelo
modelo_cargado = joblib.load('./Modelo/modelo_rf.joblib')

# %%
predicciones = modelo_cargado.predict(X_test)
accuracy = accuracy_score(y_test, predicciones)
print(f'Precisión: {accuracy}')
print(classification_report(y_test, predicciones))

#%%
X = df.drop(columns=["Country","Heart Attack Risk"])
print(X)

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_train.shape, y_train.shape

# %%
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# %%
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión: {accuracy}')

# %%
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# %%
# Guardar el modelo
joblib.dump(rf, './Modelo/modelo_rf_doctors.joblib')

#%%
# Cargar el modelo
modelo_cargado = joblib.load('./Modelo/modelo_rf_doctors.joblib')

# %%
predicciones = modelo_cargado.predict(X_test)
accuracy = accuracy_score(y_test, predicciones)
print(f'Precisión: {accuracy}')
print(classification_report(y_test, predicciones))
selected_variables = [
    'Age',
    'Gender',  # Debe estar codificada como 0 = Female, 1 = Male
    'Diabetes',
    'Family History',
    'Smoking',
    'Obesity',
    'Alcohol Consumption',  # Categórica: conviene codificar como 0-1-2-3 o con one-hot
    'Exercise Hours Per Week',
    'Diet',
    'Previous Heart Problems',
    'Medication Use',
    'Stress Level',
    'Sedentary Hours Per Day',
    'Sleep Hours Per Day'    
]

X = df[selected_variables]

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_train.shape, y_train.shape

# %%
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# %%
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión: {accuracy}')

# %%
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest Patients")
plt.show()

# %%
# Guardar el modelo
joblib.dump(rf, 'Modelo/modelo_rf_patients.joblib')

#%%
# Cargar el modelo
modelo_cargado = joblib.load('Modelo/modelo_rf_patients.joblib')

# %%
predicciones = modelo_cargado.predict(X_test)
accuracy = accuracy_score(y_test, predicciones)
print(f'Precisión: {accuracy}')
print(classification_report(y_test, predicciones))


