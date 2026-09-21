import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('data.csv')

# Preprocessing
df['overtime'] = df['overtime'].map({'Yes': 1, 'No': 0})

X = df[[
'age', 'distance_from_home', 'monthly_income', 'years_at_company', 'job_satisfaction', 'overtime', 'num_companies_worked', 'training_hours' 
]]
Y = df['attrition']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, Y_train)

Y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy:.4f}')

joblib.dump(model, 'Employee_Attrition_Model.pkl')
joblib.dump(scaler, 'Employee_Attrition_Scaler.pkl')
