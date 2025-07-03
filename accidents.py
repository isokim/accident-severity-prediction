import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset (example structure)
# In practice, you would load your actual dataset here
data = {
    'Accident_Severity': [2, 3, 1, 3, 2, 1, 3, 2],
    'Road_Type': ['urban', 'highway', 'rural', 'highway', 'urban', 'rural', 'highway', 'urban'],
    'Weather_Conditions': ['clear', 'rain', 'clear', 'fog', 'rain', 'clear', 'snow', 'clear'],
    'Light_Conditions': ['daylight', 'night-with-lighting', 'daylight', 'night-no-lighting', 
                         'daylight', 'daylight', 'night-with-lighting', 'daylight'],
    'Speed_Limit': [50, 110, 60, 110, 50, 60, 90, 50],
    'Vehicle_Type': ['car', 'truck', 'motorcycle', 'car', 'bus', 'car', 'truck', 'car'],
    'Road_Condition': ['dry', 'wet', 'dry', 'wet', 'wet', 'dry', 'icy', 'dry'],
    'Driver_Age': [32, 45, 28, 56, 39, 24, 50, 35],
    'Alcohol_Involved': [0, 1, 0, 0, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

# Preprocessing: Convert categorical variables to numerical
categorical_features = ['Road_Type', 'Weather_Conditions', 'Light_Conditions', 
                       'Vehicle_Type', 'Road_Condition']
numerical_features = ['Speed_Limit', 'Driver_Age', 'Alcohol_Involved']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# Create and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X = df.drop('Accident_Severity', axis=1)
y = df['Accident_Severity']

model.fit(X, y)

# Save the model for future use
joblib.dump(model, 'road_accident_severity_model.pkl')

print("Model trained and saved successfully.")
# Load the saved model
loaded_model = joblib.load('road_accident_severity_model.pkl')

# Create a hypothetical accident scenario
hypothetical_data = {
    'Road_Type': ['rural'],
    'Weather_Conditions': ['rain'],
    'Light_Conditions': ['night-no-lighting'],
    'Speed_Limit': [80],
    'Vehicle_Type': ['motorcycle'],
    'Road_Condition': ['wet'],
    'Driver_Age': [22],
    'Alcohol_Involved': [1]
}

hypothetical_df = pd.DataFrame(hypothetical_data)

# Make prediction
predicted_severity = loaded_model.predict(hypothetical_df)

print(f"Predicted Accident Severity: {predicted_severity[0]:.2f}")
# Output might be something like: Predicted Accident Severity: 2.83
# (indicating high probability of severe accident)