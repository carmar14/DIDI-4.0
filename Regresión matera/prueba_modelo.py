import joblib
from tensorflow.keras.models import load_model

model = load_model("model/modelo_mlp.keras")
scaler_X = joblib.load("model/scaler_X.pkl")
scaler_y = joblib.load("model/scaler_y.pkl")
'''
features = [
    'Soil_Moisture',
    'Ambient_Temperature',
    'Soil_Temperature',
    'Humidity',
    'Light_Intensity',
    'Soil_pH',
    'Nitrogen_Level',
    'Phosphorus_Level',
    'Potassium_Level',
    'Chlorophyll_Content',
    'Electrochemical_Signal'
]
'''

# Nueva muestra
new_data = [[27.8, 22.3, 20, 5.5, 6, 5.4, 1, 4.58, 3.9, 3.57, 9.41]]

# Escalar
new_data_scaled = scaler_X.transform(new_data)

# Predicción
pred_scaled = model.predict(new_data_scaled)

# Volver a escala real
pred_real = scaler_y.inverse_transform(pred_scaled)

# Outputs (multi-output regression)
'''targets = [
    'Ambient_Temperature',
    'Soil_Temperature',
    'Light_Intensity',
    'Soil_pH'
]
'''
print(pred_real)