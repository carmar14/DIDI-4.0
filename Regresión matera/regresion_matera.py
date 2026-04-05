'''
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ziya07/plant-health-data")

print("Path to dataset files:", path)
'''

#----Importación de librerías-----
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_time_lagged_data(X, y, lag=23):
    X_lagged = []
    y_lagged = []

    for i in range(lag, len(X)):
        #X_lagged.append(X[i-lag])
        X_lagged.append(X[i-lag:i].flatten())
        y_lagged.append(y[i])

    return np.array(X_lagged), np.array(y_lagged)

#----cargar dataset-----
df = pd.read_csv("plant_health_data.csv")
print(df.head())
print(df.columns)

#----seleccion de variables-----
# Inputs (NO incluir variables objetivo)
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

# Outputs (multi-output regression)
targets = [
    'Ambient_Temperature',
    'Soil_Temperature',
    'Light_Intensity',
    'Soil_pH'
]


X = df[features].values
y = df[targets].values

#---------Escalado---------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_lagged, y_lagged = create_time_lagged_data(X_scaled, y_scaled, lag=0)

#-----no tiene sentido en una serie temporal----
#---------division de dataset----

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled,
    test_size=0.2,
    random_state=42
)

#---------division de dataset----
'''
split = int(0.8 * len(X_lagged))

X_train = X_lagged[:split] 
X_test  = X_lagged[split:]

y_train = y_lagged[:split]
y_test  = y_lagged[split:]
'''
#---------arquitectura mlp----
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(4)  # 4 salidas
])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

#--------entrenamiento--------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]
)

#----------evaluacion-------
y_pred = model.predict(X_test)

# Invertir escalado
y_pred_real = scaler_y.inverse_transform(y_pred)
y_test_real = scaler_y.inverse_transform(y_test)


#------metricas por variable------
for i, name in enumerate(targets):
    rmse = np.sqrt(mean_squared_error(y_test_real[:, i], y_pred_real[:, i]))
    mae = mean_absolute_error(y_test_real[:, i], y_pred_real[:, i])
    r2 = r2_score(y_test_real[:, i], y_pred_real[:, i])

    print(f"\nVariable: {name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

#--------visualizacion de regresion-----
import matplotlib.pyplot as plt

for i, name in enumerate(targets):
    plt.figure()
    plt.scatter(y_test_real[:, i], y_pred_real[:, i])
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.title(name)
    plt.grid()
    plt.show()

for i, name in enumerate(targets):
    plt.figure()
    plt.plot(y_test_real[:, i])
    plt.plot(y_pred_real[:, i])
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.title(name)
    plt.grid()
    plt.show()

model.save('model/modelo_mlp.keras')
import joblib

joblib.dump(scaler_X, "model/scaler_X.pkl")
joblib.dump(scaler_y, "model/scaler_y.pkl")