import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Layer, Input
from tensorflow.keras import Model

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def build_lstm_model(model_type, input_shape):
    """
    Tipos: 'stacked', 'bidirectional', 'attention'
    """
    inputs = Input(shape=input_shape)

    if model_type == 'stacked':
        x = LSTM(64, return_sequences=True, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(32, activation='relu')(x)
        x = Dropout(0.2)(x)

    elif model_type == 'bidirectional':
        x = Bidirectional(LSTM(64, return_sequences=True, activation='relu'))(inputs)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(32, activation='relu'))(x)
        x = Dropout(0.2)(x)

    elif model_type == 'attention':
        # Capa LSTM que devuelve secuencias para que la Atención pueda "mirar" atrás
        lstm_out = LSTM(64, return_sequences=True, activation='relu')(inputs)
        # Mecanismo de Atención simplificado (Self-Attention)
        query = Dense(64)(lstm_out)
        value = Dense(64)(lstm_out)
        attention_layer = Attention()([query, value])
        # Reducimos a un vector global (Global Average Pooling o similar)
        x = tf.keras.layers.GlobalAveragePooling1D()(attention_layer)
        x = Dense(32, activation='relu')(x)

    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_sequences_multi(data, n_past, n_future):
        X, y = [], []
        for i in range(n_past, len(data) - n_future + 1):
            X.append(data[i - n_past:i, :])
            # Aquí guardamos un vector de 4 valores (semanas 1 a 4)
            y.append(data[i : i + n_future, 0])
        return np.array(X), np.array(y)
    
def load(nombre_provincia,carpeta_salida, n_forecast):
    
    nombre_archivo = f'{nombre_provincia.replace(" ", "_")}.csv'
    ruta_completa = os.path.join(carpeta_salida, nombre_archivo)

    df= pd.read_csv(ruta_completa)

    df= df.drop(['Unnamed: 0', "Provincias"], axis=1)
    print(df.columns)


    features = [
        'Casos_Dengue',"Precipitaciones","Temperatura med",'Humedad Relat', 'Temperatura med_lag17', 'Mes',"Semana Estadística"
    ]

    data_selected = df[features].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_selected)

    X, y = create_sequences_multi(data_scaled, n_past=8, n_future=n_forecast)

    y_train = y[:-n_forecast]
    X_train= X[:-n_forecast]
    y_test = y[-n_forecast:]
    X_test = X[-n_forecast:]
    return(data_scaled,X_train,y_train,X_test,y_test)
    
def main( nombre_provincia, carpeta_salida, n_forecast = 4):
    
    scaler = MinMaxScaler(feature_range=(0, 1))

    data_scaled,X_train,y_train,X_test,y_test=load( nombre_provincia, carpeta_salida, n_forecast = 4)
   
    architectures = ['stacked', 'bidirectional', 'attention']
    results = {}

    for arch in architectures:
        print(f"\n--- Entrenando Modelo: {arch.upper()} ---")

        model = build_lstm_model(arch, (X_train.shape[1], X_train.shape[2]))

        # Entrenar con Early Stopping para evitar Overfitting
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0 # Para no llenar la consola
        )

        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        results[arch] = {'MAE': mae, 'History': history}
        print(f"Resultado {arch}: MAE = {mae:.4f}")

        mejor_tipo = min(results, key=lambda k: results[k]['MAE'])
        print(f"\n🏆 Mejor Modelo Detectado: {mejor_tipo.upper()}")

        preds_escaladas = model.predict(X_test)

        dummy_preds = np.zeros((len(preds_escaladas), data_scaled.shape[1]))

        dummy_preds[:, 0] = preds_escaladas[:, 0]
        preds_reales = scaler.inverse_transform(dummy_preds)[:, 0]

        dummy_test = np.zeros((len(y_test), data_scaled.shape[1]))
        dummy_test[:, 0] = y_test[:, 0]
        y_test_original = scaler.inverse_transform(dummy_test)[:, 0]

        print(f"--- Métricas en Unidades Reales (Casos) ---")
        print(f"MAE Real: {mean_absolute_error(y_test_original, preds_reales):.2f} casos")
        print(f"RMSE Real: {np.sqrt(mean_squared_error(y_test_original, preds_reales)):.2f}")
        print(f"MAPE Real: {mean_absolute_percentage_error(y_test_original, preds_reales)*100:.2f}%")

        plt.figure(figsize=(12, 6))
        plt.plot(y_test_original, label="Casos Reales", color='#1f77b4', linewidth=2, marker='o', markersize=4)
        plt.plot(preds_reales, label=f"Predicción ({mejor_tipo})", color='#d62728', linestyle='--', linewidth=2)

        plt.title(f"Evaluación de Predicción de Dengue: {mejor_tipo.capitalize()}", fontsize=14)
        plt.xlabel("Semanas de Test", fontsize=12)
        plt.ylabel("Número de Casos", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
