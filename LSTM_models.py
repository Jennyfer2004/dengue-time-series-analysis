import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Layer, Input
from tensorflow.keras import Model

import pandas as pd
import numpy as np
import csv
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 

from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def preparar_datos_dl(nombre_provincia, carpeta_salida, n_forecast, n_past, cols_a_excluir):
    ruta = os.path.join(carpeta_salida, f'{nombre_provincia}.csv')
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
    
    df = pd.read_csv(ruta)
    # Seleccionar solo columnas numéricas útiles
    df_num = df.drop(cols_a_excluir, axis=1, errors='ignore')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df_num.values)

    X, y = [], []
    for i in range(n_past, len(data_scaled) - n_forecast + 1):
        X.append(data_scaled[i - n_past:i, :])
        y.append(data_scaled[i : i + n_forecast, 0]) # La columna 0 debe ser 'Casos'
    
    X, y = np.array(X), np.array(y)

    # Split: últimos 'n_forecast' para test
    X_train, X_test = X[:-1], X[-1:] # Tomamos la última secuencia disponible para predecir el futuro
    y_train, y_test = y[:-1], y[-1:]
    
    return X_train, X_test, y_train, y_test, scaler, df_num.shape[1]

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
        
        # Reducimos a un vector global
        x = tf.keras.layers.GlobalAveragePooling1D()(attention_layer)
        x = Dense(32, activation='relu')(x)

    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def desescalar_predicciones(preds, y_true, scaler, n_features):
    """
    Corrección: Invierte el escalado iterando sobre el eje del TIEMPO (pasos futuros),
    no sobre las features, para mantener la forma (n_muestras, n_pasos_futuros).
    """
    
    n_samples = preds.shape[0]
    n_steps = preds.shape[1] 
    
    preds_inv = np.zeros((n_samples, n_steps))
    y_true_inv = np.zeros((n_samples, n_steps))

    dummy = np.zeros((n_samples, n_features))

    for i in range(n_steps):

        dummy[:, 0] = preds[:, i]

        transformed = scaler.inverse_transform(dummy)
        preds_inv[:, i] = transformed[:, 0]

        dummy[:, 0] = y_true[:, i]
        transformed_true = scaler.inverse_transform(dummy)
        y_true_inv[:, i] = transformed_true[:, 0]

    return preds_inv, y_true_inv


def guardar_metricas_dl(y_true, y_pred, prov, modelo, ruta="resultados/metricas_globales.csv"):
    
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    mape = mean_absolute_percentage_error(y_true.flatten(), y_pred.flatten())

    datos_fila = {
        "Provincia": prov,
        "Modelo": f"DL_{modelo}",
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": f"{round(mape * 100, 2)}%",
        "Parametros": "N_past:8, Epochs:100, EarlyStop"
    }

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    file_exists = os.path.isfile(ruta)
    
    with open(ruta, mode='a', newline='', encoding='utf-8') as f:
        
        writer = csv.DictWriter(f, fieldnames=datos_fila.keys())
        
        if not file_exists: writer.writeheader()
        writer.writerow(datos_fila)
    
    return mae


def ejecutar_dl_workflow(nombre_provincia, carpeta_entrada, n_forecast=4, n_past=8, cols_excluir=["Provincias", "Fecha", "Año", "Semana Estadística"]):
    
    architectures = ['stacked', 'bidirectional', 'attention']
    
    X_train, X_test, y_train, y_test, scaler, n_feats = preparar_datos_dl(
        nombre_provincia, carpeta_entrada, n_forecast, n_past, cols_excluir
    )
    
    df = pd.read_csv(os.path.join(carpeta_entrada, f'{nombre_provincia}.csv'))
    df_num = df.drop(cols_excluir, axis=1, errors='ignore')

    data_scaled = scaler.transform(df_num.values)

    best_mae = float('inf')
    best_arch = None
    final_preds_plot = None
    final_reales_plot = None

    for arch in architectures:
        print(f"Entrenando {arch}...")
        
        model = build_lstm_model(arch, (X_train.shape[1], X_train.shape[2]))
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, 
                  callbacks=[early_stop], verbose=0)

        last_sequence = X_test[-1].copy() # Forma (8, n_features)
        current_preds_scaled = []
        
        for step in range(n_forecast):

            input_seq = last_sequence.reshape(1, n_past, n_feats)
            
            next_pred_scaled = model.predict(input_seq, verbose=0)
            current_preds_scaled.append(next_pred_scaled[0, 0])
            
            last_sequence = np.roll(last_sequence, -1, axis=0)
            
            last_sequence[-1, 0] = next_pred_scaled[0, 0]
            
        current_preds_scaled = np.array(current_preds_scaled).reshape(1, -1)
        
        dummy_preds = np.zeros((n_forecast, n_feats))
        dummy_preds[:, 0] = current_preds_scaled.flatten()
        p_reales = scaler.inverse_transform(dummy_preds)[:, 0]
        
        dummy_reales = np.zeros((n_forecast, n_feats))
        dummy_reales[:, 0] = y_test[-1] 
        y_reales = scaler.inverse_transform(dummy_reales)[:, 0]

        current_mae = guardar_metricas_dl(y_reales, p_reales, nombre_provincia, arch)

        if current_mae < best_mae:
            best_mae = current_mae
            best_arch = arch
            final_preds_plot = p_reales
            final_reales_plot = y_reales

    if final_preds_plot is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(final_reales_plot, label="Real", marker='o', color='black')
        plt.plot(final_preds_plot, label=f"Pred ({best_arch})", linestyle='--', color='red')
        plt.title(f"Mejor Modelo DL: {best_arch} - {nombre_provincia}")
        plt.legend()
        plt.savefig(f"resultados/graficos_dl_{nombre_provincia}.png")
        plt.close()
        print(f"✅ Finalizado {nombre_provincia}. Mejor Arquitectura: {best_arch}")

