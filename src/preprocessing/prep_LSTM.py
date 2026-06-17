from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import numpy as np


provincias=["Pinar del Río",'Artemisa',"La Habana","Mayabeque","Matanzas","Cienfuegos","Villa Clara","Sancti Spíritus","Ciego de Ávila",
            "Camagüey","Las Tunas","Granma","Holguín","Santiago de Cuba","Guantánamo","Isla de la Juventud"]
def load(nombre_provincia,carpeta_salida):
    """Cargar los datos"""
    nombre_archivo = f'{nombre_provincia.replace(" ", "_")}.csv'
    ruta_completa = os.path.join(carpeta_salida, nombre_archivo)

    df= pd.read_csv(ruta_completa)

    # df= df.drop(['Unnamed: 0', "Provincias"], axis=1)
    df= df.drop(["Provincias"], axis=1)

    return df   

def generar_acf_npast(carpeta = 'data/preprocessed/clima_lags', columna="Casos",carpeta_salida="outputs/graficos/"):
    for nombre_provincia in provincias:
        print(f"Analizando: {nombre_provincia}")
        df = load(nombre_provincia, carpeta)
        
        serie_diferenciada = df[columna].diff().dropna()
        
        acf_vals = acf(serie_diferenciada, nlags=26)
        
        plt.figure(figsize=(10, 4))
        plt.plot(acf_vals, marker='o')
        plt.axhline(y=0, linestyle='--', color='gray')
        
        # Aproximación del intervalo de confianza (1.96 / sqrt(N))
        conf_level = 1.96 / (len(serie_diferenciada)**0.5)
        plt.axhline(y=conf_level, linestyle='--', color='red', alpha=0.5)
        plt.axhline(y=-conf_level, linestyle='--', color='red', alpha=0.5)
        
        plt.title(f"ACF de la serie Diferenciada - {nombre_provincia}")
                
        ruta_carpeta = f"{carpeta_salida}{nombre_provincia}"
        
        os.makedirs(ruta_carpeta, exist_ok=True)
        
        nombre_archivo = os.path.join(ruta_carpeta, "acf_diferenciada.png")
        
        plt.savefig(nombre_archivo)
        
        plt.close()
        
def crear_secuencias_3d(data, n_past):
    """Convierte un array 2D en estructura 3D para LSTM."""
    
    X = []
    for i in range(n_past, len(data)):
        X.append(data[i - n_past:i, :])
    return np.array(X)

def preparar_datos_dl(nombre_provincia, carpeta_salida, n_forecast, ruta_n_past, cols_a_excluir):
    """Preparar datos para los modelos"""
    
    ruta = os.path.join(carpeta_salida, f'{nombre_provincia}.csv')
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
    df_n_past=pd.read_csv(ruta_n_past)
    n_past = int(df_n_past[df_n_past["provincia"]==nombre_provincia]["valor"].values[0])

    df = pd.read_csv(ruta)

    df_num = df.drop(cols_a_excluir, axis=1, errors='ignore')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df_num.values)

    X, y = [], []
    for i in range(n_past, len(data_scaled) - n_forecast + 1):
        X.append(data_scaled[i - n_past:i, :])
        y.append(data_scaled[i : i + n_forecast, 0]) # La columna 0 debe ser 'Casos'
    
    X, y = np.array(X), np.array(y)

    X_train, X_test = X[:-1], X[-1:] 
    y_train, y_test = y[:-1], y[-1:]
    
    return X_train, X_test, y_train, y_test, scaler, df_num.shape[1],n_past

def desescalar_predicciones(preds, y_true, scaler, n_features):
    """Invierte el escalado iterando sobre el eje del tiempo"""
    
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