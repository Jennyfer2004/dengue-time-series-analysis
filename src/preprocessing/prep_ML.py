import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def preparar_datos_ml(nombre_provincia, carpeta_entrada, columna_objetivo, cols_a_excluir):
    """Prepara datos para ML con diferenciación"""

    ruta = os.path.join(carpeta_entrada, f'{nombre_provincia}.csv')
    if not os.path.exists(ruta): raise FileNotFoundError(f"No se encontró {ruta}")
    
    df = pd.read_csv(ruta)
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df = df.sort_values('Fecha').set_index('Fecha')
    
    df_diff = df.copy()
    for col in df_diff.columns:
        if col not in cols_a_excluir and col != columna_objetivo:
            df_diff[col] = df_diff[col].diff()
    
    df_diff[f'{columna_objetivo}_diff'] = df_diff[columna_objetivo].diff()
    df_diff = df_diff.dropna()
    return df, df_diff

def preparar_datos(nombre_provincia, carpeta_entrada, columna_objetivo, cols_a_excluir ,n_forecast):
    """Preparar datos ML con diff"""    
    
    ruta = os.path.join(carpeta_entrada, f'{nombre_provincia}.csv')
    
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró: {ruta}")
    
    df = pd.read_csv(ruta)
    df_diff = df.copy()

    for col in df_diff.columns:
        
        if col not in cols_a_excluir:
            df_diff[col] = df_diff[col].diff()

    df_diff = df_diff.dropna()
    
    X = df_diff.drop([columna_objetivo] + cols_a_excluir, axis=1, errors='ignore')
    y = df_diff[columna_objetivo]

    X_train, X_test = X[:-n_forecast-49], X[-n_forecast-49:]
    y_train, y_test = y[:-n_forecast-49], y[-n_forecast-49:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        "df_original": df,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled
    }
    
def reconstruir_y_evaluar(data, preds_diff, model_name, col):
    """Reconstruye predicciones diferenciadas a valores originales"""
    
    def reconstruir(valor_base, diffs):
        res = []
        actual = valor_base
        for d in diffs:
            actual = max(0, actual + d)
            res.append(actual)
        return np.array(res)

    idx_base = data["df_original"].index.get_loc(data["y_test"].index[0]) - 1
    valor_base = data["df_original"][col].iloc[idx_base]
    
    preds_reales = reconstruir(valor_base, preds_diff)
    y_test_original = data["df_original"][col].loc[data["y_test"].index].values

    return preds_reales, y_test_original