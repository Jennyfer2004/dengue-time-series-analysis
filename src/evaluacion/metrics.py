import os
import ast
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

def procesar_metricas_finales(df_detallado, nombre_provincia,usar_exog,df_original,lSTM=False):
    """Calcula métricas por horizonte y el DataFrame para el reporte."""
    
    df_detallado['Semana_H'] = (np.arange(len(df_detallado)) % 4) + 1
    resumen_horizontes = []
    
    for sem in range(1, 5):
        subset = df_detallado[df_detallado['Semana_H'] == sem]
        y_true, y_pred = subset['Real'], subset['Pred']
        
        # naive_forecast = df_original[target].shift(sem)
        # naive_errors = np.abs(df_original[target] - naive_forecast
        # naive_error = naive_errors.dropna().mean()

        y_true_safe = y_true.replace(0, 1)
        mape = np.mean(np.abs((y_true_safe - y_pred) / y_true_safe)) * 100
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # mase = mae / naive_error if naive_error != 0 else 0
        
        resumen_horizontes.append({
            'Horizonte': f'Semana {sem}',
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2),
            # 'MASE': round(mase, 2)
        })

    df_resumen = pd.DataFrame(resumen_horizontes).set_index('Horizonte')
    df_final= df_resumen.unstack().to_frame().T
    
    df_final.columns = [f"{m} (S{s[-1]})" for m, s in df_final.columns]
    df_final.insert(0, 'Provincia', nombre_provincia)
    
    if usar_exog==True:
        df_final.insert(1, 'Modelo', 'SARIMAX')
    elif usar_exog==False:
        df_final.insert(1, 'Modelo', 'SARIMA')
    elif lSTM:
        df_final.insert(1, 'Modelo', lSTM)
    else:
        df_final.insert(1, 'Modelo', usar_exog)

    return df_final


def diebold_mariano_test_global(y_true, y_pred1, y_pred2, h=4):
    """    Prueba de Diebold-Mariano para comparar la precisión de dos predicciones."""
    y_true, y_pred1, y_pred2 = np.array(y_true), np.array(y_pred1), np.array(y_pred2)
    e1, e2 = y_true - y_pred1, y_true - y_pred2
    d = np.abs(e1) - np.abs(e2) 
    
    d_mean = np.mean(d)
    T = len(d)
    
    # Ajuste de Newey-West usando el horizonte máximo (4 semanas)
    gamma = np.zeros(h)
    for i in range(h):
        gamma[i] = np.mean((d[:T-i] - d_mean) * (d[i:] - d_mean))
        
    var_d = gamma[0] + 2 * np.sum([((h - i) / h) * gamma[i] for i in range(1, h)])
    
    if var_d <= 0:
        return 0.0, 1.0
        
    dm_stat = d_mean / np.sqrt(var_d / T)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value