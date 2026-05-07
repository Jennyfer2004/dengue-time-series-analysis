import pandas as pd
import numpy as np
import os
import itertools

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

import matplotlib
matplotlib.use('Agg') 



def cargar_datos_provincia(nombre_provincia, carpeta_entrada):
    
    nombre_archivo = f'{nombre_provincia}.csv'
    ruta_completa = os.path.join(carpeta_entrada, nombre_archivo)
    
    if not os.path.exists(ruta_completa):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_completa}")
    
    df = pd.read_csv(ruta_completa)
    
    return df

def obtener_hiperparametros(nombre_provincia, ruta_params):
    if not os.path.exists(ruta_params):
        return [0], [1], [0], [0], [0], [0] 

    df_params = pd.read_csv(ruta_params)
    fila = df_params[df_params['Provincia'].str.lower() == nombre_provincia.lower()]
    
    if fila.empty:
        return [0], [0], [1], [0], [0], [1]

    def parse_lista(val):
        s = str(val).strip()
        return [int(x.strip()) for x in s.split(',')] if ',' in s else [int(float(s))]

    return (parse_lista(fila['p'].iloc[0]), parse_lista(fila['q'].iloc[0]), 
            parse_lista(fila['d'].iloc[0]), parse_lista(fila['P'].iloc[0]), 
            parse_lista(fila['Q'].iloc[0]), parse_lista(fila['D'].iloc[0]))
    
    
def buscar_mejor_sarima(train_y, train_exog, p_list, d_list, q_list, P_list, D_list, Q_list, s, ruta_cache):
    
    pdq = list(itertools.product(p_list, d_list, q_list))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(P_list, D_list, Q_list))]
    
    if os.path.exists(ruta_cache):
        res_df = pd.read_csv(ruta_cache)
        print(len(res_df),ruta_cache)
    else:
        print(ruta_cache)
        res_df = pd.DataFrame(columns=["param", "seasonal", "aic"])

    mejor_aic = res_df["aic"].min() if not res_df.empty else float("inf")
    mejor_cfg = None

    for param in pdq:
        for s_param in seasonal_pdq:

            if not res_df.empty and ((res_df['param'] == str(param)) & (res_df['seasonal'] == str(s_param))).any():
                continue
            print(param,s_param)
            try:
                mod = SARIMAX(train_y, exog=train_exog, order=param, seasonal_order=s_param,
                              enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit(disp=False)
                
                nuevo = pd.DataFrame({"param": [str(param)], "seasonal": [str(s_param)], "aic": [results.aic]})
                nuevo.to_csv(ruta_cache, mode='a', header=not os.path.exists(ruta_cache), index=False)
                
                if results.aic < mejor_aic:
                    mejor_aic = results.aic
                    mejor_cfg = (param, s_param)
            except:
                continue

    if mejor_cfg is None and not res_df.empty:
        idx = res_df["aic"].idxmin()
        mejor_cfg = (eval(res_df.loc[idx, "param"]), eval(res_df.loc[idx, "seasonal"]))
        
    return mejor_cfg
import csv

def guardar_metricas(y_true, y_pred, nombre_provincia, exog,parametros, ruta_archivo):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    tiene_exog = False
    if isinstance(exog, bool):
        tiene_exog = exog
    elif exog is not None:
        tiene_exog = not exog.empty if hasattr(exog, 'empty') else True

    modelo_nombre = "SARIMAX" if tiene_exog else "SARIMA"

    datos_fila = {
        "Provincia": nombre_provincia,
        "Modelo": modelo_nombre,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": f"{round(mape * 100, 2)}%",
        "Parametros": parametros
    }

    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    file_exists = os.path.isfile(ruta_archivo)

    with open(ruta_archivo, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=datos_fila.keys())
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(datos_fila)

    print(f"\nMétricas guardadas para {nombre_provincia} ({modelo_nombre}):")
    print(f"- MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape*100:.2f}%")
    
    return datos_fila


def generar_y_guardar_graficos(y_all, test_y, forecast_df, errores, prov, carpeta_out):
    
    os.makedirs(carpeta_out, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_all.index[-30:], y_all[-30:], label='Real', color='black', linewidth=2)
    plt.plot(test_y.index, forecast_df['mean'], label='Predicción', color='red')
    plt.fill_between(test_y.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='pink', alpha=0.3)
    plt.title(f'SARIMAX: Predicción {prov}')
    plt.legend()
    plt.savefig(os.path.join(carpeta_out, f'prediccion_{prov}.png'))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(test_y.index, errores, color='gray', label='Error')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'Residuales: {prov}')
    plt.savefig(os.path.join(carpeta_out, f'residuales_{prov}.png'))
    plt.close()

def ejecutar_workflow_sarimax(df, prov, folder_out, params_csv, s=52, n_forecast=4, columna="Casos", usar_exog=True,ruta_archivo="resultados/metricas_globales.csv"):

    os.makedirs(folder_out, exist_ok=True)
    ruta_cache = os.path.join(folder_out, f"cache_{prov}.csv")
    
    exog_cols = [c for c in df.columns if c not in [columna, 'Fecha', 'Provincias', 'Unnamed: 0']]
    exog_data = df[exog_cols] if usar_exog else None

    y = df[columna]
    train_y, test_y = y[:-n_forecast], y[-n_forecast:]
    train_ex = exog_data[:-n_forecast] if usar_exog else None
    test_ex = exog_data[-n_forecast:] if usar_exog else None

    p, q, d, P, Q, D = obtener_hiperparametros(prov, params_csv)
    
    best_order, best_seasonal = buscar_mejor_sarima(train_y, train_ex, p, d, q, P, D, Q, s, ruta_cache)
    
    model = SARIMAX(train_y, exog=train_ex, order=best_order, seasonal_order=best_seasonal,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    
    forecast = model.get_forecast(steps=n_forecast, exog=test_ex)
    f_df = forecast.summary_frame()
    f_df['mean'] = np.clip(f_df['mean'], 0, None)
    
    guardar_metricas(test_y, f_df['mean'], prov,usar_exog,[best_order, best_seasonal],ruta_archivo)
    generar_y_guardar_graficos(y, test_y, f_df, test_y - f_df['mean'], prov, folder_out)
    
    return model
