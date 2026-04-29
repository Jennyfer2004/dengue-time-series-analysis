import pandas as pd
import numpy as np
import os
import itertools

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def cargar(nombre_provincia,carpeta_salida):
    
    nombre_archivo = f'{nombre_provincia.replace(" ", "_")}.csv'
    ruta_completa = os.path.join(carpeta_salida, nombre_archivo)

    df = pd.read_csv(ruta_completa)
    return df

def graficar(df,columna="Casos_Dengue"):
    
    # lags=110 para ver claramente lo que pasa en el periodo 52 y 104
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    serie_diferenciada = df[columna].diff().dropna()
    
    # ACF: Ayuda a identificar q (MA) y Q (SMA)
    plot_acf(serie_diferenciada, lags=110, ax=ax1)
    ax1.set_title('Autocorrelación (ACF) - Buscamos q y Q')

    # PACF: Ayuda a identificar p (AR) y P (SAR)
    plot_pacf(serie_diferenciada, lags=110, ax=ax2)
    ax2.set_title('Autocorrelación Parcial (PACF) - Buscamos p y P')

    plt.tight_layout()
    plt.show()


def obtener_parametros_provincia(nombre_provincia, ruta_parametros):
    """
    Busca en el CSV de parámetros la fila de la provincia y devuelve
    listas para p, d, q, P, D, Q.
    """
    if not os.path.exists(ruta_parametros):
        print(f"Error: No se encontró el archivo de parámetros en {ruta_parametros}")
        return

    df_params = pd.read_csv(ruta_parametros)
    
    fila = df_params[df_params['Provincia'].str.lower() == nombre_provincia.lower()]
    
    if fila.empty:
        print(f"Advertencia: No hay parámetros definidos para {nombre_provincia}. Usando valores base.")
        return 
    
    def limpiar_a_lista(valor):
        s = str(valor).strip()
        if ',' in s:
            return [int(x.strip()) for x in s.split(',') if x.strip() != ""]
        else:
            return [int(float(s))]

    p = limpiar_a_lista(fila['p'].iloc[0])
    d = limpiar_a_lista(fila['d'].iloc[0])
    q = limpiar_a_lista(fila['q'].iloc[0])
    P = limpiar_a_lista(fila['P'].iloc[0])
    D = limpiar_a_lista(fila['D'].iloc[0])
    Q = limpiar_a_lista(fila['Q'].iloc[0])

    return p, q, d, P, Q, D

def maunal_sarimax(df,exog_vars,nombre_provincia,carpeta,p,q,d,P,Q,D,s=52,columna="Casos_Dengue",n_forecast = 4):
    archivo_cache = f"resultados_sarimax_{nombre_provincia}.csv"
    ruta_completa = os.path.join(carpeta, archivo_cache)

    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(P, D, Q))]
    y = df["Casos_Dengue"]
     
    train_y = y[:-n_forecast]
    train_exog = exog_vars[:-n_forecast]
    test_y = y[-n_forecast:]
    test_exog = exog_vars[-n_forecast:]


    if os.path.exists(ruta_completa):
        resultados_df = pd.read_csv(ruta_completa)
        print(f"Cargado caché con {len(resultados_df)} combinaciones previas.")
    else:
        resultados_df = pd.DataFrame(columns=["param", "seasonal", "aic"])

    if not resultados_df.empty:
        
        mejor_aic = resultados_df["aic"].min()
        indice_mejor = resultados_df["aic"].idxmin()
        mejor_fila = resultados_df.loc[indice_mejor]
        
        mejor_aic = mejor_fila["aic"]
        mejores_params = (eval(mejor_fila["param"]), eval(mejor_fila["seasonal"]))
        
        print(f"Empezando con el mejor modelo previo: {mejores_params} (AIC: {mejor_aic})")
    else:
        mejor_aic = float("inf")
        mejores_params = None
        
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            
            ya_calculado = resultados_df[
                (resultados_df['param'] == str(param)) & 
                (resultados_df['seasonal'] == str(param_seasonal))
            ]
            
            if not ya_calculado.empty:
                continue
            
            try:
                mod = sm.tsa.statespace.SARIMAX(train_y,
                                                exog=train_exog,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit(disp=False)
                print(param,param_seasonal,results.aic)
                
                nuevo_resultado = pd.DataFrame({
                    "param": [str(param)], 
                    "seasonal": [str(param_seasonal)], 
                    "aic": [results.aic]
                })
                
                # Escribir al archivo (si no existe pone cabecera, si existe solo añade fila)
                nuevo_resultado.to_csv(ruta_completa, mode='a', 
                                       header=not os.path.exists(ruta_completa), 
                                       index=False)
                
                # Actualizar el DataFrame en memoria para evitar repetir en esta misma sesión
                resultados_df = pd.concat([resultados_df, nuevo_resultado], ignore_index=True)                
                if results.aic < mejor_aic:
                    mejor_aic = results.aic
                    mejores_params = (param, param_seasonal)
                    
            except:
                continue

    print(f"Mejor SARIMA: {mejores_params} con AIC: {mejor_aic}")

    best_order, best_seasonal_order = mejores_params

    model_final = SARIMAX(train_y, 
                          exog=train_exog, 
                          order=best_order,
                          seasonal_order=best_seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
    results_final = model_final.fit(disp=False)

    forecast = results_final.get_forecast(steps=n_forecast, exog=test_exog)
    forecast_df = forecast.summary_frame()
    forecast_df.index = test_y.index

    forecast_df['mean'] = np.clip(forecast_df['mean'], 0, None)
    # forecast_df['mean_ci_lower'] = np.clip(forecast_df['mean_ci_lower'], 0, None)
    # forecast_df['mean_ci_upper'] = np.clip(forecast_df['mean_ci_upper'], 0, None)

    plt.figure(figsize=(12, 6))

    plt.plot(y.index[-30:], y[-30:], label='Datos Reales', color='black', linewidth=2)

    plt.plot(test_y.index, forecast_df['mean'], label='Predicción', color='red')

    plt.fill_between(forecast_df.index, 
                    forecast_df['mean_ci_lower'], 
                    forecast_df['mean_ci_upper'], 
                    color='pink', alpha=0.3, label='Intervalo de Confianza 95%')

    plt.gcf().autofmt_xdate() 

    plt.title('Predicción de Casos de Dengue - Serie Temporal Real')
    plt.xlabel('Fecha (Semanas)')
    plt.ylabel('Número de Casos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linewidth=1, alpha=0.5) # Línea en el cero
    plt.show()
    y_true = test_y
    y_pred = forecast_df['mean']

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"Error Absoluto Medio (MAE): {mae:.2f} casos")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f} casos")
    print(f"Error Porcentual Absoluto Medio (MAPE): {mape*100:.2f}%")

    error_semanal = y_true - y_pred

    plt.figure(figsize=(14, 6))
    plt.plot(error_semanal.index, error_semanal, color='gray', label='Error (Real - Predicho)')
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title('Error de Predicción por Semana (Residuales)')
    plt.xlabel('Semana')
    plt.ylabel('Número de Casos de Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.show()
    
    return model_final

def maunal_sarima(df,nombre_provincia,carpeta,p,q,d,P,Q,D,s=52,columna="Casos_Dengue",n_forecast = 4):
    archivo_cache = f"resultados_sarimax_{nombre_provincia}.csv"
    ruta_completa = os.path.join(carpeta, archivo_cache)

    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(P, D, Q))]
    y = df["Casos_Dengue"]
     
    train_y = y[:-n_forecast]
    test_y = y[-n_forecast:]


    if os.path.exists(ruta_completa):
        resultados_df = pd.read_csv(ruta_completa)
        print(f"Cargado caché con {len(resultados_df)} combinaciones previas.")
    else:
        resultados_df = pd.DataFrame(columns=["param", "seasonal", "aic"])

    if not resultados_df.empty:
        
        mejor_aic = resultados_df["aic"].min()
        indice_mejor = resultados_df["aic"].idxmin()
        mejor_fila = resultados_df.loc[indice_mejor]
        
        mejor_aic = mejor_fila["aic"]
        mejores_params = (eval(mejor_fila["param"]), eval(mejor_fila["seasonal"]))
        
        print(f"Empezando con el mejor modelo previo: {mejores_params} (AIC: {mejor_aic})")
    else:
        mejor_aic = float("inf")
        mejores_params = None
        
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            
            ya_calculado = resultados_df[
                (resultados_df['param'] == str(param)) & 
                (resultados_df['seasonal'] == str(param_seasonal))
            ]
            
            if not ya_calculado.empty:
                continue
            
            try:
                mod = sm.tsa.statespace.SARIMAX(train_y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit(disp=False)
                print(param,param_seasonal,results.aic)
                
                nuevo_resultado = pd.DataFrame({
                    "param": [str(param)], 
                    "seasonal": [str(param_seasonal)], 
                    "aic": [results.aic]
                })
                
                # Escribir al archivo (si no existe pone cabecera, si existe solo añade fila)
                nuevo_resultado.to_csv(ruta_completa, mode='a', 
                                       header=not os.path.exists(ruta_completa), 
                                       index=False)
                
                # Actualizar el DataFrame en memoria para evitar repetir en esta misma sesión
                resultados_df = pd.concat([resultados_df, nuevo_resultado], ignore_index=True)                
                if results.aic < mejor_aic:
                    mejor_aic = results.aic
                    mejores_params = (param, param_seasonal)
                    
            except:
                continue

    print(f"Mejor SARIMA: {mejores_params} con AIC: {mejor_aic}")

    best_order, best_seasonal_order = mejores_params

    model_final = SARIMAX(train_y, 
                          order=best_order,
                          seasonal_order=best_seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
    results_final = model_final.fit(disp=False)

    forecast = results_final.get_forecast(steps=n_forecast)
    forecast_df = forecast.summary_frame()
    forecast_df.index = test_y.index

    forecast_df['mean'] = np.clip(forecast_df['mean'], 0, None)
    # forecast_df['mean_ci_lower'] = np.clip(forecast_df['mean_ci_lower'], 0, None)
    # forecast_df['mean_ci_upper'] = np.clip(forecast_df['mean_ci_upper'], 0, None)

    plt.figure(figsize=(12, 6))

    plt.plot(y.index[-30:], y[-30:], label='Datos Reales', color='black', linewidth=2)

    plt.plot(test_y.index, forecast_df['mean'], label='Predicción', color='red')

    plt.fill_between(forecast_df.index, 
                    forecast_df['mean_ci_lower'], 
                    forecast_df['mean_ci_upper'], 
                    color='pink', alpha=0.3, label='Intervalo de Confianza 95%')

    plt.gcf().autofmt_xdate() 

    plt.title('Predicción de Casos de Dengue - Serie Temporal Real')
    plt.xlabel('Fecha (Semanas)')
    plt.ylabel('Número de Casos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linewidth=1, alpha=0.5) # Línea en el cero
    plt.show()
    y_true = test_y
    y_pred = forecast_df['mean']

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"Error Absoluto Medio (MAE): {mae:.2f} casos")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f} casos")
    print(f"Error Porcentual Absoluto Medio (MAPE): {mape*100:.2f}%")

    error_semanal = y_true - y_pred

    plt.figure(figsize=(14, 6))
    plt.plot(error_semanal.index, error_semanal, color='gray', label='Error (Real - Predicho)')
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title('Error de Predicción por Semana (Residuales)')
    plt.xlabel('Semana')
    plt.ylabel('Número de Casos de Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.show()
    
    return model_final