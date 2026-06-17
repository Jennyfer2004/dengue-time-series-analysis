import os
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils import guardar_reporte
from src.preprocessing.prep_SARIMA import obtener_hiperparametros, cargar_datos_provincia
from src.modelos.sarima_parametros import entrenar_instancia_sarimax

def buscar_mejor_sarima(train_y, train_exog, p_list, d_list, q_list, P_list, D_list, Q_list, s, ruta_cache):
    """Ejecuta una búsqueda en cuadrícula (Grid Search) para minimizar el AIC."""
    
    pdq = list(itertools.product(p_list, d_list, q_list))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(P_list, D_list, Q_list))]
    
    if os.path.exists(ruta_cache):
        res_df = pd.read_csv(ruta_cache)
    else:
        res_df = pd.DataFrame(columns=["param", "seasonal", "aic"])

    mejor_aic = res_df["aic"].min() if not res_df.empty else float("inf")
    mejor_cfg = None

    for param in pdq:
        for s_param in seasonal_pdq:
            if not res_df.empty and ((res_df['param'] == str(param)) & (res_df['seasonal'] == str(s_param))).any():
                continue
            
            try:
                mod = SARIMAX(train_y, exog=train_exog, order=param, seasonal_order=s_param,
                              enforce_stationarity=False, enforce_invertibility=False)
                results = entrenar_instancia_sarimax(train_y, train_exog, param, s_param)

                aic = results.aic
                
                nuevo_registro = pd.DataFrame([{"param": str(param), "seasonal": str(s_param), "aic": aic}])
                res_df = pd.concat([res_df, nuevo_registro], ignore_index=True)
                res_df.to_csv(ruta_cache, index=False)

                if aic < mejor_aic:
                    mejor_aic = aic
                    mejor_cfg = (param, s_param)
            except:
                continue
            

    if mejor_cfg is None and not res_df.empty:
        idx = res_df["aic"].idxmin()
        mejor_cfg = (eval(res_df.loc[idx, "param"]), eval(res_df.loc[idx, "seasonal"]))
        
    return mejor_cfg

def ejecutar_workflow_sarimax(df, prov, folder_out, params_csv, s=52, n_forecast=4, columna="Casos", usar_exog=True, ruta_guardar="outputs/tables/metricas_globales.csv"):
    """Orquesta la optimización de parámetros y almacena los checkpoints de entrenamiento en disco."""
    
    os.makedirs(folder_out, exist_ok=True)
    ruta_cache = os.path.join(folder_out, f"resultados_sarimax_{prov}.csv" if usar_exog else f"resultados_sarima_{prov}.csv")

    exog_cols = [c for c in df.columns if c not in [columna, 'Fecha', 'Provincias', 'Unnamed: 0', "Mes", "Año"]]
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
    
    guardar_reporte(test_y, f_df['mean'], prov, usar_exog, [best_order, best_seasonal], ruta_guardar)
    
    return model