import json 
import os 
import csv
import matplotlib.pyplot as plt
import pandas as pd

def guardar_reporte(y_true, y_pred, prov, modelo, params_dict, ruta="outputs/tables/metricas_globales.csv"):
    print(modelo)
    if modelo in ['stacked', 'bidirectional', 'attention']:
        params_dict=json.dumps(params_dict) 
    elif modelo==True:
        modelo="SARIMAX"
    elif modelo==False:
        modelo="SARIMA"
    datos_fila = {
        "Provincia": prov,
        "Modelo": f"{modelo}",

        "Parametros": params_dict
    }

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    file_exists = os.path.isfile(ruta)
    
    with open(ruta, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=datos_fila.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(datos_fila)
    
    return 

def guardar_complejidad_modelo(provincia,modelo,complejidad,extra_info=None,archivo="outputs/tables/complejidad_modelos.csv"):

    fila = {"Provincia": provincia,"Modelo": modelo,"Complejidad": complejidad}

    if extra_info:
        fila.update(extra_info)

    df_new = pd.DataFrame([fila])

    if os.path.exists(archivo):
        df_old = pd.read_csv(archivo)

        df_old = df_old[
            ~(
                (df_old["Provincia"] == provincia) &
                (df_old["Modelo"] == modelo)
            )
        ]

        df_final = pd.concat([df_old, df_new], ignore_index=True)

    else:
        df_final = df_new

    df_final.to_csv(archivo, index=False)
    
def guardar_detalles_consolidados(df_detallado, nombre_modelo, nombre_provincia, archivo_path="outputs/tables/predicciones.csv"):
    """Guarda o actualiza el archivo CSV usando una estrategia de UPDATE + APPEND"""
    print(nombre_modelo)
    os.makedirs(os.path.dirname(archivo_path), exist_ok=True)
    
    df_save = df_detallado[['Fecha', 'Real', 'Pred']].copy()
    df_save['Provincia'] = nombre_provincia
    df_save['Semana_H'] = (df_save.reset_index(drop=True).index % 4) + 1
    
    df_save.rename(columns={'Pred': nombre_modelo}, inplace=True)
    
    df_save['Fecha'] = pd.to_datetime(df_save['Fecha'])
    
    orden_deseado = [
            'Fecha', 
            'Real', 
            'RandomForest', 
            "SVR",
            "XGBoost",
            'stacked', 
            "bidirectional",
            "attention",
            'SARIMA', 
            'SARIMAX', 
            'Semana_H', 
            'Provincia'
        ]
    
    if os.path.exists(archivo_path):
        df_existente = pd.read_csv(archivo_path)
        df_existente['Fecha'] = pd.to_datetime(df_existente['Fecha'])
        
        idx_cols = ['Fecha', 'Provincia']
        
        df_existente = df_existente.set_index(idx_cols)
        df_save_idx = df_save.set_index(idx_cols)
        
        df_existente.update(df_save_idx)
        print(df_save_idx)
        new_rows = df_save_idx[~df_save_idx.index.isin(df_existente.index)]
        print(new_rows)
        df_final = pd.concat([df_existente, new_rows])
        
        df_final = df_final.reset_index()
        print(nombre_modelo)
        print(df_final)
    else:
        df_final = df_save
        for col in orden_deseado:
            if col not in df_final.columns:
                df_final[col] = pd.NA  
        print("Archivo no encontrado. Inicializando estructura completa por primera vez.")
    
    columnas_existentes = df_final.columns.tolist()
    columnas_finales = [col for col in orden_deseado if col in columnas_existentes]
    columnas_extra = [col for col in columnas_existentes if col not in orden_deseado]
    
    if 'Provincia' in columnas_finales:
        idx_prov = columnas_finales.index('Provincia')
        columnas_finales = columnas_finales[:idx_prov] + columnas_extra + columnas_finales[idx_prov:]
    else:
        columnas_finales = columnas_finales + columnas_extra
        
    df_final = df_final[columnas_finales]
    print(df_final)
    df_final.to_csv(archivo_path, index=False)
    
    
def graficar_resultados(df_detallado, nombre_archivo,nombre_provincia):
    """Genera y guarda el gráfico de comparación."""
    
    ruta_carpeta = os.path.join("outputs", "graficos", nombre_provincia)
    
    os.makedirs(ruta_carpeta, exist_ok=True)
    
    ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
    plt.figure(figsize=(12, 6))
    plt.plot(df_detallado['Fecha'], df_detallado['Real'], label='Real', color='black')
    plt.plot(df_detallado['Fecha'], df_detallado['Pred'], label='Predicción', color='blue', linestyle='--')
    plt.title(' Real vs Predicción')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(ruta_completa, dpi=300)
    plt.close()
    

def guardar_resultados_csv(df_nuevo: pd.DataFrame,ruta_salida: str,subset_duplicates: list):

    if os.path.exists(ruta_salida):
        existente = pd.read_csv(ruta_salida)

        df_final = pd.concat([existente, df_nuevo],ignore_index=True)

        if subset_duplicates is not None:
            df_final = df_final.drop_duplicates(subset=subset_duplicates,keep="last")

    else:
        df_final = df_nuevo.copy()

    df_final.to_csv(ruta_salida, index=False)
    return df_final
