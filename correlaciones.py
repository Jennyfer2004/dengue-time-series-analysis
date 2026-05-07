import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf
import csv
import os
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg') 

# --- MOTOR ESTADÍSTICO ---
def calcular_ccf_validado(serie_y, serie_x, max_lag=26, alpha=0.05):
    """
    Motor principal para encontrar lags significativos usando diferenciación
    para asegurar estacionariedad y validación p-value.
    """
    dy = serie_y.diff().dropna()
    dx = serie_x.diff().dropna()
    
    df_diff = pd.DataFrame({'dy': dy, 'dx': dx}).dropna()
    
    ccf_values = ccf(df_diff['dx'], df_diff['dy'], adjusted=False)[:max_lag+1]
    
    lags_validados = []
    for lag in range(1, len(ccf_values)): 
        r_ccf = ccf_values[lag]
        
        y_val = df_diff['dy'].iloc[lag:]
        x_val = df_diff['dx'].shift(lag).dropna()
        
        common_idx = y_val.index.intersection(x_val.index)
        
        if len(common_idx) > 5:
            _, p_val = pearsonr(x_val.loc[common_idx], y_val.loc[common_idx])
            
            if abs(r_ccf) >= 0.05 and p_val < alpha:
                lags_validados.append({
                    'lag': lag,
                    'correlacion': r_ccf,
                    'p_valor': p_val
                })
    return lags_validados

def calcular_acf_validado(serie, max_lag=20, alpha=0.05):
    """Identifica la influencia del pasado de la propia serie (Autocorrelación)."""
    dy = serie.diff().dropna()

    return calcular_ccf_validado(dy, dy, max_lag, alpha)


# --- VISUALIZACIÓN --
def exportar_grafico_ccf(serie1, serie2, nombre1, nombre2, output_dir, max_lag=26):
    """Genera y guarda el gráfico de bastones de la CCF."""
    df_temp = pd.DataFrame({nombre1: serie1, nombre2: serie2}).dropna()
    ccf_values = ccf(df_temp[nombre2], df_temp[nombre1], adjusted=False)[:max_lag+1]
    lags = np.arange(len(ccf_values))
    
    conf_level = 1.96 / np.sqrt(len(df_temp))
    
    plt.figure(figsize=(10, 5))
    plt.stem(lags, ccf_values, basefmt=" ")
    plt.axhline(conf_level, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(-conf_level, color='blue', linestyle='--', alpha=0.5)
    
    idx_max = np.argmax(np.abs(ccf_values))
    plt.plot(lags[idx_max], ccf_values[idx_max], 'ro')
    
    plt.title(f'CCF: {nombre1} vs {nombre2}')
    plt.xlabel('Lag (semanas)')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'CCF_{nombre2}.png'))
    plt.close()

# --- GESTIÓN DE DATOS Y ARCHIVOS ---
def generar_reporte_lags(df, provincias, variables_clima, ruta_csv):
    """Calcula todos los lags y genera el archivo CSV de resumen."""
    lags_ccf_master = {}
    lags_acf_master = {}
    
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)

    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Provincia', 'Variable', 'Lags_Cruzados', 'Lags_Auto'])

        for prov in provincias:
            df_p = df[df['Provincias'] == prov].sort_index()
            lags_ccf_master[prov] = {}
            
            acf_res = calcular_acf_validado(df_p['Casos'])
            lags_acf_master[prov] = [l['lag'] for l in acf_res]

            for var in variables_clima:
                ccf_res = calcular_ccf_validado(df_p['Casos'], df_p[var])
                lags_ccf_master[prov][var] = [l['lag'] for l in ccf_res]
                
                writer.writerow([
                    prov, var, 
                    str(lags_ccf_master[prov][var]), 
                    str(lags_acf_master[prov])
                ])
                
    return lags_ccf_master, lags_acf_master

def guardar_datasets_con_lags(df, lags_ccf, lags_acf, carpeta_base):
    """Crea los archivos .csv finales con las columnas de lags ya desplazadas."""
    
    ruta_clima = os.path.join(carpeta_base, 'clima_lags')
    ruta_casos = os.path.join(carpeta_base, 'casos_acf')
    
    os.makedirs(ruta_clima, exist_ok=True)
    os.makedirs(ruta_casos, exist_ok=True)
    
    for prov in lags_ccf.keys():
        df_p = df[df['Provincias'] == prov].copy()
        
        df_c = df_p.copy()
        for var, lista_lags in lags_ccf[prov].items():
            for lag in lista_lags:
                df_c[f'{var}_lag{lag}'] = df_p[var].shift(lag)
        
        df_c.dropna().to_csv(os.path.join(ruta_clima, f'{prov}.csv'), index=False)

        df_a = df_c.copy()
        for lag in lags_acf[prov]:
            df_a[f'Casos_lag{lag}'] = df_c['Casos'].shift(lag)
            
        df_a.dropna().to_csv(os.path.join(ruta_casos, f'{prov}.csv'), index=False)

# --- FUNCIÓN MAESTRA ---
def ejecutar_analisis_lags(df, folder_output='outputs'):
    """Orquestador de todo el análisis de retardos."""
   
    provincias = df['Provincias'].unique()
    variables = ['Precipitaciones', 'Temperatura med', 'Humedad Relat']
    
    for prov in provincias:
        df_p = df[df['Provincias'] == prov].sort_index()

        for var in variables:
            exportar_grafico_ccf(
                df_p['Casos'].diff().dropna(), 
                df_p[var].diff().dropna(), 
                'Casos', var, 
                os.path.join(folder_output, 'graficos', prov)
            )

    lags_ccf, lags_acf = generar_reporte_lags(
        df, provincias, variables, 
        os.path.join(folder_output, 'resumen_lags.csv')
    )
    
    guardar_datasets_con_lags(df, lags_ccf, lags_acf, folder_output)
    
    print(f"Análisis completado. Resultados en: {folder_output}")
    
    