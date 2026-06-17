import os
import ast
import numpy as np
import pandas as pd
from src.evaluacion.metrics import diebold_mariano_test_global

def calcular_metrica_global(df_m, metrica,columna_modelo):
    """Calcula el valor de error global según la métrica seleccionada."""
   
    y_true = df_m['Real'].values
    y_pred = df_m[columna_modelo].values

    if metrica == 'MAE':
        return np.mean(np.abs(y_true - y_pred))
    elif metrica == 'RMSE':
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    elif metrica == 'MAPE':
        filtro_no_cero = y_true != 0
        if np.sum(filtro_no_cero) > 0:
            return np.mean(np.abs((y_true[filtro_no_cero] - y_pred[filtro_no_cero]) / y_true[filtro_no_cero])) * 100
        return np.nan
    return np.nan


def evaluar_parsimonia_provincia(df_prov, comp_prov, modelos_ordenados, tipo_metrica, min_muestras=20, max_diff_relativa=10.0):
    """Aplica la selección por parsimonia usando Diebold-Mariano para una métrica específica."""
   
    modelos_validos = [m for m in modelos_ordenados if m in df_prov.columns and df_prov[m].notna().sum() > min_muestras]
    
    if 'Real' not in df_prov.columns or len(modelos_validos) < 2:
        return None, None
        
    df_prov = df_prov.dropna(subset=['Real'])
    
    errores_globales = {}
    for m in modelos_validos:
        df_m = df_prov.dropna(subset=[m])
        if len(df_m) >= min_muestras:
            err = calcular_metrica_global(df_m, tipo_metrica,m)
            if not np.isnan(err):
                errores_globales[m] = err
                
    if not errores_globales:
        return None, None
        
    modelo_top = min(errores_globales, key=errores_globales.get)
    modelo_seleccionado = modelo_top
    p_val_final = 1.0
    diff_relativa_final = 0.0
    
    for m in modelos_validos:
        if m == modelo_seleccionado or m not in comp_prov or modelo_seleccionado not in comp_prov:
            continue
            
        if comp_prov[m] < comp_prov[modelo_seleccionado]:
            df_pair = df_prov.dropna(subset=[m, modelo_seleccionado])
            if len(df_pair) >= min_muestras:
                _, p_val = diebold_mariano_test_global(
                    df_pair['Real'].values, df_pair[m].values, df_pair[modelo_seleccionado].values, h=4)
                
                if p_val > 0.05:
                    error_simple = errores_globales[m]
                    error_mejor = errores_globales[modelo_seleccionado]
                    diferencia_relativa = (abs(error_simple - error_mejor) / error_mejor) * 100
                    
                    if diferencia_relativa <= max_diff_relativa:
                        modelo_seleccionado = m
                        p_val_final = p_val
                        diff_relativa_final = diferencia_relativa
    print(comp_prov)
    
    res_global = {
        'Provincia': df_prov['Provincia'].iloc[0],
        f'Ganador_{tipo_metrica}_Global': modelo_top,
        f'Complejidad_Ganador_{tipo_metrica}': comp_prov.get(modelo_top, np.nan),
        'P_Value_DM': round(p_val_final, 4),
        'Diferencia_Relativa_%': round(diff_relativa_final, 2),
        'Modelo_Elegido_Parsimonia': modelo_seleccionado,
        'Complejidad_Modelo_Elegido': comp_prov.get(modelo_seleccionado, np.nan),
        f'{tipo_metrica}_Global_Minimo': round(errores_globales[modelo_top], 3),
        'Justificación': f"DM p>0.05. Se seleccionó el modelo con menor complejidad ({comp_prov.get(modelo_seleccionado, np.nan)})." 
                         if modelo_top != modelo_seleccionado else "Superior global absoluto."
    }
    
    res_comp = {
        "Provincia": df_prov['Provincia'].iloc[0],
        "Mejor_Modelo": modelo_top,
        "Error_Mejor": round(errores_globales[modelo_top], 3),
        "Modelo_Parsimonioso": modelo_seleccionado,
        "Error_Parsimonioso": round(errores_globales[modelo_seleccionado], 3),
        "Metrica": tipo_metrica,
        "Diferencia_%": round((abs(errores_globales[modelo_seleccionado] - errores_globales[modelo_top]) / errores_globales[modelo_top]) * 100, 2)
    }
    
    return res_global, res_comp

def ejecutar_analisis_parsimonia_completo(ruta_predicciones, ruta_complejidad, carpeta_salida="outputs/tables",metricas = ['MAE', 'MAPE', 'RMSE'],    
                                        modelos_ordenados = ['SARIMA', 'SARIMAX', 'SVR', 'RandomForest', 'XGBoost', 'stacked', 'bidirectional', 'attention']):
   
    os.makedirs(carpeta_salida, exist_ok=True)
    
    df = pd.read_csv(ruta_predicciones)
    
    df_complejidad = pd.read_csv(ruta_complejidad)
    provincias = df['Provincia'].unique()
    
    comparacion_total_parsimonia = []
    
    for metrica in metricas:
        resultados_globales = []
        
        for prov in provincias:
            df_prov = df[df['Provincia'] == prov].copy()
            comp_prov = df_complejidad[df_complejidad["Provincia"] == prov].set_index("Modelo")["Complejidad"].to_dict()
            
            res_global, res_comp = evaluar_parsimonia_provincia(df_prov, comp_prov, modelos_ordenados, tipo_metrica=metrica)
            
            if res_global:
                resultados_globales.append(res_global)
                comparacion_total_parsimonia.append(res_comp)
                
        df_reporte = pd.DataFrame(resultados_globales)
        ruta_csv = os.path.join(carpeta_salida, f'resultado_parsimonia_global_{metrica.lower()}.csv')
        df_reporte.to_csv(ruta_csv, index=False)
        print(f"\n📊 Reporte Estandarizado de {metrica} guardado en: {ruta_csv}")
        print(df_reporte.to_string(index=False))
        print("-" * 80)

    df_comparacion = pd.DataFrame(comparacion_total_parsimonia)
    df_comparacion = df_comparacion.sort_values("Diferencia_%", ascending=True)
    
    ruta_comparacion = os.path.join(carpeta_salida, "comparacion_parsimonia_consolidada.csv")
    df_comparacion.to_csv(ruta_comparacion, index=False)
    print(f"✅ Análisis de parsimonia total concluido con éxito. Archivo maestro guardado en: {ruta_comparacion}")


if __name__ == "__main__":
    ejecutar_analisis_parsimonia_completo(ruta_predicciones="outputs/tables/predicciones.csv",ruta_complejidad="outputs/tables/complejidad_modelos.csv")