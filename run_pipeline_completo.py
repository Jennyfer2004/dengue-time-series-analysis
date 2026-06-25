import os

from src.preprocessing.integrador_datos import ejecutar_preprocessing
from src.preprocessing.correlaciones import ejecutar_analisis_lags
from src.tuning.tune_sarima import ejecutar_workflow_sarimax
from src.tuning.tune_ml import ejecutar_ml_optimizacion
from src.tuning.tuning_dl import ejecutar_dl_workflow 
from src.preprocessing.prep_SARIMA import cargar_datos_provincia


PROVINCIAS = ["Pinar del Río", "Artemisa", "La Habana", "Mayabeque", "Matanzas", "Cienfuegos", "Villa Clara", "Sancti Spíritus", "Ciego de Ávila", "Camagüey", "Las Tunas", "Granma", "Holguín", "Santiago de Cuba", "Guantánamo", "Isla de la Juventud"]
RUTA_CLIMA = '../../Bases de datos/base_datos.csv'
RUTA_CASOS = '../../Bases de datos/Datos_dengue_comletos.csv'
CARPETA_CLIMA_LAGS = 'data/processed/clima_lags'
CARPETA_CASOS_ACF = 'data/processed/casos_acf'
GEOJSON_PATH="./data/raw/cuba.geojson"
RESULTADOS_PAR="outputs/tables"

def main():
    print(" [FASE 1] Ejecutando limpieza profunda e ingeniería de lags climáticos...")
    df_prep = ejecutar_preprocessing(RUTA_CLIMA, RUTA_CASOS, GEOJSON_PATH, ano_inicial=2014)
    ejecutar_analisis_lags(df_prep)
    
    print("\n[FASE 2] Iniciando optimización masiva de hiperparámetros (Tuning)...")
    for prov in PROVINCIAS:
        if prov !="La Habana":
            continue
        print(f" Optimizando backend para: {prov}")
        
        # SARIMA/X
        try:
            df_limpio = cargar_datos_provincia(prov, CARPETA_CLIMA_LAGS)
            ejecutar_workflow_sarimax(df_limpio, prov, RESULTADOS_PAR+"/sarima", RESULTADOS_PAR+"/Sarima.csv", usar_exog=False, ruta_guardar=RESULTADOS_PAR+"/metricas_globales.csv")
            ejecutar_workflow_sarimax(df_limpio, prov, RESULTADOS_PAR+"/sarimax", RESULTADOS_PAR+"/Sarimax.csv", usar_exog=True, ruta_guardar=RESULTADOS_PAR+"/metricas_globales.csv")
            print("Modelos Sarima ejecutados")
        except Exception as e: print(f"      Error SARIMA en {prov}: {e}")

        # ML
        try: ejecutar_ml_optimizacion(prov, CARPETA_CASOS_ACF, ruta_guardar=RESULTADOS_PAR+"/metricas_globales.csv")
        except Exception as e: print(f"      Error ML en {prov}: {e}")
            
        LSTM
        try: ejecutar_dl_workflow(prov, CARPETA_CLIMA_LAGS, ruta_guardar=RESULTADOS_PAR+"/metricas_globales.csv")
        except Exception as e: print(f"      Error DL en {prov}: {e}")

    plot_heatmap_errores(data_pivot_mape, metric_name="MAPE", save_path="graficos/heatmap_mape.png")

    plot_heatmap_errores(data_pivot_mae, metric_name="MAE", save_path="graficos/heatmap_mae.png")

    plot_prediccion_vs_real(
        df_real=datos_reales_lahabana, 
        df_pred=predicciones_lstm_lahabana, 
        provincia="La Habana", 
        nombre_modelo="Stacked LSTM"
    )

    print("\n=========  PIPELINE DE PREPARACIÓN TERMINADO =========")
    print("Los datos intermedios y parámetros óptimos se han consolidado en el disco.")
    print("Ahora puedes correr 'python main_evaluacion.py'")

if __name__ == "__main__":
    main()
