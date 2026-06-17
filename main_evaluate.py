import os
import sys
import pandas as pd

from src.evaluacion.rolling_window import main_proceso_ml_provincia, main_proceso_dl_provincia,main_proceso_provincia
from src.evaluacion.parsimonia import ejecutar_analisis_parsimonia_completo 

PROVINCIAS = [
    "Pinar del Río", "Artemisa", "La Habana", "Mayabeque", "Matanzas", 
    "Cienfuegos", "Villa Clara", "Sancti Spíritus", "Ciego de Ávila",
    "Camagüey", "Las Tunas", "Granma", "Holguín", "Santiago de Cuba", 
    "Guantánamo", "Isla de La Juventud"
]

CARPETA_DATOS = "data/processed"
RUTA_PARAMS = "outputs/tables/metricas_globales.csv"
RUTA_REPORTE_FINAL = "outputs/tables/resultados_metricas.csv"
RUTA_REPORTE_FINAL_PARSIMONIA = "outputs/tables/resultado_parsimonia.csv" 
RUTA_REPORTE_FINAL_PREDICCIONES= "outputs/tables/predicciones.csv"
RUTA_COMPLEJIDAD = "outputs/tables/complejidad_modelos.csv"


def ejecutar_experimento_global():
    print("======  INICIANDO EVALUACIÓN GLOBAL DE MODELOS DE TESIS ======\n")
    for provincia in PROVINCIAS:
        if provincia!="La Habana":
            continue
        print(f" PROCESANDO PROVINCIA: {provincia}")
    
        # Modelo SARIMA
        main_proceso_provincia(
            nombre_provincia=provincia,
            carpeta=CARPETA_DATOS + "/clima_lags",
            ruta_parametros=RUTA_PARAMS,
            ruta_salida_excel=RUTA_REPORTE_FINAL,
            usar_exog=False,
            n_test_weeks=49,
            horizonte=4
        )
        
        # Modelo SARIMAX
        main_proceso_provincia(
            nombre_provincia=provincia,
            carpeta=CARPETA_DATOS + "/clima_lags",
            ruta_parametros=RUTA_PARAMS,
            ruta_salida_excel=RUTA_REPORTE_FINAL,
            usar_exog=True,
            n_test_weeks=49,
            horizonte=4
        )
                
        # # Modelos de Machine Learning
        # main_proceso_ml_provincia(
        #     nombre_provincia=provincia,
        #     carpeta_in=CARPETA_DATOS + "/casos_acf",
        #     ruta_params=RUTA_PARAMS,
        #     ruta_salida=RUTA_REPORTE_FINAL,
        #     modelos_a_evaluar=["SVR", "RandomForest", "XGBoost"],
        #     n_test_weeks=49,
        #     semanas=4
        # )
        
        # Modelos de Deep Learning
        # main_proceso_dl_provincia(
        #     nombre_provincia=provincia,
        #     carpeta=CARPETA_DATOS + "/clima_lags",
        #     ruta_parametros=RUTA_PARAMS,
        #     ruta_salida_excel=RUTA_REPORTE_FINAL,
        #     modelo_dl=["stacked", "bidirectional", "attention"],
        #     n_test_weeks=49,
        #     horizonte=4
        # )
        
        ejecutar_analisis_parsimonia_completo(
            ruta_predicciones=RUTA_REPORTE_FINAL_PREDICCIONES, 
            ruta_complejidad=RUTA_COMPLEJIDAD,
            carpeta_salida="outputs/tables"
        )
    print("\n Experimento completado. Métricas guardadas de forma consistente.")

if __name__ == "__main__":
    ejecutar_experimento_global()