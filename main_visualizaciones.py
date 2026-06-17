import os
import glob
import pandas as pd
import geopandas as gpd

from src.visualizaciones import guardar_serie_tiempo, guardar_barra_metricas_global,guardar_grafico_metrica_individual,guardar_lineas_evolucion_individual,guardar_mapa_error_individual,guardar_serie_temporal_provincia,guardar_heatmap_metrica_global

metricas_config = {
        "MAE": ["MAE (S1)", "MAE (S2)", "MAE (S3)", "MAE (S4)"],
        "RMSE": ["RMSE (S1)", "RMSE (S2)", "RMSE (S3)", "RMSE (S4)"],
        "MAPE": ["MAPE (S1)", "MAPE (S2)", "MAPE (S3)", "MAPE (S4)"]
    }
provincias = [
        "Pinar del Río", "Artemisa", "La Habana", "Mayabeque", "Matanzas", 
        "Cienfuegos", "Villa Clara", "Sancti Spíritus", "Ciego de Ávila", 
        "Camagüey", "Las Tunas", "Granma", "Holguín", "Santiago de Cuba", 
        "Guantánamo", "Isla de la Juventud"
    ]
ruta_excels_salida = "outputs/tables/resultados_metricas.csv"
ruta_excels_predicciones = "outputs/tables/predicciones.csv"
ruta_geojson = os.path.join("data/raw/cuba.geojson")

paleta_colores = {
        "attention": "#2b5c8f",
        "bidirectional": "#4f81bd",
        "stacked": "#91a7c0",
        "SARIMA": "#7f7f7f",
        "SARIMAX": "#a8b6c4",
        "SVR": "#d9d9d9",
        "XGBoost": "#6ba85b",
        "RandomForest": "#f4a582",
    }
carpeta_provincias = "dataframes_provincias" 

def main():
    print("🚀 Iniciando exportación automatizada de gráficos para la tesis...")
    
    rutas_csv = glob.glob(os.path.join(carpeta_provincias, "*.csv"))
    
    if not rutas_csv:
        print(f"⚠️ No se encontraron CSVs en '{carpeta_provincias}'. Verifica la ruta.")
    else:
        for ruta in rutas_csv:
            nombre_provincia = os.path.splitext(os.path.basename(ruta))[0].replace("_", " ")
            df_provincia = pd.read_csv(ruta)

            guardar_serie_tiempo(df_provincia, nombre_provincia)


    
    if os.path.exists(ruta_excels_salida):
        df_metricas = pd.read_csv(ruta_excels_salida)

        for nombre_metrica, columnas in metricas_config.items():
            guardar_barra_metricas_global(
                df=df_metricas,
                metrica_nombre=nombre_metrica,
                columnas_escenarios=columnas
            )
            
    else:
        print(f"⚠️ No se encontró el archivo de métricas en '{ruta_excels_salida}'")
        
    if not os.path.exists(ruta_geojson):
        print("❌ Falta el archivo cuba.geojson).")

        
    mapa_cuba = gpd.read_file(ruta_geojson)
    correcciones = {
        'Guantanmo': 'Guantánamo',
        'Sancti Spiritus': 'Sancti Spíritus',
        'Ciego de Avila': 'Ciego de Ávila'
    }
    mapa_cuba['province'] = mapa_cuba['province'].replace(correcciones)
    df = pd.read_csv(ruta_excels_salida)
    modelos_presentes = df["Modelo"].unique()
    
    
    df_predicciones = pd.read_csv(ruta_excels_predicciones)

    for provincia_sel in provincias:
        
        df_provincia = df_predicciones[df_predicciones["Provincia"] == provincia_sel]
        
        if df_provincia.empty:
            continue    
        
        guardar_serie_temporal_provincia(
            df_provincia=df_provincia,
            nombre_provincia=provincia_sel
        )
        
        df_provincia = df[df["Provincia"] == provincia_sel]
        
        if df_provincia.empty:
            continue
        
        for nombre_metrica, columnas in metricas_config.items():
            
            guardar_grafico_metrica_individual(
                    df_provincia=df_provincia,
                    nombre_provincia=provincia_sel,
                    metrica_nombre=nombre_metrica,
                    columnas_escenarios=columnas,
                )

            guardar_lineas_evolucion_individual(
                    df_provincia=df_provincia,
                    nombre_provincia=provincia_sel,
                    metrica_nombre=nombre_metrica,
                    columnas_escenarios=columnas,
                    paleta_colores=paleta_colores
                    )
    for modelo_sel in modelos_presentes:

        df_modelo = df[df["Modelo"] == modelo_sel]
        
        for metrica_nombre, columnas in metricas_config.items():
            guardar_mapa_error_individual(
                mapa_cuba=mapa_cuba,
                df_modelo=df_modelo,
                modelo_nombre=modelo_sel,
                metrica_nombre=metrica_nombre,
                columnas_escenarios=columnas
            )
        
    for nombre_metrica, columnas in metricas_config.items():
        guardar_heatmap_metrica_global(
            df_completo=df,
            metrica_nombre=nombre_metrica,
            columnas_escenarios=columnas
        )
if __name__ == "__main__":
    main()