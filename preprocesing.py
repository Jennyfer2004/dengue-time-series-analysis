import pandas as pd
import os 

from calculando_voroni import generar_pesos_maestro
from transformar_a_semanal import semanal 

# ---  UTILIDADES DE LIMPIEZA ---
def normalizar_nombre_provincia(nombre):
    """
    Normaliza el nombre de la provincia para hacer coincidir entre datasets
    """
    if pd.isna(nombre):
        return nombre
    
    correcciones = {
        'Pinar Del Río': 'Pinar del Río',
        'Isla De La Juventud': 'Isla de la Juventud',
        # 'Ciego De Avila': 'Ciego de Ávila',
        'Santiago De Cuba': 'Santiago de Cuba',
        'Guantanmo':'Guantánamo',
        'Ciego De Ávila':'Ciego de Ávila',
        'Sancti Spiritus':'Sancti Spíritus',
    }
    
    if nombre in correcciones:
        return correcciones[nombre]
    
    return ' '.join([word.capitalize() for word in nombre.split()])

# --- LÓGICA DE CARGA Y PROCESAMIENTO ---
def load_data(ruta_clima,ruta_casos,ano):
    df_clima = pd.read_csv(ruta_clima)
    df_clima = df_clima[df_clima['Año'] >= ano]
    df_dengue= pd.read_csv(ruta_casos)
    return df_clima,df_dengue


def procesar_clima_provincial(df_clima, geojson_path):

    pesos_maestro = generar_pesos_maestro(df_clima, gjson='cuba.geojson')
    
    variables = ['Temperatura med', 'Humedad Relat', 'Precipitaciones']
    for var in variables:
        df_clima[f'{var}_ponderada'] = df_clima[var] * df_clima['Nombres Estaciones'].map(pesos_maestro)

    df_prov = df_clima.groupby(['Provincias', 'Año', 'Mes'])[ 
        [f'{v}_ponderada' for v in variables] 
    ].sum().reset_index()
    
    df_prov.columns = ['Provincias', 'Año', 'Mes', 'Temp_Med', 'Hum_Rel', 'Precip']
    
    df_semanal = semanal(df_prov)
    
    return df_semanal.rename(columns={
        'Temp_Med': 'Temperatura med',
        'Hum_Rel': 'Humedad Relat',
        'Precip': 'Precipitaciones'
    })
    
def procesar_casos(df):

    df['Fecha'] = pd.to_datetime(
        df['Año'].astype(str) + df['Semana Estadística'].astype(str) + '1', 
        format='%Y%W%w'
    )
    
    id_vars = ['Año', 'Semana Estadística', 'Fecha']
    value_vars = [col for col in df.columns if col not in id_vars and col != 'Unnamed: 0']

    df_largo = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                       var_name='Provincias', value_name='Casos')
    
    df_largo['Provincias'] = df_largo['Provincias'].apply(normalizar_nombre_provincia)
    return df_largo

# --- INTEGRACIÓN Y SALIDA ---
def exportar_por_provincias(df_final, carpeta='dataframes_provincias'):
    os.makedirs(carpeta, exist_ok=True)
    for provincia in df_final['Provincias'].unique():
        df_prov = df_final[df_final['Provincias'] == provincia].dropna()
        if not df_prov.empty:
            ruta = os.path.join(carpeta, f'{provincia.replace(" ", "_")}.csv')
            df_prov.to_csv(ruta, index=False)
            
def ejecutar_preprocessing(ruta_clima, ruta_casos, ano_inicial=2014,ano_corte=2022):

    raw_clima, raw_casos = load_data(ruta_clima, ruta_casos, ano_inicial)
    
    df_clima_ready = procesar_clima_provincial(raw_clima, 'cuba.geojson')
    df_casos_ready = procesar_casos(raw_casos)
    
    df_completo = pd.merge(df_casos_ready, df_clima_ready, on=['Fecha', 'Provincias',"Año"], how='left')
    
    df_historico = df_completo[df_completo["Año"] <= ano_corte].copy()
    
    # 5. Exportar
    exportar_por_provincias(df_historico)
    
    return df_historico