import pandas as pd
import numpy as np

def preparar_fecha(df_provincial):
    """Prepara la columna de fecha a partir de Año y Mes"""
    
    df = df_provincial.copy()
    df['Fecha'] = pd.to_datetime(df[['Año', 'Mes']].rename(columns={'Año': 'year', 'Mes': 'month'}) .assign(day=1) )
    
    df = df.sort_values(['Provincias', 'Fecha'])
    df.set_index('Fecha', inplace=True)
    return df

def interpolar_temperatura_humedad(df_semanal):
    """Interpola valores faltantes de temperatura y humedad"""
    
    df_semanal['Temp_Med'] = df_semanal['Temp_Med'].interpolate(method='time', limit_direction='both')
    
    df_semanal['Hum_Rel'] = df_semanal['Hum_Rel'].interpolate(method='time', limit_direction='both')
    
    return df_semanal

def distribuir_precipitacion_mensual(df_semanal):
    """Distribuye la precipitación mensual entre las semanas del mes"""
    
    df_semanal['Año_temp'] = df_semanal.index.year
    df_semanal['Mes_temp'] = df_semanal.index.month
    
    counts = df_semanal.groupby(['Año_temp', 'Mes_temp'])['Precip'].transform('count')
    df_semanal['Precip'] = df_semanal['Precip'] / counts
    
    return df_semanal.drop(columns=['Año_temp', 'Mes_temp'])

def transformar_provincia_a_semanal(group):
    """Transforma datos mensuales de una provincia a frecuencia semanal"""

    semanal = group.resample('W-MON').agg({
        'Temp_Med': 'mean',
        'Hum_Rel': 'mean',
        'Precip': 'first'
    })
    
    semanal = interpolar_temperatura_humedad(semanal)
    semanal['Precip'] = semanal['Precip'].ffill()
    semanal = distribuir_precipitacion_mensual(semanal)
    
    return semanal

def agregar_año_mes(df_semanal_final):
    """Agrega columnas de Año y Mes a partir de la fecha"""
    
    df_semanal_final['Año'] = df_semanal_final['Fecha'].dt.year
    df_semanal_final['Mes'] = df_semanal_final['Fecha'].dt.month
    
    return df_semanal_final

def semanal(df_provincial):
    """Función principal que orquesta la transformación a datos semanales."""

    df_preparado = preparar_fecha(df_provincial)
    
    df_semanal = df_preparado.groupby('Provincias').apply(
        transformar_provincia_a_semanal
    )
    
    df_semanal = df_semanal.reset_index()
    df_semanal = agregar_año_mes(df_semanal)
    
    return df_semanal
