import pandas as pd
import numpy as np

def semanal(df_provincial):
    df_provincial['Fecha'] = pd.to_datetime(
        df_provincial[['Año', 'Mes']]
        .rename(columns={'Año': 'year', 'Mes': 'month'}) 
        .assign(day=1) 
    )

    df_provincial = df_provincial.sort_values(['Provincia', 'Fecha'])
    df_provincial.set_index('Fecha', inplace=True)
    def transformar_a_semanal(group):

        semanal = group.resample('W-MON').agg({
            'Temp_Med': 'mean',
            'Hum_Rel': 'mean',
            'Precip': 'first' 
        })
        
        semanal['Temp_Med'] = semanal['Temp_Med'].interpolate(method='time', limit_direction='both')
        semanal['Hum_Rel'] = semanal['Hum_Rel'].interpolate(method='time', limit_direction='both')
        
        semanal['Precip'] = semanal['Precip'].ffill()
        
        semanal['Año_temp'] = semanal.index.year
        semanal['Mes_temp'] = semanal.index.month
        
        counts = semanal.groupby(['Año_temp', 'Mes_temp'])['Precip'].transform('count')
        
        semanal['Precip'] = semanal['Precip'] / counts
        
        return semanal.drop(columns=['Año_temp', 'Mes_temp'])

    df_semanal_final = df_provincial.groupby('Provincia').apply(transformar_a_semanal)

    df_semanal_final = df_semanal_final.reset_index()
    df_semanal_final['Año'] = df_semanal_final['Fecha'].dt.year
    df_semanal_final['Mes'] = df_semanal_final['Fecha'].dt.month
    return df_semanal_final