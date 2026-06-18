import os
import pandas as pd

def cargar_datos_provincia(nombre_provincia, carpeta_entrada):
    """Carga el histórico temporal procesado de la provincia seleccionada."""
    
    nombre_archivo = f'{nombre_provincia}.csv'
    ruta_completa = os.path.join(carpeta_entrada, nombre_archivo)
    
    if not os.path.exists(ruta_completa):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_completa}")
    
    return pd.read_csv(ruta_completa)

def obtener_hiperparametros(nombre_provincia, ruta_params):
    """Parsea el archivo CSV base para extraer las listas de límites en la búsqueda en cuadrícula."""
    
    if not os.path.exists(ruta_params):
        return [0], [1], [0], [0], [0], [0] 

    df_params = pd.read_csv(ruta_params)
    fila = df_params[df_params['Provincia'].str.lower() == nombre_provincia.lower()]
    
    if fila.empty:
        return [0], [0], [1], [0], [0], [1]

    def parse_lista(val):
        s = str(val).strip()
        return [int(x.strip()) for x in s.split(',')] if ',' in s else [int(float(s))]

    return (parse_lista(fila['p'].iloc[0]), parse_lista(fila['q'].iloc[0]), 
            parse_lista(fila['d'].iloc[0]), parse_lista(fila['P'].iloc[0]), 
            parse_lista(fila['Q'].iloc[0]), parse_lista(fila['D'].iloc[0]))