import pandas as pd
import geopandas as gpd
from geovoronoi import voronoi_regions_from_coords
from shapely.ops import nearest_points
import unicodedata
import numpy as np

def normalizar_texto(texto):
    """Limpia y estandariza cadenas de texto."""
    
    if not texto: return ""
    return unicodedata.normalize('NFKD', str(texto)).encode('ascii', 'ignore').decode('utf-8').lower().strip()

def cargar_datos(df_clima, gjson_path='cuba.geojson'):
    """Carga y prepara los GeoDataFrames iniciales."""
    
    estaciones_unicas = df_clima[['Nombres Estaciones', 'Latitud', 'Longitud', 'Provincias']].drop_duplicates()
    gdf_estaciones = gpd.GeoDataFrame( estaciones_unicas,  geometry=gpd.points_from_xy(estaciones_unicas.Longitud, estaciones_unicas.Latitud),crs="EPSG:4326")
    
    provincias_map = gpd.read_file(gjson_path)
    return provincias_map, gdf_estaciones

def ajustar_puntos_a_poligono(gdf_puntos, poligono):
    """Asegura que todos los puntos estén dentro del polígono (ajusta a la costa si no)."""
    
    puntos_ajustados = []
    
    for p in gdf_puntos.geometry:
        
        if not p.within(poligono):
            _, p_ajustado = nearest_points(poligono, p)
            puntos_ajustados.append(p_ajustado)
            
        else:
            puntos_ajustados.append(p)
    return puntos_ajustados


def calcular_pesos_voronoi(gdf_estaciones, prov_shape):
    """Calcula los pesos de cada estacion"""
    
    pesos = {}
    
    prov_shape = prov_shape.buffer(0.00001) if not prov_shape.is_valid else prov_shape
    
    gdf_estaciones = gdf_estaciones.drop_duplicates(subset=['geometry']).copy()
    
    num_estaciones = len(gdf_estaciones)
    if num_estaciones == 1:
        return {gdf_estaciones.iloc[0]['Nombres Estaciones']: 1.0}

    try:
        prov_projected = gpd.GeoSeries([prov_shape], crs="EPSG:4326").to_crs("EPSG:32617").iloc[0]
        estaciones_projected = gdf_estaciones.to_crs("EPSG:32617")
        
        poly_shapes, pts_indices = voronoi_regions_from_coords(estaciones_projected.geometry, prov_projected)
        
        area_total = prov_projected.area
        for i, poly in poly_shapes.items():
            idx = pts_indices[i][0]
            nombre_est = estaciones_projected.iloc[idx]['Nombres Estaciones']
            pesos[nombre_est] = poly.area / area_total
            
    except Exception as e:

        try:
            coords = np.array([(p.x + np.random.uniform(-0.01, 0.01), 
                                p.y + np.random.uniform(-0.01, 0.01)) 
                               for p in estaciones_projected.geometry])
            
            poly_shapes, pts_indices = voronoi_regions_from_coords(coords, prov_projected)
            
            area_total = prov_projected.area
            for i, poly in poly_shapes.items():
                idx = pts_indices[i][0]
                nombre_est = estaciones_projected.iloc[idx]['Nombres Estaciones']
                pesos[nombre_est] = poly.area / area_total
        except:
            peso_igual = 1.0 / num_estaciones
            for nombre in gdf_estaciones['Nombres Estaciones']:
                pesos[nombre] = peso_igual
            
    return pesos

def generar_pesos_maestro(df_clima, gjson='cuba.geojson'):
    """Genera los pesos por estacion"""
    
    provincias_map, gdf_estaciones = cargar_datos(df_clima, gjson)
    provincias_map.loc[provincias_map['province'] == 'Guantanmo', 'province'] = 'Guantánamo'

    gdf_estaciones['Prov_Match'] = gdf_estaciones['Provincias'].apply(normalizar_texto)
    provincias_map['Prov_Match'] = provincias_map['province'].apply(normalizar_texto)

    pesos_maestro = {}

    for _, fila_prov in provincias_map.iterrows():
        id_prov = fila_prov['Prov_Match']
        nombre_real = fila_prov['province']
        prov_shape = fila_prov['geometry'] if fila_prov['geometry'].is_valid else fila_prov['geometry'].buffer(0)
        
        estaciones_prov = gdf_estaciones[gdf_estaciones['Prov_Match'] == id_prov].copy()
        
        if estaciones_prov.empty:
            continue

        estaciones_prov.geometry = ajustar_puntos_a_poligono(estaciones_prov, prov_shape)

        pesos_prov = calcular_pesos_voronoi(estaciones_prov, prov_shape)
        
        if not pesos_prov:
            n = len(estaciones_prov)
            pesos_prov = {row['Nombres Estaciones']: 1.0/n for _, row in estaciones_prov.iterrows()}
        
        pesos_maestro.update(pesos_prov)

    return pesos_maestro
