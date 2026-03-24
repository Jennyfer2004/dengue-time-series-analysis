import pandas as pd
import geopandas as gpd
from geovoronoi import voronoi_regions_from_coords
from shapely.ops import nearest_points
import unicodedata

def datos(df_clima,gjson='cuba.geojson'):
    estaciones_unicas = df_clima[['Nombres Estaciones', 'Latitud', 'Longitud', 'Provincias']].drop_duplicates()
    gdf_estaciones = gpd.GeoDataFrame(estaciones_unicas, 
                    geometry=gpd.points_from_xy(estaciones_unicas.Longitud, estaciones_unicas.Latitud))

    provincias_map = gpd.read_file(gjson) 
    return(provincias_map,gdf_estaciones)

def normalizar(t):
    return unicodedata.normalize('NFKD', str(t)).encode('ascii', 'ignore').decode('utf-8').lower().strip()

def main(df_clima,gjson='cuba.geojson'):
    provincias_map,gdf_estaciones=datos(df_clima,gjson='cuba.geojson')
    gdf_estaciones['Prov_Match'] = gdf_estaciones['Provincias'].apply(normalizar)
    provincias_map['Prov_Match'] = provincias_map['province'].apply(normalizar)

    pesos_maestro = {}

    for _, fila_prov in provincias_map.iterrows():
        nombre_real = fila_prov['province']
        id_prov = fila_prov['Prov_Match']
        prov_shape = fila_prov['geometry']
        
        if not prov_shape.is_valid:
            prov_shape = prov_shape.buffer(0)

        # Filtrar estaciones de esta provincia
        estaciones_prov = gdf_estaciones[gdf_estaciones['Prov_Match'] == id_prov].copy().reset_index(drop=True)
        
        if estaciones_prov.empty:
            continue

        puntos_dentro = []
        for p in estaciones_prov.geometry:
            if not p.within(prov_shape):
                # Si el punto está en el mar, lo pegamos a la costa más cercana
                _, p_ajustado = nearest_points(prov_shape, p)
                puntos_dentro.append(p_ajustado)
            else:
                puntos_dentro.append(p)
        
        estaciones_prov.geometry = puntos_dentro

        # Decidir qué cálculo hacer según cuántas estaciones hay DENTRO
        num_estaciones = len(estaciones_prov)

        if num_estaciones == 1:
            nombre_est = estaciones_prov.iloc[0]['Nombres Estaciones']
            pesos_maestro[nombre_est] = 1.0
            print(f"✅ {nombre_real}: 1 estación detectada (100%)")
        
        elif num_estaciones > 1:
            try:
                poly_shapes, pts_indices = voronoi_regions_from_coords(estaciones_prov.geometry, prov_shape)
                area_total = prov_shape.area
                # print(nombre_real,poly_shapes)
                if poly_shapes=={}:
                    peso_igual = 1.0 / num_estaciones
                    for nombre in estaciones_prov['Nombres Estaciones']:
                        pesos_maestro[nombre] = peso_igual
                    # print(f"⚠️ {nombre_real}: Error técnico, repartiendo áreas equitativamente")            
                for i, poly in poly_shapes.items():
                    idx = pts_indices[i][0]
                    nombre_est = estaciones_prov.iloc[idx]['Nombres Estaciones']
                    pesos_maestro[nombre_est] = poly.area / area_total
                    # print(nombre_real,nombre_est,pesos_maestro[nombre_est])
                # print(nombre_real,poly_shapes)

                print(f"✅ {nombre_real}: {num_estaciones} estaciones calculadas con Voronoi")
        
            except Exception as e:
                # Si Voronoi falla por puntos idénticos, repartir 50/50
                print(e)
                # peso_igual = 1.0 / num_estaciones
                # for nombre in estaciones_prov['Nombres Estaciones']:
                #     pesos_maestro[nombre] = peso_igual
                # print(f"⚠️ {nombre_real}: Error técnico, repartiendo áreas equitativamente")
        else:
            print(nombre_real)

    print("\n--- PROCESO COMPLETADO ---")
    print(f"Total de estaciones en el diccionario: {len(pesos_maestro)}")
    return(pesos_maestro)