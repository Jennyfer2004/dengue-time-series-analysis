import os
import ast
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
import tensorflow as tf

from src.modelos.lstm_models import build_lstm_model
from src.preprocessing.prep_LSTM import crear_secuencias_3d
from src.utils import guardar_complejidad_modelo, guardar_detalles_consolidados
from src.evaluacion.metrics import procesar_metricas_finales
from src.modelos.sarima_parametros import entrenar_instancia_sarimax
from src.preprocessing.prep_SARIMA import cargar_datos_provincia
from src.preprocessing.prep_ML import preparar_datos_ml
from src.modelos.ML_models import obtener_modelo_configurado
from src.utils import guardar_resultados_csv

def ejecutar_sarimax_rolling(df, n_test_weeks, horizonte,nombre_provincia,order, seasonal_order, target, exog_cols,usar_exog):
    """Ejecuta el modelo SARIMA/X con ventana deslizante."""

    todas_metricas_semanales = []
    inicio_test = len(df) - n_test_weeks

    for i in range(inicio_test, len(df) - horizonte + 1, horizonte):
        train = df.iloc[:i]
        test = df.iloc[i : i + horizonte]
        
        try:
            model_fit = entrenar_instancia_sarimax(train[target], train[exog_cols], order, seasonal_order)
            
            n_params = len(model_fit.params)

            guardar_complejidad_modelo(provincia=nombre_provincia,modelo="SARIMAX" if usar_exog else "SARIMA",complejidad=n_params,extra_info={"Num_Coeficientes": n_params})

            y_pred = model_fit.get_forecast(steps=horizonte, exog=test[exog_cols]).predicted_mean
            y_true = test[target]
        
            for t in range(horizonte):
                todas_metricas_semanales.append({'Fecha': y_true.index[t],'Real': y_true.iloc[t],'Pred':max(0, y_pred.iloc[t]),'Error_Abs': abs(y_true.iloc[t] - y_pred.iloc[t]) })
       
        except Exception as e:
            print(f"Error en periodo {i}: {e}")
            
    return pd.DataFrame(todas_metricas_semanales)

def main_proceso_provincia(nombre_provincia, carpeta, ruta_parametros, ruta_salida_excel,usar_exog=True,n_test_weeks=49, horizonte=4,target='Casos'):
    """ SARIMA/X por provincia"""
    
    df = cargar_datos_provincia(nombre_provincia, carpeta)
    resumen_params = pd.read_csv(ruta_parametros)
    exog_cols = [c for c in df.columns if c not in [target, 'Fecha', 'Provincias', 'Unnamed: 0',"Mes","Año"]]

    modelo_busqueda = "SARIMAX" if usar_exog else "SARIMA"

    params_str = resumen_params[(resumen_params["Provincia"] == nombre_provincia) & (resumen_params["Modelo"] == "SARIMAX")]["Parametros"].iloc[0]
    order, seasonal_order = ast.literal_eval(params_str)

    df = df.sort_values('Fecha').set_index('Fecha')
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')].asfreq('W-MON').ffill()

    df_detallado = ejecutar_sarimax_rolling(df, n_test_weeks, horizonte,nombre_provincia ,order, seasonal_order, target, exog_cols,usar_exog)

    guardar_detalles_consolidados(df_detallado, modelo_busqueda, nombre_provincia)
   
    df_metricas = procesar_metricas_finales(df_detallado, nombre_provincia,usar_exog,df_detallado)
    
    try:
        existente = pd.read_csv(ruta_salida_excel)
        df_final = pd.concat([existente, df_metricas], ignore_index=True)
    
    except FileNotFoundError:
        df_final = df_metricas

    guardar_resultados_csv(df_metricas,ruta_salida_excel,subset_duplicates=["Provincia", "Modelo"])
    
    print(f"Proceso completado para {nombre_provincia}. Métricas guardadas.")
    return df_final

def ejecutar_ml_rolling(df_original, df_diff, n_test_weeks, horizonte,nombre_provincia, modelo_nombre, params, target, cols_a_excluir):
    """Modelos de ML por provincias"""
       
    todas_predicciones = []
    inicio_test = len(df_original) - n_test_weeks
    
    df_work = df_diff.copy()
    
    if 'Semana_H' in df_work.columns:

        week_rad = (df_work['Semana_H'] - 1) * (2 * np.pi / 4)
        df_work['Semana_H_Sin'] = np.sin(week_rad)
        df_work['Semana_H_Cos'] = np.cos(week_rad)

        cols_a_excluir = list(cols_a_excluir) + ['Semana_H'] 

    y_diff = df_work[f'{target}_diff']

    X = df_work.drop([target, f'{target}_diff'] + cols_a_excluir, axis=1, errors='ignore')

    for i in range(inicio_test, len(df_original) - horizonte + 1, horizonte):
        X_train, X_test = X.iloc[:i], X.iloc[i : i + horizonte]
        y_train_diff = y_diff.iloc[:i]
        
        start_loc = df_work.index.get_loc(X_test.index[0])
        y_test_real = df_work[target].iloc[start_loc : start_loc + horizonte]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = obtener_modelo_configurado(modelo_nombre,params)
        
        X_train = X_train_scaled if modelo_nombre == "SVR" else X_train
        X_test = X_test_scaled if modelo_nombre == "SVR" else X_test
        
        model.fit(X_train, y_train_diff)
        
        if modelo_nombre == "SVR":
            complejidad = len(model.support_)
            guardar_complejidad_modelo(provincia=nombre_provincia, modelo="SVR",complejidad=complejidad,extra_info={"SupportVectors": complejidad})
            
        if modelo_nombre == "RandomForest":
            total_nodos = sum(tree.tree_.node_count for tree in model.estimators_)
            guardar_complejidad_modelo(provincia=nombre_provincia,modelo="RandomForest",complejidad=total_nodos,extra_info={"Arboles": len(model.estimators_),   "Nodos": total_nodos})
            
        if modelo_nombre == "XGBoost":
            booster = model.get_booster()
            trees_df = booster.trees_to_dataframe()
            total_nodos = len(trees_df)

            guardar_complejidad_modelo(provincia=nombre_provincia, modelo="XGBoost", complejidad=total_nodos, extra_info={"Nodos": total_nodos })
    
        preds_diff = model.predict(X_test)
        
        valor_base = df_work[target].iloc[start_loc - 1]
        preds_reales = valor_base + np.cumsum(preds_diff)

        for t in range(horizonte):
            todas_predicciones.append({'Fecha': y_test_real.index[t],'Real': y_test_real.iloc[t],'Pred': max(0, preds_reales[t]) })
            
    return pd.DataFrame(todas_predicciones)


def main_proceso_ml_provincia(nombre_provincia, carpeta_in, ruta_params, ruta_salida, modelos_a_evaluar=["SVR", "RandomForest", "XGBoost"], n_test_weeks=49, semanas=4, cols_a_excluir=["Provincias", "Fecha", "Semana Estadística", "Año"]):
    """ML multi-modelo por provincia    """
    
    params_df = pd.read_csv(ruta_params)
    
    df_original, df_diff = preparar_datos_ml(nombre_provincia, carpeta_in, "Casos", ["Provincias", "Año", "Semana Estadística"])

    for modelo_nombre in modelos_a_evaluar:
        try:

            mask = (params_df["Provincia"] == nombre_provincia) & (params_df["Modelo"] == modelo_nombre)
            if params_df[mask].empty:
                print(f"⚠️ No se encontraron parámetros para {nombre_provincia} - {modelo_nombre}. Saltando...")
                continue

            row = params_df[mask].iloc[0]
            params_dict = ast.literal_eval(row["Parametros"])

            df_detallado = ejecutar_ml_rolling(df_original, df_diff, n_test_weeks, semanas, nombre_provincia,modelo_nombre, params_dict, "Casos", cols_a_excluir)
            
            guardar_detalles_consolidados(df_detallado, modelo_nombre, nombre_provincia)
            
            df_metricas = procesar_metricas_finales(df_detallado, nombre_provincia, modelo_nombre,df_original)
            
            if os.path.exists(ruta_salida):
                existente = pd.read_csv(ruta_salida)
                df_final = pd.concat([existente, df_metricas], ignore_index=True).drop_duplicates(subset=['Provincia', 'Modelo'], keep='last')
            else:
                df_final = df_metricas


            guardar_resultados_csv(df_metricas,ruta_salida,subset_duplicates=["Provincia", "Modelo"] )
                        
        except Exception as e:
            print(f"❌ Error procesando {modelo_nombre} en {nombre_provincia}: {e}")
    # return df_final
            
            
def ejecutar_dl_rolling(df, n_test_weeks,nombre_provincia, horizonte, arch_type, params, target, cols_excluir):
    """Rolling DL con LSTM"""
    
    n_past = params['n_past']
    u1 = params.get('best_lstm_u1', 64)
    u2 = params.get('best_lstm_u2', 32)
    dropout = params.get('best_dropout', 0.2)
    dense_u = params.get('best_dense_u', 32)
    
    todas_predicciones = []
    inicio_test = len(df) - n_test_weeks

    df_num = df.drop(cols_excluir, axis=1, errors='ignore')
    target_idx = df_num.columns.get_loc(target)

    for i in range(inicio_test, len(df) - horizonte + 1, horizonte):

        train_df = df_num.iloc[:i]
        test_df  = df_num.iloc[i : i + horizonte]
        
        # aplicar log1p ANTES de escalar
        train_log = np.log1p(train_df.values)
        test_log  = np.log1p(test_df.values)

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_log)
        test_scaled  = scaler.transform(test_log)

        X_train = crear_secuencias_3d(train_scaled, n_past)
        y_train = train_scaled[n_past:, target_idx]

        model = build_lstm_model(
                    model_type=arch_type, 
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    lstm_u1=u1,
                    lstm_u2=u2,
                    dropout_rate=dropout,
                    dense_u=dense_u
                )
        total_params = model.count_params()
        n_lstm_layers = 0
        n_dense_layers = 0

        lstm_units = []
        dense_units = []

        for layer in model.layers:

            nombre = layer.__class__.__name__

            if "LSTM" in nombre:
                n_lstm_layers += 1

                try:
                    lstm_units.append(layer.units)
                except:
                    pass

            if "Dense" in nombre:

                n_dense_layers += 1
                dense_units.append(layer.units)
        attention_size = None

        for layer in model.layers:

            if "Attention" in layer.__class__.__name__:
                attention_size = 64
        
        guardar_complejidad_modelo(provincia=nombre_provincia,modelo=f"DL_{arch_type}",complejidad=total_params,
            extra_info={"Parametros_Entrenables": total_params,"Capas_LSTM": n_lstm_layers,"Capas_Densas": n_dense_layers,"Neuronas_LSTM": str(lstm_units),"Neuronas_Densas": str(dense_units),"Attention_Size": attention_size})
            
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True )
        lr_actual = params.get('best_lr', 0.001)  # Toma 'best_lr' si existe, si no, usa 0.001

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_actual),loss='mse')
        model.fit(X_train, y_train, epochs=params.get('epochs', 50), batch_size=params.get('batch_size', 32), verbose=0, callbacks=[early_stop])

        last_sequence = train_scaled[-n_past:].copy()
        preds_horizonte = []

        for h in range(horizonte):
            input_seq = last_sequence.reshape(1, n_past, train_scaled.shape[1])
            pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
            preds_horizonte.append(pred_scaled)

            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1, target_idx] = pred_scaled

        dummy = np.zeros((horizonte, train_scaled.shape[1]))
        dummy[:, target_idx] = preds_horizonte
        
        preds_log    = scaler.inverse_transform(dummy)[:, target_idx]
        preds_reales = np.expm1(preds_log)          # ← expm1 es la inversa exacta de log1p

        y_true = test_df[target].values
        for t in range(horizonte):
            todas_predicciones.append({'Fecha': test_df.index[t],'Real':  y_true[t],'Pred':  max(0, preds_reales[t])})

    return pd.DataFrame(todas_predicciones)

def main_proceso_dl_provincia(nombre_provincia, carpeta, ruta_parametros, ruta_salida_excel,target='Casos', modelo_dl = ["stacked", "bidirectional" , "attention"], n_test_weeks=49, horizonte=4):
    """
    modelo_dl: "stacked", "bidirectional" o "attention"
    """        

    df = pd.read_csv(os.path.join(carpeta, f'{nombre_provincia}.csv'))
    df = df.sort_values('Fecha').set_index('Fecha')
    df.index = pd.to_datetime(df.index)
    
    resumen_params = pd.read_csv(ruta_parametros)
    for model in modelo_dl:

        mask = (resumen_params["Provincia"] == nombre_provincia) & (resumen_params["Modelo"] == model)

        if resumen_params[mask].empty:
            print(f"No se hallaron parámetros para {model} en {nombre_provincia}")
            return
            
        params_str = resumen_params[mask]["Parametros"].iloc[0]

        params = json.loads(params_str) 
        arch_type = params['architecture'] 

        cols_excluir = ["Provincias", "Año", "Semana Estadística"]
        df_detallado = ejecutar_dl_rolling(df, n_test_weeks,nombre_provincia, horizonte, arch_type, params, target, cols_excluir)
        print(11111111111111111111111,model)
        print(df_detallado)
        guardar_detalles_consolidados(df_detallado, model, nombre_provincia)
        
        df_metricas = procesar_metricas_finales(df_detallado, nombre_provincia,df_original=df,lSTM=model, usar_exog=False)
        df_metricas['Modelo'] = model

        if os.path.exists(ruta_salida_excel):
            existente = pd.read_csv(ruta_salida_excel)
            df_final = pd.concat([existente, df_metricas], ignore_index=True)
        else:
            df_final = df_metricas

        guardar_resultados_csv( df_metricas,ruta_salida_excel,subset_duplicates=["Provincia", "Modelo"])
        
        print(f"✅ Proceso DL ({model}) completado para {nombre_provincia}")
        
