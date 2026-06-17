# tuning_dl.py (En la raíz del proyecto)
import os
import csv
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Importamos lo que modularizamos en src/ para no duplicar código
from src.modelos.lstm_models import build_lstm_model
from src.preprocessing.prep_LSTM import preparar_datos_dl, desescalar_predicciones
from src.utils import guardar_reporte

PROVINCIAS = ["Pinar_del_Río", "Artemisa", "La Habana", "Mayabeque", "Matanzas", "Cienfuegos", 
              "Villa Clara", "Sancti_Spíritus", "Ciego_de_Ávila", "Camagüey", "Las Tunas", 
              "Granma", "Holguín", "Santiago_de_Cuba", "Guantánamo", "Isla_de_La_Juventud"]

    
def ejecutar_dl_workflow(nombre_provincia, carpeta_entrada, n_forecast=4, ruta_n_past="outputs/tables/par_lstm.csv", cols_excluir=["Provincias", "Fecha", "Año", "Semana Estadística"],epochs=100, batch_size=32, ruta_guardar="outputs/tables/metricas_globales.csv"):
    
    architectures = ['stacked', 'bidirectional', 'attention']
    
    X_train, X_test, y_train, y_test, scaler, n_feats,n_past = preparar_datos_dl(
        nombre_provincia, carpeta_entrada, n_forecast, ruta_n_past, cols_excluir
    )
    
    df = pd.read_csv(os.path.join(carpeta_entrada, f'{nombre_provincia}.csv'))
    df_num = df.drop(cols_excluir, axis=1, errors='ignore')

    data_scaled = scaler.transform(df_num.values)

    best_mae = float('inf')
    best_arch = None
    final_preds_plot = None

    for arch in architectures:
        print(f"Entrenando {arch}...")
        config_actual = {"n_past": n_past,"n_forecast": n_forecast,"epochs": epochs,"batch_size": batch_size,"architecture": arch,"n_features": n_feats}
        
        model = build_lstm_model(arch, (X_train.shape[1], X_train.shape[2]))
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, 
                  callbacks=[early_stop], verbose=0)

        last_sequence = X_test[-1].copy()
        current_preds_scaled = []
        
        for step in range(n_forecast):

            input_seq = last_sequence.reshape(1, n_past, n_feats)
            
            next_pred_scaled = model.predict(input_seq, verbose=0)
            current_preds_scaled.append(next_pred_scaled[0, 0])
            
            last_sequence = np.roll(last_sequence, -1, axis=0)
            
            last_sequence[-1, 0] = next_pred_scaled[0, 0]
            
        current_preds_scaled = np.array(current_preds_scaled).reshape(1, -1)
        
        dummy_preds = np.zeros((n_forecast, n_feats))
        dummy_preds[:, 0] = current_preds_scaled.flatten()
        p_reales = scaler.inverse_transform(dummy_preds)[:, 0]
        
        dummy_reales = np.zeros((n_forecast, n_feats))
        dummy_reales[:, 0] = y_test[-1] 
        y_reales = scaler.inverse_transform(dummy_reales)[:, 0]

        guardar_reporte(y_reales, p_reales, nombre_provincia, arch, config_actual, ruta=ruta_guardar)
        

    if final_preds_plot is not None:
        print(f"✅ Finalizado {nombre_provincia}")



if __name__ == "__main__":
    print("🔮 Iniciando optimización y guardado de parámetros DL...")
    for prov in PROVINCIAS:
        try:
            ejecutar_dl_workflow(nombre_provincia=prov, carpeta_entrada="data/processed")
        except Exception as e:
            print(f"Saltando {prov} por error: {e}")