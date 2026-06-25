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
    
        def objective(trial):
        lstm_u1 = trial.suggest_categorical('lstm_u1', [32, 64, 128])
        lstm_u2 = trial.suggest_categorical('lstm_u2', [16, 32, 64])
        dense_u = trial.suggest_categorical('dense_u', [16, 32, 64])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4, step=0.1)
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        # Construir el modelo
        model = build_lstm_model(
            arch, 
            (X_train.shape[1], X_train.shape[2]), 
            lstm_u1=lstm_u1, 
            lstm_u2=lstm_u2, 
            dropout_rate=dropout_rate,
            dense_u=dense_u
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse')
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2, 
            callbacks=[early_stop], 
            verbose=0
        )
        
        val_loss = min(history.history['val_loss'])
        return val_loss


    for arch in architectures:
        print(f"\n🚀 Optimizando hiperparámetros para {nombre_provincia} con arquitectura: {arch}")
        
        study = optuna.create_study(direction='minimize')
        
        study.optimize(objective, n_trials, timeout) 
        
        print(f"Mejores parámetros para {arch}: {study.best_params}")
        
        best_p = study.best_params
        best_model = build_lstm_model(
            arch, (X_train.shape[1], X_train.shape[2]),
            lstm_u1=best_p['lstm_u1'], lstm_u2=best_p['lstm_u2'],
            dropout_rate=best_p['dropout_rate'], dense_u=best_p['dense_u']
        )
        
        best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_p['learning_rate']), loss='mse')
        
        best_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
        
        last_sequence = X_test[-1].copy()
        current_preds_scaled = []
        for step in range(n_forecast):
            input_seq = last_sequence.reshape(1, n_past, n_feats)
            next_pred_scaled = best_model.predict(input_seq, verbose=0)
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

        config_actual = {
            "n_past": n_past, "n_forecast": n_forecast, "epochs": epochs, "batch_size": batch_size, 
            "architecture": arch, "n_features": n_feats,
            "best_lstm_u1": best_p['lstm_u1'], "best_lstm_u2": best_p['lstm_u2'], 
            "best_lr": best_p['learning_rate'], "best_dropout": best_p['dropout_rate']
        }
        
        guardar_reporte(y_reales, p_reales, nombre_provincia, arch, config_actual, ruta=ruta_guardar)



if __name__ == "__main__":
    print("🔮 Iniciando optimización y guardado de parámetros DL...")
    for prov in PROVINCIAS:
        try:
            ejecutar_dl_workflow(nombre_provincia=prov, carpeta_entrada="data/processed")
        except Exception as e:
            print(f"Saltando {prov} por error: {e}")
