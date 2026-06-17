
import numpy as np
import optuna
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from src.modelos.ML_models import crear_modelo, obtener_params_por_modelo
from src.preprocessing.prep_ML import preparar_datos, reconstruir_y_evaluar
from src.utils import guardar_reporte  

def ejecutar_ml_optimizacion(prov, ruta_in, col="Casos", trials=100, cols_a_excluir=["Provincias", "Fecha", "Semana Estadística", "Año"], ruta_guardar="outputs/tables/metricas_globales.csv", n_forecast=4,    modelos_a_probar = ["RandomForest", "XGBoost", "SVR"]):
    
    data = preparar_datos(prov, ruta_in, col, cols_a_excluir, n_forecast)
    
    for modelo_name in modelos_a_probar:
        print(modelo_name)

        study = optuna.create_study(direction="minimize")
        
        def objetivo_modelo(trial):

            params = obtener_params_por_modelo(trial, modelo_name)
            X_data = data["X_train_scaled"] if modelo_name == "SVR" else data["X_train"].values
            
            model = crear_modelo(modelo_name, params)
            tscv = TimeSeriesSplit(n_splits=3)
            errors = []
            for t_idx, v_idx in tscv.split(X_data):
                model.fit(X_data[t_idx], data["y_train"].iloc[t_idx])
                p = model.predict(X_data[v_idx])
                errors.append(mean_squared_error(data["y_train"].iloc[v_idx], p))
            return np.mean(errors)

        study.optimize(objetivo_modelo, n_trials=trials)
        
        best_params = study.best_params
        model = crear_modelo(modelo_name, best_params)
        X_train_final = data["X_train_scaled"] if modelo_name == "SVR" else data["X_train"]
        X_test_final = data["X_test_scaled"] if modelo_name == "SVR" else data["X_test"]
        
        model.fit(X_train_final, data["y_train"])
        preds_diff = model.predict(X_test_final)
        
        preds_reales, y_real = reconstruir_y_evaluar(data, preds_diff, modelo_name, col)
        guardar_reporte(y_real, preds_reales, prov, modelo_name, str(best_params), ruta_guardar)
        
        print(f"✅ {prov} - {modelo_name} completado.")
        