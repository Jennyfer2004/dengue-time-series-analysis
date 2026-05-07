import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

import optuna
import optuna.visualization as vis
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


def preparar_datos(nombre_provincia, carpeta_entrada, columna_objetivo, cols_a_excluir ,n_forecast):
    
    ruta = os.path.join(carpeta_entrada, f'{nombre_provincia}.csv')
    
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró: {ruta}")
    
    df = pd.read_csv(ruta)
    df_diff = df.copy()

    for col in df_diff.columns:
        
        if col not in cols_a_excluir:
            df_diff[col] = df_diff[col].diff()

    df_diff = df_diff.dropna()
    
    X = df_diff.drop([columna_objetivo] + cols_a_excluir, axis=1, errors='ignore')
    y = df_diff[columna_objetivo]

    X_train, X_test = X[:-n_forecast], X[-n_forecast:]
    y_train, y_test = y[:-n_forecast], y[-n_forecast:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        "df_original": df,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled
    }
    
def crear_modelo(nombre, params):
    """Crea una instancia del modelo y limpia los prefijos de Optuna."""
    
    # Limpiar prefijos de los nombres de parámetros (rf_, xgb_, svr_)
    p = {k.split('_', 1)[-1]: v for k, v in params.items() if k != 'classifier'}

    if 'min_split' in p: p['min_samples_split'] = p.pop('min_split')
    if 'min_leaf' in p: p['min_samples_leaf'] = p.pop('min_leaf')
    if 'max_feat' in p: p['max_features'] = p.pop('max_feat')
    if 'lr' in p: p['learning_rate'] = p.pop('lr')
    if 'sub' in p: p['subsample'] = p.pop('sub')
    if 'colsample' in p: p['colsample_bytree'] = p.pop('colsample')

    if nombre == "RandomForest":
        return RandomForestRegressor(**p, random_state=42, n_jobs=-1)
    
    elif nombre == "XGBoost":
        return XGBRegressor(**p, random_state=42, n_jobs=-1)
    
    elif nombre == "SVR":
        return SVR(**p)
    return None

def objetivo_optuna(trial, data):
    
    model_name = trial.suggest_categorical("classifier", ["RandomForest", "XGBoost", "SVR"])
    
    params = {}
    
    if model_name == "RandomForest":
        params = {
            'rf_n_estimators': trial.suggest_int("rf_n_estimators", 100, 1000),
            'rf_max_depth': trial.suggest_int("rf_max_depth", 5, 50),
            'rf_min_split': trial.suggest_int("rf_min_split", 2, 20),
            'rf_max_feat': trial.suggest_float("rf_max_feat", 0.1, 1.0)
        }
        X_data = data["X_train"].values
        
    elif model_name == "XGBoost":
        params = {
            'xgb_n_estimators': trial.suggest_int("xgb_n_estimators", 100, 1000),
            'xgb_lr': trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True),
            'xgb_max_depth': trial.suggest_int("xgb_max_depth", 3, 15)
        }
        X_data = data["X_train"].values
        
    else: # SVR
        params = {
            'svr_C': trial.suggest_float("svr_C", 1e-3, 1e3, log=True),
            'svr_epsilon': trial.suggest_float("svr_epsilon", 1e-3, 1.0, log=True),
            'svr_kernel': trial.suggest_categorical("svr_kernel", ["rbf", "poly"])
        }
        X_data = data["X_train_scaled"]

    model = crear_modelo(model_name, params)
    
    # Validación Cruzada Temporal
    tscv = TimeSeriesSplit(n_splits=3)
    errors = []
    for t_idx, v_idx in tscv.split(X_data):
        model.fit(X_data[t_idx], data["y_train"].iloc[t_idx])
        p = model.predict(X_data[v_idx])
        errors.append(mean_squared_error(data["y_train"].iloc[v_idx], p))
    
    return np.mean(errors)

def reconstruir_y_evaluar(data, preds_diff, model_name, col):
    
    def reconstruir(valor_base, diffs):
        res = []
        actual = valor_base
        for d in diffs:
            actual = max(0, actual + d)
            res.append(actual)
        return np.array(res)

    idx_base = data["df_original"].index.get_loc(data["y_test"].index[0]) - 1
    valor_base = data["df_original"][col].iloc[idx_base]
    
    preds_reales = reconstruir(valor_base, preds_diff)
    y_test_original = data["df_original"][col].loc[data["y_test"].index].values

    metrics = {
        "MAE": mean_absolute_error(y_test_original, preds_reales),
        "RMSE": np.sqrt(mean_squared_error(y_test_original, preds_reales)),
        "MAPE": mean_absolute_percentage_error(y_test_original, preds_reales) * 100
    }
    return preds_reales, y_test_original, metrics

def guardar_reporte(y_true, y_pred, nombre_provincia, modelo_nombre, parametros, ruta_archivo):
    """
    Guarda las métricas de modelos de ML (RF, XGB, SVR).
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    datos_fila = {
        "Provincia": nombre_provincia,
        "Modelo": modelo_nombre,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": f"{round(mape * 100, 2)}%", 
        "Parametros": parametros            
    }

    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    file_exists = os.path.isfile(ruta_archivo)

    with open(ruta_archivo, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=datos_fila.keys())
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(datos_fila)

    print(f"\nMétricas ML guardadas para {nombre_provincia} ({modelo_nombre}):")
    print(f"- MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape*100:.2f}%")
    
    return datos_fila


def graficar_resultados(y_real, y_pred, index, prov, modelo, folder):
    
    plt.figure(figsize=(10, 5))
    plt.plot(index, y_real, label="Real", color='black', marker='o')
    plt.plot(index, y_pred, label=f"Pred: {modelo}", color='red', linestyle='--')
    plt.title(f"ML Casos - {prov}")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"ML_{prov}.png"))
    plt.close()



def ejecutar_ml_optimizacion(prov, ruta_in, col="Casos", trials=100,cols_a_excluir = ["Provincias", "Fecha", "Semana Estadística", "Año"],ruta_guardar="resultados/metricas_globales.csv",n_forecast=4):

    data = preparar_datos(prov, ruta_in, col,cols_a_excluir,n_forecast)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objetivo_optuna(t, data), n_trials=trials)
    
    best_params = study.best_params
    best_model_name = best_params['classifier']
    model = crear_modelo(best_model_name, best_params)
    
    X_train_final = data["X_train_scaled"] if best_model_name == "SVR" else data["X_train"]
    X_test_final = data["X_test_scaled"] if best_model_name == "SVR" else data["X_test"]
    
    model.fit(X_train_final, data["y_train"])
    preds_diff = model.predict(X_test_final)
    
    preds_reales, y_real, metrics = reconstruir_y_evaluar(data, preds_diff, best_model_name, col)
    params_str = str(best_params)

    guardar_reporte(y_real,preds_reales,prov,best_model_name,params_str,ruta_guardar )
    graficar_resultados(y_real, preds_reales, data["y_test"].index, prov, best_model_name, "resultados/graficos_ml")
    
    print(f"✅ {prov} terminado. Mejor modelo: {best_model_name} (MAE: {metrics['MAE']:.2f})")
    