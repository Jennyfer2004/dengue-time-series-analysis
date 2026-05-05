import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import optuna
import optuna.visualization as vis
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

def load(nombre_provincia,ruta_archivo,lags_por_provincia,columnas_clima):

    if os.path.exists(ruta_archivo):
        df = pd.read_csv(ruta_archivo)

        cols_a_eliminar = ['Unnamed: 0', "Provincias", "Año_x", "Año_y", "Fecha"]
        df = df.drop(columns=[c for c in cols_a_eliminar if c in df.columns])

        lags_significativos = lags_por_provincia.get(nombre_provincia, [])

        for lag in lags_significativos:

            df[f'Dengue_Lag{lag}'] = df['Casos_Dengue'].shift(lag)

        df_modelo = df.dropna().reset_index(drop=True)

        print(f"✅ Dataset para {nombre_provincia} listo.")
        print(f"Columnas finales: {df_modelo.columns.tolist()}")
        # return df_modelo

    else:
        print(f"❌ No se encontró el archivo para {nombre_provincia}")
        
    df_diff = df_modelo.copy()
    df_diff['Casos_Dengue'] = df_modelo['Casos_Dengue'].diff()

    for col in columnas_clima:
        if col in df_diff.columns:
            df_diff[col] = df_diff[col].diff()

    # Eliminamos nulos generados por .diff() ANTES de separar X e y
    df_diff = df_diff.dropna()

    X = df_diff.drop(['Casos_Dengue'], axis=1, errors='ignore')
    y = df_diff['Casos_Dengue']

    split = int(len(X) * 0.991)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ESCALADO (para SVR) 
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    return df_modelo,X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def objective(trial, X_train, X_train_scaled, y_train):
    classifier_name = trial.suggest_categorical("classifier", ["RandomForest", "XGBoost", "SVR"])

    if classifier_name == "RandomForest":
        params = {
            'n_estimators': trial.suggest_int("rf_n_estimators", 100, 1000),
            'max_depth': trial.suggest_int("rf_max_depth", 5, 50),
            'min_samples_split': trial.suggest_int("rf_min_split", 2, 20),
            'min_samples_leaf': trial.suggest_int("rf_min_leaf", 1, 10),
            'max_features': trial.suggest_float("rf_max_feat", 0.1, 1.0),
            'bootstrap': trial.suggest_categorical("rf_bootstrap", [True, False])
        }
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        X_data = X_train.values # Usamos values para evitar problemas de índices en CV

    elif classifier_name == "XGBoost":
        params = {
            'n_estimators': trial.suggest_int("xgb_n_estimators", 100, 2000),
            'learning_rate': trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int("xgb_max_depth", 3, 15),
            'subsample': trial.suggest_float("xgb_sub", 0.5, 1.0),
            'colsample_bytree': trial.suggest_float("xgb_colsample", 0.3, 1.0),
            'gamma': trial.suggest_float("xgb_gamma", 1e-8, 1.0, log=True)
        }
        model = XGBRegressor(**params, random_state=42, n_jobs=-1)
        X_data = X_train.values

    else: # SVR
        params = {
            'C': trial.suggest_float("svr_c", 1e-3, 1e3, log=True),
            'epsilon': trial.suggest_float("svr_epsilon", 1e-3, 1.0, log=True),
            'gamma': trial.suggest_float("svr_gamma", 1e-4, 1.0, log=True),
            'kernel': trial.suggest_categorical("svr_kernel", ["rbf", "poly"])
        }
        model = SVR(**params)
        X_data = X_train_scaled

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, val_idx in tscv.split(X_data):
        X_t, X_v = X_data[train_idx], X_data[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        scores.append(np.sqrt(mean_squared_error(y_v, preds)))

    return np.mean(scores)

def main(nombre_provincia,ruta_archivo,lags_por_provincia,columnas_clima):
    
    df_modelo,X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled=load(nombre_provincia,ruta_archivo,lags_por_provincia,columnas_clima)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, X_train_scaled, y_train), n_trials=100)

    best_params = study.best_params.copy()
    model_type = best_params.pop('classifier')

    final_params = {}
    for k, v in best_params.items():

        new_key = k.replace('rf_', '').replace('xgb_', '').replace('svr_', '')

        if new_key == 'min_split': new_key = 'min_samples_split'
        if new_key == 'min_leaf':  new_key = 'min_samples_leaf'
        if new_key == 'max_feat':  new_key = 'max_features'
        if new_key == 'c':         new_key = 'C'

        final_params[new_key] = v

    if model_type == "RandomForest":
        best_model = RandomForestRegressor(**final_params, random_state=42, n_jobs=-1)
        best_model.fit(X_train, y_train)
        preds_diff = best_model.predict(X_test)

    elif model_type == "XGBoost":
        best_model = XGBRegressor(**final_params, random_state=42, n_jobs=-1)
        best_model.fit(X_train, y_train)
        preds_diff = best_model.predict(X_test)

    else: # SVR
        if 'eps' in final_params:
            final_params['epsilon'] = final_params.pop('eps')
        best_model = SVR(**final_params)
        best_model.fit(X_train_scaled, y_train)
        preds_diff = best_model.predict(X_test_scaled)


    def reconstruir_serie(primer_valor, diferencias):
        reconstruida = []
        actual = primer_valor
        for diff in diferencias:
            actual = actual + diff
            if actual < 0: actual = 0
            reconstruida.append(actual)
        return np.array(reconstruida)


    idx_base = df_modelo.index.get_loc(y_test.index[0]) - 1
    valor_base = df_modelo['Casos_Dengue'].iloc[idx_base]

    preds_reales = reconstruir_serie(valor_base, preds_diff)
    y_test_original = df_modelo['Casos_Dengue'].loc[y_test.index].values

    print(f"\n🏆 Mejor Modelo: {model_type}")
    print(f"MAE Real: {mean_absolute_error(y_test_original, preds_reales):.2f} casos")
    print(f"RMSE Real: {np.sqrt(mean_squared_error(y_test_original, preds_reales)):.2f}")
    print(f"MAPE Real: {mean_absolute_percentage_error(y_test_original, preds_reales)*100:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test_original, label="Real", color='blue', marker='o')
    plt.plot(y_test.index, preds_reales, label=f"Predicción ({model_type})", color='red', linestyle='--')
    plt.title(f"Predicción de Dengue - Mejor Modelo: {model_type}")
    plt.xlabel("Índice Temporal")
    plt.ylabel("Casos de Dengue")
    plt.legend()
    plt.grid(True)
    plt.show()

    vis.plot_optimization_history(study).show()
