from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

def crear_modelo(nombre, params):
    """Crea una instancia del modelo y limpia los prefijos de Optuna."""
    
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


def obtener_params_por_modelo(trial, model_name):
    """Definicion de hyperparametros modelos """
    
    if model_name == "RandomForest":
        return {
            'rf_n_estimators': trial.suggest_int("rf_n_estimators", 100, 1000),
            'rf_max_depth': trial.suggest_int("rf_max_depth", 5, 50),
            'rf_min_split': trial.suggest_int("rf_min_split", 2, 20),
            'rf_max_feat': trial.suggest_float("rf_max_feat", 0.1, 1.0)
        }
    elif model_name == "XGBoost":
        return {
            'xgb_n_estimators': trial.suggest_int("xgb_n_estimators", 100, 1000),
            'xgb_lr': trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True),
            'xgb_max_depth': trial.suggest_int("xgb_max_depth", 3, 15)
        }
    else: # SVR
        return {
            'svr_C': trial.suggest_float("svr_C", 1e-3, 1e3, log=True),
            # 'svr_epsilon': trial.suggest_float("svr_epsilon", 1e-3, 1.0, log=True),
            'svr_epsilon': trial.suggest_float("svr_epsilon", 1e-6, 0.1, log=True),
            'svr_kernel': trial.suggest_categorical("svr_kernel", ["rbf", "poly"])
        }

def obtener_modelo_configurado(nombre_modelo, params_dict):

    # Limpiar prefijos de Optuna
    p = {k.split('_', 1)[-1]: v for k, v in params_dict.items() if k != 'classifier'}
    mapping = {'min_split': 'min_samples_split', 'min_leaf': 'min_samples_leaf', 
               'max_feat': 'max_features', 'lr': 'learning_rate'}
    p = {mapping.get(k, k): v for k, v in p.items()}

    if nombre_modelo in ["RandomForest", "RF"]: return RandomForestRegressor(**p, random_state=42)
    if nombre_modelo in ["XGBoost", "XGB"]: return XGBRegressor(**p, random_state=42)
    if nombre_modelo == "SVR": return SVR(**p)
    return None