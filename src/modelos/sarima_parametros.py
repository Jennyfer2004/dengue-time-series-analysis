import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

def entrenar_instancia_sarimax(train_y, train_exog, order, seasonal_order):
    """Ajustar y retornar el objeto matemático de SARIMA/X"""
    
    model = SARIMAX(train_y, exog=train_exog, order=order, seasonal_order=seasonal_order,enforce_stationarity=False,enforce_invertibility=False)
   
    return model.fit(disp=False)