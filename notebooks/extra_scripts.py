# %% [markdown]
# ## K-Fold Cross Validation
# A la hora de realizar los experimentos al igual que en el tp anterior se debe usar cross-validation para evitar overfitting y obtener resultados estadisticamente mas robustos.
# 
# ## Metricas
# Se utilizan las funciones provistas por sklearn
# ###  RMSE 
# - La raíz del error cuadrático medio 
# ###  RMSLE 
# - Error logarítmico cuadrático medio
# 
# ### R^2
# - "is the proportion of the variance in the dependent variable that is predictable from the independent variable(s)"
# - (Solo la usamos cuando sea necesario)

# %%
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

import metnum
import numpy as np

def rmse(A,B):
    return np.sqrt(mean_squared_error(A, B))

def rmsle(A,B):
    return np.sqrt(mean_squared_log_error(A, B))

def cross_validate(K, X, Y, visualize = False, R2 = False):
    """
    Funcion custom para el TP 
    """
    def display_metrics(dict_k):
        """
        Funcion que permite visualizar la data
        """
        rmse, rmsle, r2= dict_k["RMSE"], dict_k["RMSLE"], dict_k["R2"]

        print(f"RMSE = {(np.mean(rmse), np.std(rmse))}")
        print(f"RMSLE = {(np.mean(rmsle), np.std(rmsle))}")
        if R2 == True:
            print(f"R2 = {(np.mean(r2), np.std(r2))}")

    # Usamos la funcion de skleanr para generar los splits 
    kfold = KFold(n_splits=K)
    kfold.get_n_splits(X)
    splits = kfold.split(X)

    #Almacenamos las metricas en arrays
    rmse_values = []
    rmsle_values = []
    R2_values = []
    
    for index_train, index_test in splits:
        # Tomamos el split correspondiente
        X_train, X_test = X[index_train], X[index_test]
        Y_train, Y_test = Y[index_train], Y[index_test]

        # Se calcula la prediccion
        regression = metnum.LinearRegression()
        
        regression.fit(X_train, Y_train)
        y_pred = regression.predict(X_test)
        # regression.fit(X_train.reshape(-1, 1), Y_train)
        # y_pred = regression.predict(X_test.reshape(-1, 1))
        # y_pred = y_pred.reshape(y_pred.shape[0])           
        
        # Se evaluan las metricas
        R2_values.append(r2_score(Y_test, y_pred))
        if np.any(y_pred < 0):
            print("Error! Se obtuvo un negativo, se procede a tomar el valor absoluto")
            y_pred = np.abs(y_pred)
        rmse_values.append(rmse(Y_test, y_pred))
        rmsle_values.append(rmsle(Y_test, y_pred))
        
    #Lo guardamos en un dict de numpy arrays
    metrics_dict = {"RMSE": np.asarray(rmse_values), "RMSLE": np.asarray(rmsle_values), "R2": np.asarray(R2_values)}
    
    if(visualize):
        display_metrics(metrics_dict)

    return metrics_dict

def remove_outliers(df, col_name, sigmas = 3):
    from scipy.stats import zscore
    z_score = zscore(df[col_name])
    filtered_entries = []
    for zi in z_score:
        filtered_entries.append(np.abs(zi) < sigmas)
    print(df[filtered_entries].shape)
    return df[filtered_entries]
