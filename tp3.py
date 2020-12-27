#!/usr/bin/python3

""" 
--------------------------------------------------------------------------------------
-----------------------------------| FORMA DE USO |-----------------------------------
--------------------------------------------------------------------------------------
Las instrucciones de uso son mostradas al ejecutar el siguiente comando:
   python3 tp3.py --help

Ejemplo de uso: 
   python tp3.py -t data/train.csv -o output.csv -f precio -c 2 metroscubiertos banos
   El comando de ejemplo tiene las siguientes caracteristicas:
       - Lee el dataset:                                   data/train.csv
       - Guarda el resultado en el archivo:                output.csv
       - Se estima del dataset la variable:                "precio"
       - La cant de variables que estiman son:             2
       - La estimación se hace en base a las variables:    "metroscubiertos" y "banos"
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
"""

import sys
sys.path.append('notebooks')

import pandas as pd
import numpy as np
import metnum 

# Importo las funciones necesarias desde otro archivo nuestro
from extra_scripts import cross_validate

# Lectura de parametros pasados por CLI
argc = len(sys.argv)

if (argc <= 8) & (argc != 2) & (argc != 1):
    print("Parametros Invalidos\n")
    sys.exit()
elif (argc == 2) & (sys.argv[1] == "--help"):
    print("Uso: tp3 -t <train> -o <output> -f <predict> -c <cant_features> [feature1 feature2 ...]" )
    print("     <train>: Direccion para el archivo CSV de train")
    print("     <output>: Direccion para el archivo CSV donde guardar el resultado")
    print("     <predict>: Feature a predecir")
    print("     <cant_features>: cantidad de features utilizadas para el training")
    print("     [feature1 feature2 ...]: las features en cuestion (la cantidad debe corresponder a cant_features)")
    print("        OBS: Sin los corchetes")
    sys.exit()
elif (argc >= 9):
    train_file = sys.argv[2]
    output_file = sys.argv[4]
    feature_predict = sys.argv[6]
    cant_features = int(sys.argv[8])
    
    if (cant_features + 9) != argc:
        print("Parametros Invalidos")
        
    features = []
    for i in range(cant_features):
        features.append(sys.argv[i + 9])
        
    print(f"Train file: {train_file} ")
    print(f"Output file: {output_file}")
    print(f"Caracteristica a predecir: {feature_predict} ")
    print(f"Cantidad de features utilizadas: {cant_features} ")
    print(f"Features utilizadas: {features}")
    
    # Con todos los datos pasamos a ejecutar el training
    
    # Carguemos el dataset de training
    print("---------------------------------------------------")
    print("Cargando el Dataset, espere por favor...")
    df_train = pd.read_csv(train_file)
    print("Lectura de Dataset terminada!")
    # Veamos el tamaño
    print("Dimensiones del dataset: ", df_train.shape)
    
    #Limpiemos los NaNs de las features
    df_train = df_train[df_train[feature_predict].notna()]
    for feature in features:
        df_train = df_train[df_train[feature].notna()]
    
    # Preparo el fitteo
    cant_train = int((df_train.shape[0] / 100) * 80)
    shuffle = df_train.sample(random_state=np.random.seed(1), frac=1)
    new_train = shuffle.iloc[:cant_train, :]
    new_test = shuffle.iloc[cant_train:df_train.shape[0], :]
    x_train, x_test = new_train[features].values, new_test[features].values
    y_train, y_test = new_train[feature_predict].values, new_test[feature_predict].values

    print(f"train: {x_train.shape}, test: {x_test.shape}")
    print("OBS: El 80 por ciento va para training y el 20 por ciento restante para testing")

    # Ahora queremos entrenar el modelo
    modelo = metnum.LinearRegression()
    modelo.fit(x_train, y_train)
    y_pred = modelo.predict(x_test)
    
    print("--------------------------------")
    print("Training completo!")
    print("--------------------------------")
    
    # Guardo la prediccion
    pd.DataFrame(y_pred).to_csv(output_file)
    
    # Metricas del conjunto de training
    print("Metricas del entrenamiento: ")
    results = cross_validate(10, x_train, y_train, True, True)
    print("----------------------------------------------------")
    print("---------EL PROGRAMA FINALIZO CORRECTAMENTE---------")
    print("----------------------------------------------------")
    
    sys.exit()
