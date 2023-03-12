#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:46:24 2022

@author: felixganga
"""

from flask import Flask, jsonify
import pandas as pd
import pickle
from flask import request

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

PATH = '/Users/felixganga/Documents/GitHub/Projet7_Felix_GANGA/'
#Chargement des données 

#df = pd.read_parquet(PATH+'test_df.parquet')
#df = pd.read_csv(PATH+'test_df.csv')
df = pd.read_csv('/Users/felixganga/Documents/OpenClass_Data/P7_GANGA_felix/X_rfecv.csv')
print('df shape = ', df.shape)

#Chargement du modèle
model = pickle.load(open('/Users/felixganga/Documents/GitHub/Projet7_Felix_GANGA/Projet7_dossier/Backend/lgbm.pkl', 'rb'))


@app.route('/',methods=['GET'])
def hello():
    return 'Hello, World!'

@app.route('/credit/',methods=['GET'])
def credit():
    
    id_client = request.args.get('id_client')
    print('id client = ', id_client)
    
    #Récupération des données du client en question
    ID = int(id_client)
    X = df[df['SK_ID_CURR'] == ID]
    
    ignore_features = ['Unnamed: 0','SK_ID_CURR', 'INDEX', 'TARGET']
    relevant_features = [col for col in df.columns if col not in ignore_features]

    X = X[relevant_features]
    
    print('X shape = ', X.shape)
    
    proba = model.predict_proba(X)
    prediction = model.predict(X)

    #DEBUG
    #print('id_client : ', id_client)
  
    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(proba[0][0])
        }

    print('Nouvelle Prédiction : \n', dict_final)

    return jsonify(dict_final)




#lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)
    