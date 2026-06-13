# datos del proyecto - Encuesta Origen Destino (EOD) Santiago 2012
# fuente: SECTRA / U. Alberto Hurtado. datos reales a nivel de viaje
# cada fila es un viaje real. queremos predecir cuanto dura (en minutos)

import pandas as pd
import numpy as np

df_viajes = pd.read_csv("eod_viajes.csv")

# one-hot a las categoricas
df_modelo = pd.get_dummies(df_viajes[["modo", "proposito", "periodo"]], dtype=int)
df_modelo["es_hombre"] = df_viajes["es_hombre"]
df_modelo["edad"]      = df_viajes["edad"]
df_modelo["etapas"]    = df_viajes["etapas"]
df_modelo["dist_km"]   = df_viajes["dist_km"]
df_modelo["tiempo"]    = df_viajes["tiempo"]

FEATURES  = [c for c in df_modelo.columns if c != "tiempo"]
TARGET    = "tiempo"
NUMERICAS = ["tiempo", "dist_km", "edad", "etapas", "es_hombre"]
