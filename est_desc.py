import numpy as np
import pandas as pd
from datos import df_viajes, df_modelo, NUMERICAS

def titulo(texto):
    print(f"\n{'='*50}\n  {texto}\n{'='*50}")

def estadisticas_completas(serie, nombre):
    datos = serie.dropna().values

    media    = np.mean(datos)
    mediana  = np.median(datos)
    varianza = np.var(datos, ddof=1)
    desv_std = np.std(datos, ddof=1)

    valores, conteos = np.unique(datos.round(2), return_counts=True)
    moda = valores[np.argmax(conteos)]

    print(f"\n  Variable: {nombre}")
    print(f"    Media          : {media:.4f}")
    print(f"    Mediana        : {mediana:.4f}")
    print(f"    Moda           : {moda:.4f}")
    print(f"    Varianza       : {varianza:.4f}")
    print(f"    Desv. estandar : {desv_std:.4f}")


def covarianza_correlacion(df, col1, col2):
    x = df[col1].values
    y = df[col2].values
    n = len(x)

    cov_mues = np.sum((x-x.mean()) * (y-y.mean()))/(n-1)
    corr     = cov_mues/(np.std(x, ddof=1) * np.std(y, ddof=1))

    print(f"\n  [{col1}  vs  {col2}]")
    print(f"    Covarianza muestral    : {cov_mues:.4f}")
    print(f"    Correlacion de Pearson : {corr:.4f}")

def ejecutar():
    titulo("1.1  TIEMPO DE VIAJE")
    estadisticas_completas(df_viajes["tiempo"],  "Tiempo de viaje (min)")

    titulo("1.2  DISTANCIA Y EDAD")
    estadisticas_completas(df_viajes["dist_km"], "Distancia de viaje (km)")
    estadisticas_completas(df_viajes["edad"],    "Edad del viajero")
    estadisticas_completas(df_viajes["etapas"],  "Etapas por viaje")

    titulo("1.3  TIEMPO PROMEDIO POR COMUNA")
    por_comuna = df_viajes.groupby("comuna_origen")["tiempo"].mean().sort_values(ascending=False)
    print()
    for comuna, t in por_comuna.head(12).items():
        print(f"    {comuna:<22}: {t:6.1f} min")
    print(f"\n    (promedio general: {df_viajes['tiempo'].mean():.1f} min)")

    titulo("1.4  COVARIANZA Y CORRELACION (vs tiempo)")
    pares = [
        ("dist_km",   "tiempo"),
        ("etapas",    "tiempo"),
        ("edad",      "tiempo"),
        ("es_hombre", "tiempo"),
    ]
    for col1, col2 in pares:
        covarianza_correlacion(df_modelo, col1, col2)

    titulo("1.5  MATRIZ DE CORRELACION")
    print("\n", df_modelo[NUMERICAS].corr().round(3).to_string())

if __name__ == "__main__":
    ejecutar()
