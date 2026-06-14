import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from datos import df_modelo, FEATURES, TARGET

def titulo(texto, nivel=1):
    if nivel == 1:
        linea = "=" * 60
        print(f"\n{linea}\n  {texto}\n{linea}")
    else:
        linea = "-" * 40
        print(f"\n{linea}\n  {texto}\n{linea}")

def evaluar(nombre, y_real, y_pred):
    mse  = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_real, y_pred)
    r2   = r2_score(y_real, y_pred)
    print(f"\n  [{nombre}]")
    print(f"    MSE   : {mse:.4f}")
    print(f"    RMSE  : {rmse:.4f} min")
    print(f"    MAE   : {mae:.4f} min")
    print(f"    R²    : {r2:.4f}  ({r2*100:.2f}% varianza explicada)")
    return {"nombre": nombre, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

def preparar_datos():
    X = df_modelo[FEATURES].values
    y = df_modelo[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    sx = StandardScaler().fit(X_train)
    X_train = sx.transform(X_train)
    X_test  = sx.transform(X_test)

    sy = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = sy.transform(y_train.reshape(-1, 1)).ravel()

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_train_s": y_train_s, "sy": sy,
    }

def regresion_lineal(d):
    titulo("2.1  REGRESIÓN LINEAL MÚLTIPLE", nivel=2)
    lr = LinearRegression()
    lr.fit(d["X_train"], d["y_train"])
    evaluar("Regresión Lineal — TRAIN", d["y_train"], lr.predict(d["X_train"]))
    metricas = evaluar("Regresión Lineal — TEST", d["y_test"], lr.predict(d["X_test"]))
    return lr, metricas

def arbol_decision(d):
    titulo("2.2  ÁRBOL DE DECISIÓN", nivel=2)
    dt = DecisionTreeRegressor(max_depth=8, random_state=42)
    dt.fit(d["X_train"], d["y_train"])
    evaluar("Árbol de Decisión — TRAIN", d["y_train"], dt.predict(d["X_train"]))
    metricas = evaluar("Árbol de Decisión — TEST", d["y_test"], dt.predict(d["X_test"]))
    return dt, metricas

def random_forest(d):
    titulo("2.3  RANDOM FOREST", nivel=2)
    rf = RandomForestRegressor(n_estimators=100, max_depth=16,
                               random_state=42, n_jobs=-1)
    rf.fit(d["X_train"], d["y_train"])
    evaluar("Random Forest — TRAIN", d["y_train"], rf.predict(d["X_train"]))
    metricas = evaluar("Random Forest — TEST", d["y_test"], rf.predict(d["X_test"]))

    print("\n  Importancia de variables (top 8):")
    importancias = sorted(zip(FEATURES, rf.feature_importances_),
                          key=lambda x: x[1], reverse=True)
    for feat, imp in importancias[:8]:
        barra = "█" * int(imp * 40)
        print(f"    {feat:<22}: {imp:.4f}  {barra}")
    return rf, metricas, importancias

def comparar_modelos(metricas_lista):
    titulo("2.4  COMPARATIVA DE MODELOS (TEST SET)", nivel=2)
    tabla = pd.DataFrame(metricas_lista)
    print("\n", tabla[["nombre", "r2", "rmse", "mae"]].to_string(index=False))
    mejor = tabla.loc[tabla["r2"].idxmax(), "nombre"]
    print(f"\n  ✔ Mejor modelo (mayor R²): {mejor}")
    return mejor

def ejecutar():
    titulo("PARTE 2 — MODELOS PREDICTIVOS DEL TIEMPO DE VIAJE")

    d = preparar_datos()
    print(f"\n  filas -> train (80%): {len(d['y_train'])}   test (20%): {len(d['y_test'])}")

    lr, met_lr = regresion_lineal(d)
    dt, met_dt = arbol_decision(d)
    rf, met_rf, importancias = random_forest(d)

    mejor = comparar_modelos([met_lr, met_dt, met_rf])

    return {
        "modelos":      {"lr": lr, "dt": dt, "rf": rf},
        "datos":        d,
        "metricas":     [met_lr, met_dt, met_rf],
        "importancias": importancias,
        "mejor":        mejor,
    }

if __name__ == "__main__":
    ejecutar()
