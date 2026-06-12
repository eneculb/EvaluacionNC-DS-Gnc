import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from datos import df_comunas, FEATURES, TARGET


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
    print(f"    RMSE  : {rmse:.4f}")
    print(f"    MAE   : {mae:.4f}")
    print(f"    R²    : {r2:.4f}  ({r2*100:.2f}% varianza explicada)")
    return {"nombre": nombre, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def preparar_datos():
    X = df_comunas[FEATURES].values
    y = df_comunas[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # escalado necesario para redes neuronales
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, scaler


def construir_red(X_train, y_train):
    red = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.005,
        max_iter=1500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=42,
    )
    red.fit(X_train, y_train)
    return red


def analizar_arquitectura(red):
    titulo("3.2  ARQUITECTURA DE LA RED", nivel=2)
    capas = [len(FEATURES)] + list(red.hidden_layer_sizes) + [1]

    print(f"\n  Capa de entrada   : {capas[0]} neuronas ({len(FEATURES)} features)")
    for i, n in enumerate(capas[1:-1], 1):
        print(f"  Capa oculta {i}     : {n} neuronas")
    print(f"  Capa de salida    : {capas[-1]} neurona")

    total_params = sum(
        capas[i] * capas[i+1] + capas[i+1]
        for i in range(len(capas) - 1)
    )
    print(f"\n  Parámetros totales: {total_params}")
    print(f"  Iteraciones       : {red.n_iter_}")
    print(f"  Función activación: {red.activation}")
    print(f"  Optimizador       : {red.solver}")


def importancia_por_permutacion(red, X_test, y_test):
    titulo("3.3  IMPORTANCIA DE VARIABLES (Permutación)", nivel=2)

    r2_base = r2_score(y_test, red.predict(X_test))
    importancias = []

    rng = np.random.default_rng(42)
    for i, feat in enumerate(FEATURES):
        X_perm = X_test.copy()
        rng.shuffle(X_perm[:, i])
        r2_perm = r2_score(y_test, red.predict(X_perm))
        importancias.append((feat, r2_base - r2_perm))

    importancias.sort(key=lambda x: x[1], reverse=True)

    print()
    for feat, imp in importancias:
        barra = "█" * max(0, int(imp * 100))
        print(f"    {feat:<25}: {imp:+.4f}  {barra}")

    return importancias


def predecir_escenarios(red, scaler):
    titulo("3.4  PREDICCIÓN DE ESCENARIOS", nivel=2)

    escenarios = pd.DataFrame([
        {"es_hombre": 1, "edad_tramo": 1, "dias_teletrabajo": 0,
         "ingreso_percentil": 3.5, "modo_principal": 3, "proposito_viaje": 0,
         "tiempo_viaje_min": 42, "descripcion": "Hombre, sin teletrabajo, metro"},
        {"es_hombre": 0, "edad_tramo": 2, "dias_teletrabajo": 5,
         "ingreso_percentil": 2.0, "modo_principal": 1, "proposito_viaje": 1,
         "tiempo_viaje_min": 55, "descripcion": "Mujer, teletrabajo full, caminata"},
        {"es_hombre": 1, "edad_tramo": 3, "dias_teletrabajo": 2,
         "ingreso_percentil": 4.5, "modo_principal": 0, "proposito_viaje": 0,
         "tiempo_viaje_min": 45, "descripcion": "Hombre mayor, auto, ingreso alto"},
        {"es_hombre": 0, "edad_tramo": 0, "dias_teletrabajo": 0,
         "ingreso_percentil": 1.5, "modo_principal": 2, "proposito_viaje": 3,
         "tiempo_viaje_min": 52, "descripcion": "Mujer joven, bus, estudia"},
    ])

    X_esc    = scaler.transform(escenarios[FEATURES].values)
    pred_nn  = red.predict(X_esc)

    print(f"\n  {'Escenario':<45} {'Red Neuronal':>13}")
    print(f"  {'-'*45} {'-'*13}")
    for i, row in escenarios.iterrows():
        print(f"  {row['descripcion']:<45} {pred_nn[i]:>13.3f}")

    return escenarios, pred_nn


def curva_aprendizaje(red):
    titulo("3.5  EVOLUCIÓN DEL ENTRENAMIENTO", nivel=2)
    perdidas = red.loss_curve_

    print(f"\n  Pérdida inicial  : {perdidas[0]:.4f}")
    print(f"  Pérdida final    : {perdidas[-1]:.4f}")
    print(f"  Reducción total  : {(1 - perdidas[-1]/perdidas[0])*100:.2f}%")
    print(f"  Épocas ejecutadas: {len(perdidas)}")

    return perdidas


def ejecutar():
    titulo("PARTE 3 — RED NEURONAL ARTIFICIAL (MLP)")

    X_train, X_test, y_train, y_test, scaler = preparar_datos()

    titulo("3.1  ENTRENAMIENTO DE LA RED", nivel=2)
    red = construir_red(X_train, y_train)
    evaluar("Red Neuronal — TRAIN", y_train, red.predict(X_train))
    metricas = evaluar("Red Neuronal — TEST",  y_test,  red.predict(X_test))

    analizar_arquitectura(red)
    importancias = importancia_por_permutacion(red, X_test, y_test)
    escenarios, pred_nn = predecir_escenarios(red, scaler)
    perdidas = curva_aprendizaje(red)

    return {
        "modelo":       red,
        "scaler":       scaler,
        "splits":       (X_train, X_test, y_train, y_test),
        "metricas":     metricas,
        "importancias": importancias,
        "escenarios":   (escenarios, pred_nn),
        "perdidas":     perdidas,
    }


if __name__ == "__main__":
    ejecutar()
