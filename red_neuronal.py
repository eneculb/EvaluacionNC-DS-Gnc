#red neuronal (predecir tiempo de viaje)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from datos import FEATURES
import modelos_pred
from modelos_pred import titulo, evaluar

#keras no usa random_state como sklearn, utilizamos set_seed que es lo mismo
tf.random.set_seed(42)

#modelo
def red_neuronal(d):
    titulo("3.1  RED NEURONAL", nivel=2)

    # capas densas con relu; la ultima es lineal porque predecimos un numero
    modelo = keras.Sequential([
        keras.Input(shape=(len(FEATURES),)),
        layers.Dense(64), layers.BatchNormalization(), layers.Activation("relu"),
        layers.Dropout(0.2),
        layers.Dense(32), layers.Activation("relu"),
        layers.Dropout(0.1),
        layers.Dense(16), layers.Activation("relu"),
        layers.Dense(1),
    ])
    modelo.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])

    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                       restore_best_weights=True)
    hist = modelo.fit(d["X_train"], d["y_train_s"],
                      validation_split=0.15,
                      epochs=200, batch_size=256, verbose=0, callbacks=[es])
    print(f"\n  entreno {len(hist.history['loss'])} epocas")

    #las volvemos a minutos
    sy = d["sy"]
    pred_train = sy.inverse_transform(modelo.predict(d["X_train"], verbose=0)).ravel()
    pred_test  = sy.inverse_transform(modelo.predict(d["X_test"],  verbose=0)).ravel()

    evaluar("Red Neuronal — TRAIN", d["y_train"], pred_train)
    metricas = evaluar("Red Neuronal — TEST",  d["y_test"],  pred_test)
    return modelo, metricas, hist, pred_test

#comparar
def comparar_modelos(metricas_lista):
    titulo("3.5  COMPARATIVA FINAL (TEST SET)", nivel=2)
    tabla = pd.DataFrame(metricas_lista)
    print("\n", tabla[["nombre", "r2", "rmse", "mae"]].to_string(index=False))
    mejor = tabla.loc[tabla["r2"].idxmax(), "nombre"]
    print(f"\n  ✔ Mejor modelo (mayor R²): {mejor}")
    return tabla, mejor


def ejecutar():
    titulo("PARTE 3 — RED NEURONAL: TIEMPO DE VIAJE")

    # mismo split que la parte 2
    d = modelos_pred.preparar_datos()
    print(f"\n  filas -> train (80%): {len(d['y_train'])}   test (20%): {len(d['y_test'])}")

    nn, met_nn, hist, pred_test = red_neuronal(d)
    lr, met_lr = modelos_pred.regresion_lineal(d)
    dt, met_dt = modelos_pred.arbol_decision(d)
    rf, met_rf, _ = modelos_pred.random_forest(d)

    tabla, mejor = comparar_modelos([met_nn, met_lr, met_dt, met_rf])

    return {
        "modelos":  {"nn": nn, "lr": lr, "dt": dt, "rf": rf},
        "datos":    d,
        "metricas": [met_nn, met_lr, met_dt, met_rf],
        "hist":     hist,
        "pred_test": pred_test,
        "tabla":    tabla,
        "mejor":    mejor,
    }

if __name__ == "__main__":
    ejecutar()
