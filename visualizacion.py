# figuras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

from datos import df_viajes, df_modelo, NUMERICAS

plt.rcParams.update({
    "figure.facecolor": "#0f1626",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#2a2a3e",
    "axes.labelcolor":  "#c0c8d8",
    "xtick.color":      "#c0c8d8",
    "ytick.color":      "#c0c8d8",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2a3e",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})

PALETTE = [
    "#667eea", "#4ecdc4", "#ffd93d", "#ff6b6b", "#a8dadc",
    "#764ba2", "#f093fb", "#43e97b", "#fa709a", "#fee140",
]


# f1 - histogramas de tiempo y distancia
def fig1_distribuciones():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FIGURA 1 — Distribución del Tiempo y la Distancia",
                 fontsize=13, color="#e0e0e0", fontweight="bold")

    ax = axes[0]
    ax.hist(df_viajes["tiempo"], bins=40, color="#667eea", alpha=0.85)
    ax.axvline(df_viajes["tiempo"].mean(), color="#ffd93d", ls="--", lw=1.5,
               label=f"Media = {df_viajes['tiempo'].mean():.1f} min")
    ax.set_xlabel("Tiempo (min)"); ax.set_ylabel("Viajes")
    ax.set_title("Tiempo de viaje"); ax.legend(fontsize=8); ax.grid(axis="y")

    ax = axes[1]
    ax.hist(df_viajes["dist_km"], bins=40, color="#4ecdc4", alpha=0.85)
    ax.axvline(df_viajes["dist_km"].mean(), color="#ffd93d", ls="--", lw=1.5,
               label=f"Media = {df_viajes['dist_km'].mean():.1f} km")
    ax.set_xlabel("Distancia (km)"); ax.set_ylabel("Viajes")
    ax.set_title("Distancia de viaje"); ax.legend(fontsize=8); ax.grid(axis="y")

    plt.tight_layout(); plt.show()


# f2 - boxplot del tiempo segun proposito
def fig2_boxplot():
    propositos = df_viajes["proposito"].unique()
    datos = [df_viajes[df_viajes["proposito"] == p]["tiempo"].values for p in propositos]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("FIGURA 2 — Tiempo de viaje según propósito (boxplot)",
                 fontsize=13, color="#e0e0e0", fontweight="bold")
    bp = ax.boxplot(datos, labels=propositos, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#ffd93d", lw=2))
    for caja, color in zip(bp["boxes"], PALETTE):
        caja.set(facecolor=color, alpha=0.7)
    ax.set_ylabel("Tiempo (min)"); ax.grid(axis="y")
    plt.tight_layout(); plt.show()


# f3 - barras (vertical) y linea
def fig3_barras_linea():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("FIGURA 3 — Tiempo por propósito y por edad",
                 fontsize=13, color="#e0e0e0", fontweight="bold")

    # barras verticales: tiempo promedio por proposito
    ax = axes[0]
    prom_prop = df_viajes.groupby("proposito")["tiempo"].mean().sort_values()
    ax.bar(prom_prop.index, prom_prop.values, color=PALETTE[:len(prom_prop)])
    ax.set_ylabel("Tiempo promedio (min)")
    ax.set_title("Por propósito de viaje")
    ax.tick_params(axis="x", rotation=20); ax.grid(axis="y")

    # linea: tiempo promedio por tramo de edad
    ax = axes[1]
    df_viajes["tramo_edad"] = (df_viajes["edad"] // 10) * 10
    prom_edad = df_viajes.groupby("tramo_edad")["tiempo"].mean()
    prom_edad = prom_edad[prom_edad.index <= 80]
    ax.plot(prom_edad.index, prom_edad.values, "o-", color="#4ecdc4", lw=2, ms=6)
    ax.set_xlabel("Edad"); ax.set_ylabel("Tiempo promedio (min)")
    ax.set_title("Por edad del viajero"); ax.grid(True)

    plt.tight_layout(); plt.show()


# f4 - tiempo promedio por comuna (barh)
def fig4_tiempo_comuna():
    por_comuna = df_viajes.groupby("comuna_origen")["tiempo"].mean().sort_values().tail(15)
    media = df_viajes["tiempo"].mean()
    col = ["#ff6b6b" if t > media else "#667eea" for t in por_comuna.values]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle("FIGURA 4 — Tiempo promedio de viaje por comuna",
                 fontsize=13, color="#e0e0e0", fontweight="bold")
    ax.barh(por_comuna.index, por_comuna.values, color=col)
    ax.axvline(media, color="#ffd93d", ls="--", lw=1.5, label=f"Promedio = {media:.1f} min")
    ax.set_xlabel("Tiempo promedio (min)"); ax.legend(fontsize=8); ax.grid(axis="x")
    plt.tight_layout(); plt.show()


# f5 - mapa de calor de correlacion (seaborn)
def fig5_correlacion():
    matriz = df_modelo[NUMERICAS].corr()
    labels = ["Tiempo", "Distancia", "Edad", "Etapas", "Es hombre"]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("FIGURA 5 — Matriz de correlación",
                 fontsize=12, color="#e0e0e0", fontweight="bold")
    sns.heatmap(matriz, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.tight_layout(); plt.show()


# f6 - comparativa de modelos (scatter real vs predicho)
def fig6_comparativa_modelos(resultados):
    d = resultados["datos"]; modelos = resultados["modelos"]
    y_test = d["y_test"]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_test), size=min(2000, len(y_test)), replace=False)

    info = [
        ("Regresión Lineal",  modelos["lr"].predict(d["X_test"]), "#667eea"),
        ("Árbol de Decisión", modelos["dt"].predict(d["X_test"]), "#4ecdc4"),
        ("Random Forest",     modelos["rf"].predict(d["X_test"]), "#ffd93d"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("FIGURA 6 — Comparativa de Modelos (Test Set)",
                 fontsize=13, color="#e0e0e0", fontweight="bold")
    for ax, (nombre, y_pred, color) in zip(axes, info):
        r2_v = r2_score(y_test, y_pred)
        ax.scatter(y_test[idx], y_pred[idx], alpha=0.35, color=color, s=14, edgecolors="none")
        ax.plot([0, 180], [0, 180], "w--", lw=1.2, alpha=0.7, label="Predicción perfecta")
        ax.set_xlabel("Tiempo real (min)"); ax.set_ylabel("Tiempo predicho (min)")
        ax.set_title(f"{nombre}\nR² = {r2_v:.3f}"); ax.legend(fontsize=8); ax.grid(True)
    plt.tight_layout(); plt.show()


# f7 - importancia de variables (barh)
def fig7_importancia(resultados):
    importancias = resultados["importancias"][:10]
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("FIGURA 7 — Importancia de Variables (Random Forest)",
                 fontsize=13, color="#e0e0e0", fontweight="bold")
    feats_sorted = sorted(importancias, key=lambda x: x[1])
    labels_f = [f.replace("_", " ") for f, _ in feats_sorted]
    vals_f   = [v for _, v in feats_sorted]
    bars = ax.barh(labels_f, vals_f, color=[PALETTE[i % len(PALETTE)] for i in range(len(vals_f))])
    ax.set_xlabel("Importancia")
    for b in bars:
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height()/2,
                f"{b.get_width():.3f}", va="center", fontsize=8)
    ax.grid(axis="x")
    plt.tight_layout(); plt.show()


# f8 - residuales (scatter)
def fig8_residuales(resultados):
    d = resultados["datos"]; modelos = resultados["modelos"]
    y_test = d["y_test"]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_test), size=min(2000, len(y_test)), replace=False)

    info = [
        ("Regresión Lineal",  modelos["lr"].predict(d["X_test"]), "#667eea"),
        ("Árbol de Decisión", modelos["dt"].predict(d["X_test"]), "#4ecdc4"),
        ("Random Forest",     modelos["rf"].predict(d["X_test"]), "#ffd93d"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("FIGURA 8 — Residuales por Modelo",
                 fontsize=13, color="#e0e0e0", fontweight="bold")
    for ax, (nombre, y_pred, color) in zip(axes, info):
        residuales = y_test - y_pred
        ax.scatter(y_pred[idx], residuales[idx], alpha=0.35, color=color, s=14, edgecolors="none")
        ax.axhline(0, color="white", lw=1.5, ls="--")
        sigma = residuales.std()
        ax.axhline( sigma, color="#ffd93d", lw=1, ls=":", label=f"+1σ = {sigma:.1f} min")
        ax.axhline(-sigma, color="#ffd93d", lw=1, ls=":")
        ax.set_xlabel("Tiempo predicho (min)"); ax.set_ylabel("Residual (min)")
        ax.set_title(f"Residuales — {nombre}"); ax.legend(fontsize=8); ax.grid(True)
    plt.tight_layout(); plt.show()

#funcion principal
def generar_todas(resultados):
    print("\n  Mostrando figuras...")
    fig1_distribuciones()
    fig2_boxplot()
    fig3_barras_linea()
    fig4_tiempo_comuna()
    fig5_correlacion()
    fig6_comparativa_modelos(resultados)
    fig7_importancia(resultados)
    fig8_residuales(resultados)
