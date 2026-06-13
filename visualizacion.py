# figuras

import numpy as np
import matplotlib.pyplot as plt
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


# f1
def fig1_distribuciones():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FIGURA 1 — Distribución del Tiempo y la Distancia de Viaje",
                 fontsize=13, color="#e0e0e0", fontweight="bold")

    ax = axes[0]
    ax.hist(df_viajes["tiempo"], bins=40, color="#667eea", alpha=0.85)
    ax.axvline(df_viajes["tiempo"].mean(), color="#ffd93d", ls="--", lw=1.5,
               label=f"Media = {df_viajes['tiempo'].mean():.1f} min")
    ax.set_xlabel("Tiempo de viaje (min)")
    ax.set_ylabel("Cantidad de viajes")
    ax.set_title("Tiempo de viaje")
    ax.legend(fontsize=8); ax.grid(axis="y")

    ax = axes[1]
    ax.hist(df_viajes["dist_km"], bins=40, color="#4ecdc4", alpha=0.85)
    ax.axvline(df_viajes["dist_km"].mean(), color="#ffd93d", ls="--", lw=1.5,
               label=f"Media = {df_viajes['dist_km'].mean():.1f} km")
    ax.set_xlabel("Distancia de viaje (km)")
    ax.set_ylabel("Cantidad de viajes")
    ax.set_title("Distancia de viaje")
    ax.legend(fontsize=8); ax.grid(axis="y")

    plt.tight_layout(); plt.show()


# f2 - tiempo por comuna
def fig2_tiempo_comuna():
    por_comuna = (df_viajes.groupby("comuna_origen")["tiempo"]
                  .mean().sort_values().tail(15))
    media = df_viajes["tiempo"].mean()
    col = ["#ff6b6b" if t > media else "#667eea" for t in por_comuna.values]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle("FIGURA 2 — Tiempo promedio de viaje por comuna de origen",
                 fontsize=13, color="#e0e0e0", fontweight="bold")
    ax.barh(por_comuna.index, por_comuna.values, color=col)
    ax.axvline(media, color="#ffd93d", ls="--", lw=1.5, label=f"Promedio = {media:.1f} min")
    ax.set_xlabel("Tiempo promedio (min)")
    ax.legend(fontsize=8); ax.grid(axis="x")
    plt.tight_layout(); plt.show()


# f3
def fig3_correlacion():
    matriz_corr = df_modelo[NUMERICAS].corr()
    labels = ["Tiempo", "Distancia", "Edad", "Etapas", "Es hombre"]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("FIGURA 3 — Matriz de Correlación (variables numéricas)",
                 fontsize=12, color="#e0e0e0", fontweight="bold")
    im = ax.imshow(matriz_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matriz_corr.iloc[i, j]
            c = "black" if abs(val) > 0.5 else "#e0e0e0"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=c, fontweight="bold")
    plt.tight_layout(); plt.show()


# f4
def fig4_comparativa_modelos(resultados):
    d = resultados["datos"]; modelos = resultados["modelos"]
    y_test = d["y_test"]
    # muestra de 2000 puntos
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_test), size=min(2000, len(y_test)), replace=False)

    info = [
        ("Regresión Lineal",  modelos["lr"].predict(d["X_test"]), "#667eea"),
        ("Árbol de Decisión", modelos["dt"].predict(d["X_test"]), "#4ecdc4"),
        ("Random Forest",     modelos["rf"].predict(d["X_test"]), "#ffd93d"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("FIGURA 4 — Comparativa de Modelos (Test Set)",
                 fontsize=13, color="#e0e0e0", fontweight="bold")
    for ax, (nombre, y_pred, color) in zip(axes, info):
        r2_v = r2_score(y_test, y_pred)
        ax.scatter(y_test[idx], y_pred[idx], alpha=0.35, color=color, s=14, edgecolors="none")
        lims = [0, 180]
        ax.plot(lims, lims, "w--", lw=1.2, alpha=0.7, label="Predicción perfecta")
        ax.set_xlabel("Tiempo real (min)"); ax.set_ylabel("Tiempo predicho (min)")
        ax.set_title(f"{nombre}\nR² = {r2_v:.3f}")
        ax.legend(fontsize=8); ax.grid(True)
    plt.tight_layout(); plt.show()


# f5
def fig5_importancia(resultados):
    importancias = resultados["importancias"][:10]
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("FIGURA 5 — Importancia de Variables (Random Forest, top 10)",
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


# f6
def fig6_residuales(resultados):
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
    fig.suptitle("FIGURA 6 — Análisis de Residuales por Modelo",
                 fontsize=13, color="#e0e0e0", fontweight="bold")
    for ax, (nombre, y_pred, color) in zip(axes, info):
        residuales = y_test - y_pred
        ax.scatter(y_pred[idx], residuales[idx], alpha=0.35, color=color, s=14, edgecolors="none")
        ax.axhline(0, color="white", lw=1.5, ls="--")
        sigma = residuales.std()
        ax.axhline( sigma, color="#ffd93d", lw=1, ls=":", label=f"+1σ = {sigma:.1f} min")
        ax.axhline(-sigma, color="#ffd93d", lw=1, ls=":")
        ax.set_xlabel("Tiempo predicho (min)"); ax.set_ylabel("Residual (min)")
        ax.set_title(f"Residuales — {nombre}")
        ax.legend(fontsize=8); ax.grid(True)
    plt.tight_layout(); plt.show()

#funcion principal
def generar_todas(resultados):
    print("\n  Mostrando figuras...")
    fig1_distribuciones()
    fig2_tiempo_comuna()
    fig3_correlacion()
    fig4_comparativa_modelos(resultados)
    fig5_importancia(resultados)
    fig6_residuales(resultados)
