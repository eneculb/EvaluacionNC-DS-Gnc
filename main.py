import est_desc
import modelos_pred
import visualizacion

try:
    import red_neuronal
    red_enabled = True
except ModuleNotFoundError:
    red_neuronal = None
    red_enabled = False

def banner(texto):
    linea = "█" * 62
    print(f"\n{linea}")
    print(f"  {texto}")
    print(linea)

if __name__ == "__main__":
    banner("ANÁLISIS DE MOVILIDAD URBANA — GRAN SANTIAGO")
    print("  Fuente: Encuesta Origen-Destino (EOD) Santiago 2012 — SECTRA\n")

    banner("PARTE 1 — ESTADÍSTICA DESCRIPTIVA")
    est_desc.ejecutar()

    banner("PARTE 2 — MODELOS PREDICTIVOS")
    resultados = modelos_pred.ejecutar()

    resultados_nn = None
    if red_enabled:
        banner("PARTE 3 — RED NEURONAL")
        resultados_nn = red_neuronal.ejecutar()

    banner("GENERANDO FIGURAS")
    visualizacion.generar_todas(resultados)
    if resultados_nn is not None:
        visualizacion.fig6_comparativa_modelos(resultados_nn)
        visualizacion.fig8_residuales(resultados_nn)

    banner("RESUMEN EJECUTIVO")
    mejor    = resultados["mejor"]
    metricas = resultados["metricas"]
    imps     = resultados["importancias"]

    if resultados_nn is not None:
        mejor = resultados_nn["mejor"]
        metricas = resultados_nn["metricas"]

    print("  PARTE 2 — Métricas en test set")
    print("  ──────────────────────────────────")
    for m in metricas:
        print(f"  {m['nombre']:<28} R²={m['r2']:+.3f}  RMSE={m['rmse']:.2f} min")

    print(f"\n  ✔ Mejor modelo: {mejor}")

    print("  PARTE 2 y 3 — Métricas en test set")
    print("  ──────────────────────────────────")
    for m in metricas:
        print(f"  {m['nombre']:<28} R²={m['r2']:+.3f}  RMSE={m['rmse']:.2f} min")

    print(f"\n  ✔ Mejor modelo: {mejor}")
    print("\n  Top 3 variables más importantes (Random Forest):")
    for feat, imp in imps[:3]:
        print(f"    • {feat}: {imp:.3f}")

    print("""
  CONCLUSIONES
  ─────────────
  1. El tiempo de viaje promedio en el Gran Santiago es de ~41 minutos.
  2. La distancia es, por lejos, la variable que más explica el tiempo de viaje.
  3. Hay diferencias por comuna: las comunas periféricas (La Pintana, Padre
     Hurtado) tienen tiempos de viaje más altos que el promedio.
  4. El modo de transporte también influye (las combinaciones con Metro y los
     viajes en bus toman más tiempo que la caminata o el auto).
  5. Los modelos explican ~60% de la varianza del tiempo de viaje.
""")
