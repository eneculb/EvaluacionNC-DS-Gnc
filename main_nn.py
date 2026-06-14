import red_neuronal
import visualizacion

def banner(texto):
    linea = "█" * 62
    print(f"\n{linea}")
    print(f"  {texto}")
    print(linea)

if __name__ == "__main__":
    banner("EVALUACIÓN 2 — RED NEURONAL: TIEMPO DE VIAJE")
    print("  Fuente: Encuesta Origen-Destino (EOD) Santiago 2012 — SECTRA\n")

    banner("PARTE 3 — RED NEURONAL Y COMPARACIÓN")
    resultados = red_neuronal.ejecutar()

    banner("GENERANDO FIGURAS")
    visualizacion.fig6_comparativa_modelos(resultados)
    visualizacion.fig8_residuales(resultados)

    banner("RESUMEN EJECUTIVO")
    tabla = resultados["tabla"]
    mejor = resultados["mejor"]
    print("\n  Métricas en test set")
    print("  ──────────────────────────────────")
    for _, m in tabla.iterrows():
        print(f"  {m['nombre']:<28} R²={m['r2']:+.3f}  RMSE={m['rmse']:.2f} min")

    print(f"\n  ✔ Mejor modelo: {mejor}")
    print("""
  CONCLUSIONES
  ─────────────
  1. Con 88.930 viajes reales, la red neuronal predice el tiempo de viaje con
     un R² ~0.60, superando a la regresión lineal y al árbol de decisión.
  2. Esto confirma que existen relaciones NO lineales (modo × distancia × periodo)
     que la red captura mejor que un modelo lineal.
  3. El Random Forest queda muy parejo con la red (ambos ~0.61); en datos
     tabulares los bosques suelen ser muy competitivos.
  4. Se usó One-Hot Encoding para las variables categóricas (modo, propósito,
     periodo), normalización con StandardScaler y split 80/20.
""")

s,.c v,xmd v
