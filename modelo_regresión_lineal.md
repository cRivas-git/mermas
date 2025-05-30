# Resultados de Predicción: Regresión Lineal

## Resumen de Métricas

- **R²**: 0.2315 (Proporción de varianza explicada por el modelo)
- **RMSE**: 1.40 (Error cuadrático medio, en unidades de la variable objetivo)
- **MAE**: 0.75 (Error absoluto medio, en unidades de la variable objetivo)

## Interpretación

El modelo de Regresión Lineal explica aproximadamente el 23.2% de la variabilidad en las mermas. 
En promedio, las predicciones difieren de los valores reales en ±1.40 unidades.

## Muestra de Predicciones (Top 10)

| # | Valor Real | Predicción | Error | Error % | Categoría | Tienda |
|---|------------|------------|--------|---------|-----------|---------|
| 3070 | 12.00 | 1.51 | 10.49 | 87.5% | FIDEOS Y PASTAS | ANGOL |
| 5227 | 12.00 | 1.90 | 10.10 | 84.2% | CONSERVAS DE PESCADOS | TEMUCO III |
| 1579 | 12.00 | 2.31 | 9.69 | 80.7% | YOGHURT | TEMUCO IV |
| 5157 | 12.00 | 2.41 | 9.59 | 79.9% | SNACKS | ANGOL |
| 5159 | 12.00 | 2.46 | 9.54 | 79.5% | SNACKS | TEMUCO V |
| 7864 | 10.00 | 0.93 | 9.07 | 90.7% | DETERGENTES | TEMUCO II |
| 1761 | 11.00 | 2.01 | 8.99 | 81.7% | HARINAS | TEMUCO IV |
| 331 | 11.00 | 2.19 | 8.81 | 80.1% | AZUCAR | TEMUCO V |
| 6949 | 10.00 | 1.28 | 8.72 | 87.2% | FIDEOS Y PASTAS | TEMUCO III |
| 7923 | 11.00 | 2.29 | 8.71 | 79.2% | BULBOS | ANGOL |

## Distribución del Error

- **Error Mínimo**: -6.79
- **Error Máximo**: 10.49
- **Error Promedio**: 0.01
- **Desviación Estándar del Error**: 1.40

*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*

## Visualizaciones
Las siguientes visualizaciones están disponibles en formato PNG:
1. Predicciones vs Valores Reales: `predicciones_vs_reales.png`
2. Análisis de Residuos: `analisis_residuos.png`
3. Distribución de Errores: `distribucion_errores.png`
