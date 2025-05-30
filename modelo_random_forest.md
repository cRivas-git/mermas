# Resultados de Predicción: Random Forest

## Resumen de Métricas

- **R²**: 0.1758 (Proporción de varianza explicada por el modelo)
- **RMSE**: 1.45 (Error cuadrático medio, en unidades de la variable objetivo)
- **MAE**: 0.70 (Error absoluto medio, en unidades de la variable objetivo)

## Interpretación

El modelo de Random Forest explica aproximadamente el 17.6% de la variabilidad en las mermas. 
En promedio, las predicciones difieren de los valores reales en ±1.45 unidades.

## Muestra de Predicciones (Top 10)

| # | Valor Real | Predicción | Error | Error % | Categoría | Tienda |
|---|------------|------------|--------|---------|-----------|---------|
| 5227 | 12.00 | 1.83 | 10.17 | 84.7% | CONSERVAS DE PESCADOS | TEMUCO III |
| 2623 | 11.00 | 1.03 | 9.97 | 90.7% | CECINAS VACIO | TEMUCO III |
| 3070 | 12.00 | 2.12 | 9.88 | 82.3% | FIDEOS Y PASTAS | ANGOL |
| 3276 | 11.00 | 1.22 | 9.78 | 88.9% | CECINAS VACIO | TEMUCO V |
| 4836 | 1.00 | 9.98 | -8.98 | 897.8% | CUCURBITACEAS | ANGOL |
| 323 | 10.00 | 1.05 | 8.95 | 89.5% | AZUCAR | TEMUCO IV |
| 8732 | 10.00 | 1.07 | 8.93 | 89.3% | SALSAS DE TOMATES | ANGOL |
| 7864 | 10.00 | 1.08 | 8.92 | 89.2% | DETERGENTES | TEMUCO II |
| 6949 | 10.00 | 1.24 | 8.76 | 87.6% | FIDEOS Y PASTAS | TEMUCO III |
| 5159 | 12.00 | 3.40 | 8.60 | 71.6% | SNACKS | TEMUCO V |

## Distribución del Error

- **Error Mínimo**: -8.98
- **Error Máximo**: 10.17
- **Error Promedio**: 0.03
- **Desviación Estándar del Error**: 1.45

*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*

## Visualizaciones
Las siguientes visualizaciones están disponibles en formato PNG:
1. Predicciones vs Valores Reales: `predicciones_vs_reales.png`
2. Análisis de Residuos: `analisis_residuos.png`
3. Distribución de Errores: `distribucion_errores.png`
