# Resultados de Predicción: XGBoost

## Resumen de Métricas

- **R²**: 0.1993 (Proporción de varianza explicada por el modelo)
- **RMSE**: 1.43 (Error cuadrático medio, en unidades de la variable objetivo)
- **MAE**: 0.71 (Error absoluto medio, en unidades de la variable objetivo)

## Interpretación

El modelo de XGBoost explica aproximadamente el 19.9% de la variabilidad en las mermas. 
En promedio, las predicciones difieren de los valores reales en ±1.43 unidades.

## Muestra de Predicciones (Top 10)

| # | Valor Real | Predicción | Error | Error % | Categoría | Tienda |
|---|------------|------------|--------|---------|-----------|---------|
| 3070 | 12.00 | 1.43 | 10.57 | 88.1% | FIDEOS Y PASTAS | ANGOL |
| 5227 | 12.00 | 1.49 | 10.51 | 87.6% | CONSERVAS DE PESCADOS | TEMUCO III |
| 1579 | 12.00 | 2.26 | 9.74 | 81.2% | YOGHURT | TEMUCO IV |
| 5159 | 12.00 | 2.39 | 9.61 | 80.1% | SNACKS | TEMUCO V |
| 2623 | 11.00 | 1.68 | 9.32 | 84.8% | CECINAS VACIO | TEMUCO III |
| 8732 | 10.00 | 0.80 | 9.20 | 92.0% | SALSAS DE TOMATES | ANGOL |
| 7864 | 10.00 | 1.17 | 8.83 | 88.3% | DETERGENTES | TEMUCO II |
| 331 | 11.00 | 2.23 | 8.77 | 79.7% | AZUCAR | TEMUCO V |
| 6949 | 10.00 | 1.29 | 8.71 | 87.1% | FIDEOS Y PASTAS | TEMUCO III |
| 4836 | 1.00 | 9.68 | -8.68 | 867.8% | CUCURBITACEAS | ANGOL |

## Distribución del Error

- **Error Mínimo**: -8.68
- **Error Máximo**: 10.57
- **Error Promedio**: 0.01
- **Desviación Estándar del Error**: 1.43

*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*

## Visualizaciones
Las siguientes visualizaciones están disponibles en formato PNG:
1. Predicciones vs Valores Reales: `predicciones_vs_reales.png`
2. Análisis de Residuos: `analisis_residuos.png`
3. Distribución de Errores: `distribucion_errores.png`
