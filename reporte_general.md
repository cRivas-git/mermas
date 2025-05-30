# Reporte General de Análisis Predictivo de Mermas

## Resumen Ejecutivo

El análisis predictivo de mermas ha evaluado tres modelos diferentes: Regresión Lineal, Random Forest y XGBoost.
El mejor modelo identificado es **Regresión Lineal** con un R² de 0.2315.

## Comparación de Modelos

### Tabla de Métricas
| Modelo | R² | RMSE | MAE |
|--------|-----|------|-----|
| Regresión Lineal | 0.2315 | 1.40 | 0.75 |
| Random Forest | 0.1758 | 1.45 | 0.70 |
| XGBoost | 0.1993 | 1.43 | 0.71 |

### Interpretación General

1. **Mejor Modelo: Regresión Lineal**
   - R² = 0.2315 (Explica el 23.2% de la variabilidad)
   - RMSE = 1.40 (Error promedio en unidades de merma)

2. **Comparación de Modelos**
   - La diferencia en rendimiento entre los modelos es de 5.6 puntos porcentuales en R²
   - Todos los modelos muestran un RMSE entre 1.40 y 1.45 unidades

## Archivos de Análisis Individual
- [Regresión Lineal](modelo_regresión_lineal.md)
- [Random Forest](modelo_random_forest.md)
- [XGBoost](modelo_xgboost.md)

## Visualizaciones Generadas
1. **predicciones_vs_reales.png**: Comparación de predicciones vs valores reales para cada modelo
2. **analisis_residuos.png**: Análisis de residuos para cada modelo
3. **distribucion_errores.png**: Distribución de errores para cada modelo
4. **importancia_caracteristicas.png**: Importancia de las características (para Random Forest)
