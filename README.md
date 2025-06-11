# Análisis Predictivo de Mermas

Este proyecto implementa un análisis predictivo de mermas utilizando diferentes modelos de machine learning:

* Regresión Lineal
* Random Forest
* XGBoost

## Características

* Análisis y preprocesamiento de datos históricos de mermas
* Implementación de tres modelos predictivos
* Evaluación y comparación de modelos
* Generación de visualizaciones y reportes detallados
* Análisis de reglas de asociación

## Requisitos

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost

## Estructura del Proyecto

* `main.py`: Script principal con la implementación de los modelos
* `mermas.csv`: Dataset de mermas (no incluido en el repositorio)
* `*.png`: Visualizaciones generadas
* `*.md`: Reportes detallados de cada modelo y análisis

## Uso

1. Asegúrate de tener todos los requisitos instalados
2. Coloca tu archivo `mermas.csv` en el directorio raíz
3. Ejecuta `python main.py`
4. Revisa los reportes generados en los archivos markdown

## Resultados

El análisis genera varios archivos de visualización y reportes:

* Predicciones vs valores reales
* Análisis de residuos
* Distribución de errores
* Importancia de características
* Reportes detallados por modelo
* Análisis de reglas de asociación 