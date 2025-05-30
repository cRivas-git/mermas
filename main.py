# IMPLEMENTACIÓN DE ANÁLISIS PREDICTIVO PARA MERMAS
# Utilizamos mermas.csv que contiene datos históricos de mermas

# PASO 1: IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime as dt

print("*IMPLEMENTACIÓN DE MODELOS PREDICTIVOS PARA MERMAS*")

# PASO 2: CARGA Y PREPARACIÓN DE DATOS
# Cargar el dataset
data = pd.read_csv('mermas.csv', sep=';', encoding='latin1')

# Convertir fechas a formato datetime
data['fecha'] = pd.to_datetime(data['fecha'], format='%d-%m-%Y')

# Crear nuevas características para las fechas
data['dia_semana'] = data['fecha'].dt.dayofweek
data['dia_mes'] = data['fecha'].dt.day

# Convertir columnas numéricas (reemplazar comas por puntos)
def clean_numeric(x):
    if isinstance(x, str):
        return float(x.replace(',', '.'))
    return float(x)

# Limpiar y convertir columnas numéricas
numeric_columns = ['merma_unidad', 'merma_monto', 'merma_unidad_p', 'merma_monto_p']
for col in numeric_columns:
    data[col] = data[col].apply(clean_numeric)

# PASO 3: DETECCIÓN Y MANEJO DE OUTLIERS
def remove_outliers(df, column, n_sigmas=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores < n_sigmas]

# Remover outliers de merma_monto y merma_unidad
print("\nTamaño del dataset antes de remover outliers:", len(data))
data = remove_outliers(data, 'merma_monto')
data = remove_outliers(data, 'merma_unidad')
print("Tamaño del dataset después de remover outliers:", len(data))

# PASO 4: SELECCIÓN DE CARACTERÍSTICAS
# Características para predecir mermas
features = ['negocio', 'seccion', 'linea', 'categoria', 'abastecimiento', 
           'comuna', 'region', 'tienda', 'motivo', 'ubicación_motivo',
           'dia_semana', 'dia_mes', 'mes']

X = data[features]
# Usaremos merma_unidad_p como variable objetivo (porcentaje de merma por unidad)
y = data['merma_unidad_p']

# PASO 5: DIVISIÓN DE DATOS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PASO 6: PREPROCESAMIENTO
# Definir qué variables son categóricas y numéricas
categorical_features = ['negocio', 'seccion', 'linea', 'categoria', 'abastecimiento', 
                       'comuna', 'region', 'tienda', 'motivo', 'ubicación_motivo', 'mes']
numeric_features = ['dia_semana', 'dia_mes']

# Crear preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# PASO 7: IMPLEMENTACIÓN DE MODELOS
# Modelo 1: Regresión Lineal
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Modelo 2: Random Forest
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Modelo 3: XGBoost
pipeline_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
])

# PASO 8: ENTRENAMIENTO DE MODELOS
print("\nEntrenando modelos...")
print("Entrenando Regresión Lineal...")
pipeline_lr.fit(X_train, y_train)

print("Entrenando Random Forest...")
pipeline_rf.fit(X_train, y_train)

print("Entrenando XGBoost...")
pipeline_xgb.fit(X_train, y_train)

print("Modelos entrenados correctamente")

# -------------------------------------------------
# EVALUACIÓN DE LOS MODELOS
# -------------------------------------------------

print("\n=== EVALUACIÓN DE MODELOS PREDICTIVOS ===")

# PASO 9: REALIZAR PREDICCIONES
y_pred_lr = pipeline_lr.predict(X_test)
y_pred_rf = pipeline_rf.predict(X_test)
y_pred_xgb = pipeline_xgb.predict(X_test)

# PASO 10: CALCULAR MÉTRICAS
def calculate_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'Modelo': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

# Calcular métricas para cada modelo
metrics_lr = calculate_metrics(y_test, y_pred_lr, 'Regresión Lineal')
metrics_rf = calculate_metrics(y_test, y_pred_rf, 'Random Forest')
metrics_xgb = calculate_metrics(y_test, y_pred_xgb, 'XGBoost')

# Crear DataFrame con las métricas
metrics_df = pd.DataFrame([metrics_lr, metrics_rf, metrics_xgb])
print("\nComparación de métricas entre modelos:")
print(metrics_df)

# PASO 11: VISUALIZACIONES
# Configuración global de estilo para todos los gráficos
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Predicciones vs Valores Reales para cada modelo
plt.figure(figsize=(20, 6))

# Función para mejorar la visualización
def plot_prediction_vs_real(ax, y_true, y_pred, title):
    # Calcular límites para los ejes
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    
    # Crear scatter plot con densidad de puntos
    scatter = ax.hexbin(y_true, y_pred, gridsize=30, cmap='viridis', bins='log')
    
    # Línea de predicción perfecta
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    
    # Añadir barra de color
    cbar = plt.colorbar(scatter, ax=ax, label='Densidad de puntos')
    cbar.ax.tick_params(labelsize=10)
    
    # Configurar gráfico
    ax.set_xlabel('% Mermas Reales', fontsize=12, labelpad=10)
    ax.set_ylabel('% Mermas Predichas', fontsize=12, labelpad=10)
    ax.set_title(title, pad=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Añadir R² al gráfico
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
            transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=12,
            fontweight='bold')

# Crear subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Graficar cada modelo
plot_prediction_vs_real(ax1, y_test, y_pred_lr, 'Regresión Lineal')
plot_prediction_vs_real(ax2, y_test, y_pred_rf, 'Random Forest')
plot_prediction_vs_real(ax3, y_test, y_pred_xgb, 'XGBoost')

plt.tight_layout()
plt.savefig('predicciones_vs_reales.png', dpi=300, bbox_inches='tight')
print("\nGráfico guardado: predicciones_vs_reales.png")

# PASO 12: ANÁLISIS DE RESIDUOS
plt.figure(figsize=(20, 6))

# Función para mejorar la visualización de residuos
def plot_residuals(ax, y_pred, residuals, title):
    scatter = ax.hexbin(y_pred, residuals, gridsize=30, cmap='viridis', bins='log')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Sin Error')
    cbar = plt.colorbar(scatter, ax=ax, label='Densidad de puntos')
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_xlabel('Predicciones (%)', fontsize=12, labelpad=10)
    ax.set_ylabel('Residuos (%)', fontsize=12, labelpad=10)
    ax.set_title(title, pad=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Añadir estadísticas de residuos
    stats_text = f'Media: {np.mean(residuals):.3f}\nDesv. Est.: {np.std(residuals):.3f}'
    ax.text(0.05, 0.95, stats_text, 
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=11)

# Crear subplots para residuos
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Calcular y graficar residuos
residuals_lr = y_test - y_pred_lr
residuals_rf = y_test - y_pred_rf
residuals_xgb = y_test - y_pred_xgb

plot_residuals(ax1, y_pred_lr, residuals_lr, 'Residuos - Regresión Lineal')
plot_residuals(ax2, y_pred_rf, residuals_rf, 'Residuos - Random Forest')
plot_residuals(ax3, y_pred_xgb, residuals_xgb, 'Residuos - XGBoost')

plt.tight_layout()
plt.savefig('analisis_residuos.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado: analisis_residuos.png")

# PASO 13: DISTRIBUCIÓN DE ERRORES
plt.figure(figsize=(20, 6))

# Función para mejorar la visualización de distribución de errores
def plot_error_distribution(ax, residuals, title):
    sns.histplot(residuals, kde=True, ax=ax, color='skyblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Sin Error')
    ax.set_xlabel('Error (%)', fontsize=12, labelpad=10)
    ax.set_ylabel('Frecuencia', fontsize=12, labelpad=10)
    ax.set_title(title, pad=15, fontweight='bold')
    
    # Añadir estadísticas
    stats_text = f'Media: {np.mean(residuals):.3f}\nDesv. Est.: {np.std(residuals):.3f}'
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
            horizontalalignment='right',
            fontsize=11)

# Crear subplots para distribución de errores
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

plot_error_distribution(ax1, residuals_lr, 'Distribución de Errores - Regresión Lineal')
plot_error_distribution(ax2, residuals_rf, 'Distribución de Errores - Random Forest')
plot_error_distribution(ax3, residuals_xgb, 'Distribución de Errores - XGBoost')

plt.tight_layout()
plt.savefig('distribucion_errores.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado: distribucion_errores.png")

# PASO 14: IMPORTANCIA DE CARACTERÍSTICAS
if hasattr(pipeline_rf['regressor'], 'feature_importances_'):
    print("\n--- IMPORTANCIA DE CARACTERÍSTICAS (RANDOM FOREST) ---")
    # Obtener nombres de características después de one-hot encoding
    preprocessor = pipeline_rf.named_steps['preprocessor']
    cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numeric_features, cat_cols])
    
    # Obtener importancias
    importances = pipeline_rf['regressor'].feature_importances_
    
    # Crear DataFrame para visualización
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Mostrar las 10 características más importantes
    print("\nTop 10 características más importantes:")
    print(feature_importance.head(10))
    
    # Visualizar
    plt.figure(figsize=(12, 8))
    
    # Crear gráfico de barras mejorado
    ax = sns.barplot(
        x='importance',
        y='feature',
        data=feature_importance.head(10),
        palette='viridis'
    )
    
    plt.title('Top 10 Características Más Importantes', pad=20, fontsize=16, fontweight='bold')
    plt.xlabel('Importancia', fontsize=12, labelpad=10)
    plt.ylabel('Característica', fontsize=12, labelpad=10)
    
    # Añadir valores en las barras
    for i, v in enumerate(feature_importance.head(10)['importance']):
        ax.text(v, i, f'{v:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('importancia_caracteristicas.png', dpi=300, bbox_inches='tight')
    print("Gráfico guardado: importancia_caracteristicas.png")

# PASO 15: CONCLUSIÓN Y GENERACIÓN DE REPORTES
def generate_model_markdown(model_name, metrics, residuals, y_true, y_pred, data):
    """Genera documentación en formato Markdown para cada modelo."""
    
    # Crear DataFrame con predicciones y errores
    predictions_df = pd.DataFrame({
        'Valor Real': y_true,
        'Predicción': y_pred,
        'Error': y_true - y_pred,
        'Error %': abs((y_true - y_pred) / y_true * 100),
        'Categoría': data.loc[y_true.index, 'categoria'],
        'Tienda': data.loc[y_true.index, 'tienda']
    }).reset_index()
    
    # Ordenar por error absoluto descendente
    predictions_df['Error_Abs'] = abs(predictions_df['Error'])
    predictions_df = predictions_df.sort_values('Error_Abs', ascending=False)
    
    # Generar tabla de top 10 predicciones
    table_rows = []
    for _, row in predictions_df.head(10).iterrows():
        table_rows.append(
            f"| {row['index']} | {row['Valor Real']:.2f} | {row['Predicción']:.2f} | "
            f"{row['Error']:.2f} | {row['Error %']:.1f}% | {row['Categoría']} | {row['Tienda']} |"
        )
    
    predictions_table = "\n".join(table_rows)
    
    md_content = f"""# Resultados de Predicción: {model_name}

## Resumen de Métricas

- **R²**: {metrics['R²']:.4f} (Proporción de varianza explicada por el modelo)
- **RMSE**: {metrics['RMSE']:.2f} (Error cuadrático medio, en unidades de la variable objetivo)
- **MAE**: {metrics['MAE']:.2f} (Error absoluto medio, en unidades de la variable objetivo)

## Interpretación

El modelo de {model_name} explica aproximadamente el {metrics['R²']*100:.1f}% de la variabilidad en las mermas. 
En promedio, las predicciones difieren de los valores reales en ±{metrics['RMSE']:.2f} unidades.

## Muestra de Predicciones (Top 10)

| # | Valor Real | Predicción | Error | Error % | Categoría | Tienda |
|---|------------|------------|--------|---------|-----------|---------|
{predictions_table}

## Distribución del Error

- **Error Mínimo**: {min(residuals):.2f}
- **Error Máximo**: {max(residuals):.2f}
- **Error Promedio**: {np.mean(residuals):.2f}
- **Desviación Estándar del Error**: {np.std(residuals):.2f}

*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*

## Visualizaciones
Las siguientes visualizaciones están disponibles en formato PNG:
1. Predicciones vs Valores Reales: `predicciones_vs_reales.png`
2. Análisis de Residuos: `analisis_residuos.png`
3. Distribución de Errores: `distribucion_errores.png`
"""
    return md_content

def generate_summary_report(metrics_df, best_model, best_r2, best_rmse):
    """Genera un reporte general en Markdown con la comparación de todos los modelos."""
    summary_md = f"""# Reporte General de Análisis Predictivo de Mermas

## Resumen Ejecutivo

El análisis predictivo de mermas ha evaluado tres modelos diferentes: Regresión Lineal, Random Forest y XGBoost.
El mejor modelo identificado es **{best_model}** con un R² de {best_r2:.4f}.

## Comparación de Modelos

### Tabla de Métricas
| Modelo | R² | RMSE | MAE |
|--------|-----|------|-----|
"""
    
    # Añadir métricas de cada modelo
    for _, row in metrics_df.iterrows():
        summary_md += f"| {row['Modelo']} | {row['R²']:.4f} | {row['RMSE']:.2f} | {row['MAE']:.2f} |\n"

    summary_md += f"""
### Interpretación General

1. **Mejor Modelo: {best_model}**
   - R² = {best_r2:.4f} (Explica el {best_r2*100:.1f}% de la variabilidad)
   - RMSE = {best_rmse:.2f} (Error promedio en unidades de merma)

2. **Comparación de Modelos**
   - La diferencia en rendimiento entre los modelos es de {(metrics_df['R²'].max() - metrics_df['R²'].min())*100:.1f} puntos porcentuales en R²
   - Todos los modelos muestran un RMSE entre {metrics_df['RMSE'].min():.2f} y {metrics_df['RMSE'].max():.2f} unidades

## Archivos de Análisis Individual
- [Regresión Lineal](modelo_regresión_lineal.md)
- [Random Forest](modelo_random_forest.md)
- [XGBoost](modelo_xgboost.md)

## Visualizaciones Generadas
1. **predicciones_vs_reales.png**: Comparación de predicciones vs valores reales para cada modelo
2. **analisis_residuos.png**: Análisis de residuos para cada modelo
3. **distribucion_errores.png**: Distribución de errores para cada modelo
4. **importancia_caracteristicas.png**: Importancia de las características (para Random Forest)
"""
    return summary_md

# Calcular métricas del mejor modelo
best_model = metrics_df.loc[metrics_df['R²'].idxmax(), 'Modelo']
best_r2 = metrics_df['R²'].max()
best_rmse = metrics_df.loc[metrics_df['R²'].idxmax(), 'RMSE']

# Generar y guardar documentación individual de cada modelo
print("\nGenerando documentación detallada...")
for model_name, metrics, residuals, y_pred in [
    ('Regresión Lineal', metrics_lr, residuals_lr, y_pred_lr),
    ('Random Forest', metrics_rf, residuals_rf, y_pred_rf),
    ('XGBoost', metrics_xgb, residuals_xgb, y_pred_xgb)
]:
    md_content = generate_model_markdown(model_name, metrics, residuals, y_test, y_pred, data)
    with open(f'modelo_{model_name.lower().replace(" ", "_")}.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Archivo generado: modelo_{model_name.lower().replace(' ', '_')}.md")

# Generar y guardar reporte general
summary_content = generate_summary_report(metrics_df, best_model, best_r2, best_rmse)
with open('reporte_general.md', 'w', encoding='utf-8') as f:
    f.write(summary_content)
print("Archivo generado: reporte_general.md")

print("\nEl análisis predictivo del porcentaje de mermas ha sido completado exitosamente.")
print("Consulte reporte_general.md para ver el análisis completo.") 