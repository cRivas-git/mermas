import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def remove_outliers(df, columns, n_std=3):
    """Elimina outliers usando el método IQR"""
    df_clean = df.copy()
    
    for column in columns:
        # Calcular Q1, Q3 e IQR
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir límites para outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filtrar outliers
        df_clean = df_clean[
            (df_clean[column] >= lower_bound) & 
            (df_clean[column] <= upper_bound)
        ]
        
        print(f"Removidos {len(df) - len(df_clean)} outliers de {column}")
    
    return df_clean

def remove_constant_columns(df):
    """Elimina columnas que tienen un solo valor único"""
    return df.loc[:, df.nunique() > 1]

def listar_variables(data):
    """Lista las variables disponibles en el dataset organizadas por tipo"""
    
    # Variables del dataset
    print("\nVariables disponibles en el dataset:")
    print("\n1. Variables de Identificación:")
    print("   - codigo_producto")
    print("   - descripcion")
    
    print("\n2. Variables Categóricas:")
    print("   - negocio")
    print("   - seccion")
    print("   - linea")
    print("   - categoria")
    print("   - abastecimiento")
    print("   - comuna")
    print("   - region")
    print("   - tienda")
    print("   - motivo")
    print("   - ubicación_motivo")
    
    print("\n3. Variables Temporales:")
    print("   - mes")
    print("   - año")
    print("   - semestre")
    print("   - fecha")
    
    print("\n4. Variables Numéricas:")
    print("   - merma_unidad")
    print("   - merma_monto")
    print("   - merma_unidad_p")
    print("   - merma_monto_p")

def generate_correlation_analysis(data):
    """Genera análisis de correlación entre variables categóricas y numéricas del dataset"""
    
    # Convertir fechas y crear nuevas características
    data['fecha'] = pd.to_datetime(data['fecha'], format='%d-%m-%Y')
    data['dia_semana'] = data['fecha'].dt.dayofweek
    data['dia_mes'] = data['fecha'].dt.day
    
    # Limpiar la variable merma_unidad
    data['merma_unidad_p'] = data['merma_unidad_p'].apply(lambda x: abs(float(str(x).replace(',', '.'))))
    
    # Seleccionar variables categóricas y temporales
    categorical_vars = ['negocio', 'seccion', 'linea', 'categoria', 'abastecimiento', 
                       'comuna', 'region', 'tienda', 'motivo', 'ubicación_motivo']
    
    temporal_vars = ['dia_semana', 'dia_mes']
    numeric_vars = ['merma_unidad_p']
    
    # Crear un diccionario para almacenar los label encoders
    label_encoders = {}
    encoded_data = pd.DataFrame()
    
    # Codificar variables categóricas
    for col in categorical_vars:
        if data[col].nunique() > 1:  # Solo codificar si hay más de un valor único
            label_encoders[col] = LabelEncoder()
            encoded_data[col] = label_encoders[col].fit_transform(data[col])
    
    # Añadir variables temporales y numéricas
    for col in temporal_vars + numeric_vars:
        encoded_data[col] = data[col]
    
    # Eliminar columnas con varianza cero o que generan NaN
    encoded_data = remove_constant_columns(encoded_data)
    
    # Calcular matriz de correlación
    correlation_matrix = encoded_data.corr()
    
    # Eliminar columnas que tienen NaN en la correlación
    columns_with_nan = correlation_matrix.columns[correlation_matrix.isna().any()].tolist()
    if columns_with_nan:
        print("\nEliminando las siguientes variables por generar NaN en la correlación:")
        for col in columns_with_nan:
            print(f"- {col}")
        encoded_data = encoded_data.drop(columns=columns_with_nan)
        correlation_matrix = encoded_data.corr()
    
    # Generar visualización
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(correlation_matrix), k=1)
    
    # Crear el mapa de calor
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True,
                cmap='RdYlBu',
                center=0,
                fmt='.2f',
                square=True,
                linewidths=0.5)
    
    # Ajustar la visualización
    plt.title('Matriz de Correlación - Variables Categóricas y Merma (Sin NaN)', pad=20, size=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Ajustar márgenes y guardar
    plt.tight_layout()
    plt.savefig('correlacion_variables_categoricas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generar reporte en Markdown
    md_content = """# Análisis de Correlación de Variables (Sin NaN)

## Variables Analizadas

### Variables Incluidas en el Análisis
"""
    # Listar las variables incluidas
    remaining_categorical = [col for col in categorical_vars if col in encoded_data.columns]
    remaining_temporal = [col for col in temporal_vars if col in encoded_data.columns]
    remaining_numeric = [col for col in numeric_vars if col in encoded_data.columns]
    
    if remaining_numeric:
        md_content += "\n#### Variables Numéricas\n"
        for var in remaining_numeric:
            md_content += f"- {var}\n"
    
    if remaining_categorical:
        md_content += "\n#### Variables Categóricas\n"
        for var in remaining_categorical:
            unique_values = data[var].unique()
            md_content += f"\n##### {var.title()}\n"
            md_content += "| Valores Posibles |\n|-------------------|\n"
            for val in sorted(unique_values):
                md_content += f"| {val} |\n"
    
    if remaining_temporal:
        md_content += "\n#### Variables Temporales\n"
        md_content += "- **Día de la semana** (0-6, donde 0 es lunes)\n"
        md_content += "- **Día del mes** (1-31)\n"
    
    md_content += """
## Matriz de Correlación

La siguiente matriz muestra las correlaciones entre todas las variables (sin NaN):

| Variables |"""
    
    # Crear encabezado de la tabla
    all_vars = encoded_data.columns.tolist()
    md_content += " | ".join(all_vars) + " |\n"
    md_content += "|" + "|".join(["-" * max(len(header), 7) for header in ["Variables"] + all_vars]) + "|\n"
    
    # Añadir valores de correlación
    for idx, var in enumerate(all_vars):
        row = [var]
        for val in correlation_matrix.loc[var]:
            if abs(val) > 0.7:
                row.append(f"**{val:.2f}**")
            else:
                row.append(f"{val:.2f}")
        md_content += "| " + " | ".join(row) + " |\n"
    
    md_content += """
## Interpretación de Correlaciones

### Niveles de Correlación:
- **Correlación Fuerte**: > 0.7 o < -0.7 (mostrado en negrita)
- **Correlación Moderada**: entre 0.4 y 0.7 o -0.4 y -0.7
- **Correlación Débil**: entre 0 y 0.4 o 0 y -0.4

### Correlaciones Significativas:
"""
    
    # Encontrar correlaciones significativas
    correlations = []
    for i in range(len(all_vars)):
        for j in range(i+1, len(all_vars)):
            corr = correlation_matrix.iloc[i,j]
            if abs(corr) > 0.3:  # Solo mostrar correlaciones significativas
                correlations.append((abs(corr), corr, all_vars[i], all_vars[j]))
    
    # Ordenar por magnitud de correlación
    correlations.sort(reverse=True)
    
    # Añadir las correlaciones más significativas al reporte
    for _, corr, var1, var2 in correlations:
        strength = "fuerte" if abs(corr) > 0.7 else "moderada" if abs(corr) > 0.4 else "débil"
        direction = "positiva" if corr > 0 else "negativa"
        md_content += f"- **{var1}** y **{var2}**: Correlación {direction} {strength} ({corr:.2f})\n"
    
    # Guardar reporte
    with open('reporte_correlacion.md', 'w', encoding='utf-8') as f:
        f.write(md_content)

def main():
    # Cargar el dataset
    data = pd.read_csv('mermas.csv', sep=';', encoding='latin1')
    
    # Mostrar variables disponibles
    listar_variables(data)
    
    # Generar análisis de correlación
    generate_correlation_analysis(data)
    
    print("\nAnálisis completado. Se han generado los siguientes archivos:")
    print("- correlacion_variables_categoricas.png: Visualización de la matriz de correlación")
    print("- reporte_correlacion.md: Reporte detallado del análisis")

if __name__ == "__main__":
    main()