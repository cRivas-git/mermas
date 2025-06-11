# Análisis de Reglas de Asociación para Mermas

## Interpretación de Métricas

### Soporte (Support)
- Frecuencia con la que aparece un patrón en el dataset
- Un soporte de 0.05 significa que el patrón aparece en el 5% de los casos
- Indica qué tan común es el patrón en los datos

### Confianza (Confidence)
- Probabilidad de que ocurra el consecuente cuando ocurre el antecedente
- Una confianza de 0.8 significa que el 80% de las veces que ocurre A, también ocurre B

### Lift
- Mide la fuerza de la relación entre los elementos de la regla
- Lift > 1: Correlación positiva (los elementos tienden a aparecer juntos)
- Lift = 1: Independencia (no hay relación especial)
- Lift < 1: Correlación negativa (los elementos tienden a no aparecer juntos)

## Reglas Encontradas

### Patrón 1
- **Regla**: ubicacion_Bodega Y tienda_TEMUCO II => categoria_CECINAS GRANEL
- **Soporte**: 8.87%
- **Confianza**: 55.65%
- **Lift**: 1.96

### Patrón 2
- **Regla**: tienda_TEMUCO II => categoria_CECINAS GRANEL
- **Soporte**: 10.11%
- **Confianza**: 47.15%
- **Lift**: 1.66

### Patrón 3
- **Regla**: tienda_TEMUCO II => ubicacion_Bodega Y categoria_CECINAS GRANEL
- **Soporte**: 8.87%
- **Confianza**: 41.33%
- **Lift**: 1.66

### Patrón 4
- **Regla**: ubicacion_Bodega Y tienda_ANGOL => categoria_CECINAS GRANEL
- **Soporte**: 6.25%
- **Confianza**: 45.61%
- **Lift**: 1.61

### Patrón 5
- **Regla**: ubicacion_Bodega Y tienda_TEMUCO IV => categoria_CECINAS GRANEL
- **Soporte**: 5.28%
- **Confianza**: 43.72%
- **Lift**: 1.54

### Patrón 6
- **Regla**: tienda_ANGOL Y categoria_CECINAS GRANEL => ubicacion_Bodega
- **Soporte**: 6.25%
- **Confianza**: 87.82%
- **Lift**: 1.42

### Patrón 7
- **Regla**: ubicacion_Bodega => categoria_CECINAS GRANEL
- **Soporte**: 24.91%
- **Confianza**: 40.30%
- **Lift**: 1.42

### Patrón 8
- **Regla**: categoria_CECINAS GRANEL => ubicacion_Bodega
- **Soporte**: 24.91%
- **Confianza**: 87.78%
- **Lift**: 1.42

### Patrón 9
- **Regla**: tienda_TEMUCO II Y categoria_CECINAS GRANEL => ubicacion_Bodega
- **Soporte**: 8.87%
- **Confianza**: 87.67%
- **Lift**: 1.42

### Patrón 10
- **Regla**: tienda_TEMUCO IV Y categoria_CECINAS GRANEL => ubicacion_Bodega
- **Soporte**: 5.28%
- **Confianza**: 87.12%
- **Lift**: 1.41

### Patrón 11
- **Regla**: tienda_TEMUCO V => ubicacion_Sala de Ventas
- **Soporte**: 7.07%
- **Confianza**: 48.51%
- **Lift**: 1.27

### Patrón 12
- **Regla**: tienda_TEMUCO II => ubicacion_Bodega
- **Soporte**: 15.93%
- **Confianza**: 74.28%
- **Lift**: 1.20

### Patrón 13
- **Regla**: tienda_TEMUCO III => ubicacion_Sala de Ventas
- **Soporte**: 9.81%
- **Confianza**: 43.80%
- **Lift**: 1.15

### Patrón 14
- **Regla**: tienda_TEMUCO IV => ubicacion_Sala de Ventas
- **Soporte**: 8.40%
- **Confianza**: 41.03%
- **Lift**: 1.07

### Patrón 15
- **Regla**: tienda_ANGOL => ubicacion_Bodega
- **Soporte**: 13.71%
- **Confianza**: 64.97%
- **Lift**: 1.05

### Patrón 16
- **Regla**: categoria_YOGHURT => ubicacion_Bodega
- **Soporte**: 5.13%
- **Confianza**: 64.71%
- **Lift**: 1.05

### Patrón 17
- **Regla**: tienda_TEMUCO IV => ubicacion_Bodega
- **Soporte**: 12.08%
- **Confianza**: 58.97%
- **Lift**: 0.95

### Patrón 18
- **Regla**: tienda_TEMUCO III => ubicacion_Bodega
- **Soporte**: 12.59%
- **Confianza**: 56.20%
- **Lift**: 0.91

### Patrón 19
- **Regla**: categoria_FIDEOS Y PASTAS => ubicacion_Bodega
- **Soporte**: 5.20%
- **Confianza**: 54.48%
- **Lift**: 0.88

### Patrón 20
- **Regla**: tienda_TEMUCO V => ubicacion_Bodega
- **Soporte**: 7.50%
- **Confianza**: 51.49%
- **Lift**: 0.83

## Análisis de Distribución de Métricas

### Distribución de Lift
- Mínimo: 0.83
- Máximo: 1.96
- Promedio: 1.29

### Distribución de Soporte
- Mínimo: 5.13%
- Máximo: 24.91%
- Promedio: 10.35%

### Distribución de Confianza
- Mínimo: 40.30%
- Máximo: 87.82%
- Promedio: 59.13%
