
Este proyecto implementa un sistema de **clasificación de imágenes** utilizando **Deep Learning**.
Permite analizar una imagen y predecir su categoría con un nivel de confianza, mostrando además métricas del modelo.

## Tecnologías

* Python
* TensorFlow / Keras
* MobileNetV2 (Transfer Learning)
* Tkinter
* NumPy, Matplotlib, Seaborn

### Entrenar el modelo--
# Ejecutar: 
    python codigo.py

Esto generará:
* modelo.h5
* modelo.tflite
* accuracy.png
* loss.png
* confusion_matrix.png
* historial.json

### Ejecutar la aplicación
# Ejecutar:
    python app.py


# Funcionalidades
* Clasificación de imágenes en tiempo real
* Top 3 predicciones
* Nivel de confianza
* Tiempo de inferencia
* Historial de resultados
* Exportación a CSV
* Visualización de métricas
* Matriz de confusión

