import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import json  

print("Cargando dataset CIFAR-10...")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# PREPROCESAMIENTO
# ==============================
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = tf.image.resize(x_train, (96,96))
x_test = tf.image.resize(x_test, (96,96))

print("Datos listos.")

# MODELO (Transfer Learning)
# ==============================

print("Cargando MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96,96,3),
    include_top=False,
    weights='imagenet'
)

# FASE 1
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nEntrenando fase 1...")
history1 = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    batch_size=64
)

# FASE 2: Fine-Tuning
# ==============================
print("\nIniciando fine-tuning...")

base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history2 = model.fit(
    x_train, y_train,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
    batch_size=64
)

# EVALUACIÓN FINAL
# ==============================
print("\nEvaluando modelo...")

test_loss, test_acc = model.evaluate(x_test, y_test)

print("\n===================================")
print("RESULTADOS DEL MODELO V4 FINAL")
print(f"Accuracy final: {test_acc*100:.2f}%")
print(f"Loss final: {test_loss:.4f}")
print("===================================\n")

# MATRIZ DE CONFUSIÓN + F1
# ==============================
print("Generando métricas avanzadas...")

y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

y_test_flat = y_test.flatten()

cm = confusion_matrix(y_test_flat, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.savefig("confusion_matrix.png")
plt.close()

print("\nReporte de Clasificación (incluye F1-score):")
print(classification_report(y_test_flat, y_pred_classes))

# TIEMPO DE INFERENCIA
# ==============================
print("\nCalculando tiempo de inferencia...")

start_time = time.time()
model.predict(x_test[:100], verbose=0)
end_time = time.time()

inference_time = (end_time - start_time) / 100
print(f"Tiempo promedio de inferencia: {inference_time*1000:.4f} ms por imagen")

# GUARDAR MODELO
# ==============================
model.save("modelo.h5")
print("Modelo guardado como modelo.h5")

# CONVERSIÓN A TFLITE
# ==============================
print("\nConvirtiendo a TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("modelo.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo convertido a TFLite correctamente.")

# PRUEBA DE PREDICCIÓN
# ==============================
print("\nProbando predicción...")

img = x_test[0]
img = np.expand_dims(img, axis=0)

pred = model.predict(img, verbose=0)
clase = np.argmax(pred)

print("Clase predicha:", clase)

# GRÁFICAS
# ==============================
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']

loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

# Accuracy
plt.figure()
plt.plot(acc)
plt.plot(val_acc)
plt.title("Precisión del modelo")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend(["Entrenamiento", "Validación"])
plt.savefig("accuracy.png")
plt.close()

# Loss
plt.figure()
plt.plot(loss)
plt.plot(val_loss)
plt.title("Pérdida del modelo")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend(["Entrenamiento", "Validación"])
plt.savefig("loss.png")
plt.close()

print("Gráficas guardadas.")

# GUARDAR HISTORIAL PARA APP
# ==============================
historial = {
    "accuracy": acc,
    "val_accuracy": val_acc,
    "loss": loss,
    "val_loss": val_loss
}

with open("historial.json", "w") as f:
    json.dump(historial, f)

print("Historial guardado en historial.json")

print("\n✅ PROCESO FINAL COMPLETADO")