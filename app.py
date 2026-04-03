import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from datetime import datetime
import csv
import json
import matplotlib.pyplot as plt
import time
import os

# MODELO
# ==============================
model = tf.keras.models.load_model("modelo.h5")

clases = [
    "Avión", "Automóvil", "Pájaro", "Gato", "Ciervo",
    "Perro", "Rana", "Caballo", "Barco", "Camión"
]

ruta_imagen = None
datos_historial = []
contador = 0

# FUNCIONES
# ==============================

def interpretar_confianza(valor):
    if valor > 80:
        return "Alta confianza"
    elif valor > 50:
        return "Confianza media"
    else:
        return "Baja confianza"


def color_confianza(valor):
    if valor > 80:
        return "green"
    elif valor > 50:
        return "orange"
    else:
        return "red"


def ver_rendimiento():
    try:
        with open("historial.json", "r") as f:
            data = json.load(f)

        plt.figure()
        plt.plot(data["accuracy"], label="Entrenamiento")
        plt.plot(data["val_accuracy"], label="Validación")
        plt.title("Accuracy del Modelo")
        plt.xlabel("Épocas")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(data["loss"], label="Entrenamiento")
        plt.plot(data["val_loss"], label="Validación")
        plt.title("Loss del Modelo")
        plt.xlabel("Épocas")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    except:
        messagebox.showerror("Error", "No se encontró historial.json")

def ver_matriz_confusion():
    try:
        img = Image.open("confusion_matrix.png")
        img = img.resize((500, 350))
        img_tk = ImageTk.PhotoImage(img)

        ventana_cm = tk.Toplevel()
        ventana_cm.title("Matriz de Confusión")

        lbl = tk.Label(ventana_cm, image=img_tk)
        lbl.image = img_tk
        lbl.pack()

    except:
        messagebox.showerror("Error", "No se encontró confusion_matrix.png")

def cargar_imagen():
    global ruta_imagen

    ruta_imagen = filedialog.askopenfilename(
        filetypes=[("Imágenes", "*.jpg *.png *.jpeg")]
    )

    if ruta_imagen:
        try:
            img = Image.open(ruta_imagen)
            img = img.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)

            panel_img.config(image=img_tk)
            panel_img.image = img_tk

            resultado_label.config(text="Imagen cargada ✔")
            mensaje_confianza.config(text="")

            limpiar_resultados()

        except:
            messagebox.showerror("Error", "No se pudo cargar la imagen")

def limpiar_resultados():
    top1.config(text="")
    top2.config(text="")
    top3.config(text="")
    barra1["value"] = 0
    barra2["value"] = 0
    barra3["value"] = 0
    resultado_grande.config(text="")
    img_procesada_label.config(image='')

def analizar_imagen():
    global ruta_imagen, datos_historial, contador

    if ruta_imagen is None:
        messagebox.showwarning("Aviso", "Primero cargue una imagen")
        return

    try:
        img = Image.open(ruta_imagen)

        img_modelo = img.resize((96, 96))
        img_array = np.array(img_modelo) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        img_small = img_modelo.resize((100, 100))
        img_tk2 = ImageTk.PhotoImage(img_small)
        img_procesada_label.config(image=img_tk2)
        img_procesada_label.image = img_tk2

        inicio = time.time()
        pred = model.predict(img_array, verbose=0)[0]
        fin = time.time()
        tiempo = (fin - inicio) * 1000

        top_indices = np.argsort(pred)[-3:][::-1]

        clase_principal = clases[top_indices[0]]
        confianza = pred[top_indices[0]] * 100
        nivel = interpretar_confianza(confianza)

        resultado_grande.config(
            text=f"ES UN {clase_principal.upper()}",
            fg=color_confianza(confianza),
            font=("Arial", 16, "bold")
        )

        resultado_label.config(
            text=f"Confianza: {confianza:.2f}% | Tiempo: {tiempo:.2f} ms"
        )

        mensaje_confianza.config(text=nivel, fg=color_confianza(confianza))

        if confianza < 50:
            messagebox.showwarning("Advertencia", "⚠️ El modelo no está seguro")

        top1.config(text=f"{clases[top_indices[0]]} ({pred[top_indices[0]]*100:.2f}%)")
        barra1["value"] = pred[top_indices[0]] * 100

        top2.config(text=f"{clases[top_indices[1]]} ({pred[top_indices[1]]*100:.2f}%)")
        barra2["value"] = pred[top_indices[1]] * 100

        top3.config(text=f"{clases[top_indices[2]]} ({pred[top_indices[2]]*100:.2f}%)")
        barra3["value"] = pred[top_indices[2]] * 100

        contador += 1
        contador_label.config(text=f"Imágenes analizadas: {contador}")

        hora = datetime.now().strftime("%H:%M:%S")
        nombre_img = os.path.basename(ruta_imagen)

        historial.insert("", "end", values=(
            hora,
            nombre_img,
            clase_principal,
            f"{confianza:.2f}%",
            f"{tiempo:.2f} ms",
            nivel
        ))

        datos_historial.append([
            hora,
            nombre_img,
            clase_principal,
            f"{confianza:.2f}%",
            f"{tiempo:.2f} ms",
            nivel
        ])

    except:
        messagebox.showerror("Error", "No se pudo analizar la imagen")

def exportar_csv():
    if not datos_historial:
        messagebox.showinfo("Info", "No hay datos para exportar")
        return

    ruta = filedialog.asksaveasfilename(defaultextension=".csv")

    if ruta:
        with open(ruta, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Hora", "Imagen", "Clase", "Confianza", "Tiempo", "Nivel"])
            writer.writerows(datos_historial)

        messagebox.showinfo("Éxito", "Historial exportado correctamente")

# UI
# ==============================

ventana = tk.Tk()
ventana.title("Sistema Inteligente de Clasificación de Imágenes")
ventana.geometry("1000x700")
ventana.minsize(1000, 700)
ventana.configure(bg="#F0F2F5")

left = tk.Frame(ventana, bg="white", width=400)
left.pack(side="left", fill="both", padx=(15, 5), pady=10)

frame_img = tk.Frame(left, width=300, height=300, bg="#E0E0E0")
frame_img.pack(pady=20, padx=10)
frame_img.pack_propagate(False)

panel_img = tk.Label(frame_img, bg="#E0E0E0")
panel_img.pack(fill="both", expand=True)

tk.Label(left, text="Imagen procesada (96x96)", bg="white").pack()
img_procesada_label = tk.Label(left, bg="white")
img_procesada_label.pack(pady=5)

tk.Button(left, text="Cargar Imagen", bg="#1976D2", fg="white", width=20, command=cargar_imagen).pack(pady=5)
tk.Button(left, text="Analizar", bg="#388E3C", fg="white", width=20, command=analizar_imagen).pack(pady=5)

contenedor = tk.Frame(ventana)
contenedor.pack(side="right", fill="both", expand=True, padx=(5, 15), pady=10)

canvas = tk.Canvas(contenedor, bg="#F0F2F5")
scrollbar = tk.Scrollbar(contenedor, orient="vertical", command=canvas.yview)

scrollable_frame = tk.Frame(canvas, bg="#F0F2F5")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# CONTENIDO
tk.Label(scrollable_frame, text="Resultados del Modelo", font=("Arial", 16, "bold"), bg="#F0F2F5").pack(pady=10)

resultado_grande = tk.Label(scrollable_frame, bg="#F0F2F5")
resultado_grande.pack()

resultado_label = tk.Label(scrollable_frame, font=("Arial", 12), bg="#F0F2F5")
resultado_label.pack()

mensaje_confianza = tk.Label(scrollable_frame, font=("Arial", 11, "bold"), bg="#F0F2F5")
mensaje_confianza.pack()

contador_label = tk.Label(scrollable_frame, text="Imágenes analizadas: 0", bg="#F0F2F5")
contador_label.pack(pady=5)

top1 = tk.Label(scrollable_frame, font=("Arial", 12, "bold"), bg="#F0F2F5")
top1.pack(pady=5)
barra1 = ttk.Progressbar(scrollable_frame, length=300)
barra1.pack()

top2 = tk.Label(scrollable_frame, font=("Arial", 12), bg="#F0F2F5")
top2.pack(pady=5)
barra2 = ttk.Progressbar(scrollable_frame, length=300)
barra2.pack()

top3 = tk.Label(scrollable_frame, font=("Arial", 12), bg="#F0F2F5")
top3.pack(pady=5)
barra3 = ttk.Progressbar(scrollable_frame, length=300)
barra3.pack()

tk.Label(scrollable_frame, text="Historial", font=("Arial", 12, "bold"), bg="#F0F2F5").pack(pady=10)

columnas = ("Hora", "Imagen", "Clase", "Confianza", "Tiempo", "Nivel")

frame_historial = tk.Frame(scrollable_frame)
frame_historial.pack(fill="both", expand=True)

historial = ttk.Treeview(frame_historial, columns=columnas, show="headings", height=6)

for col in columnas:
    historial.heading(col, text=col)
    historial.column(col, anchor="center", width=100, stretch=True)

scroll = ttk.Scrollbar(frame_historial, orient="vertical", command=historial.yview)
historial.configure(yscroll=scroll.set)

historial.pack(side="left", fill="both", expand=True)
scroll.pack(side="right", fill="y")

tk.Button(scrollable_frame, text="Exportar a CSV", bg="#6A1B9A", fg="white", command=exportar_csv).pack(pady=5)
tk.Button(scrollable_frame, text="Ver rendimiento del modelo", bg="#FF9800", fg="white", command=ver_rendimiento).pack(pady=5)
tk.Button(scrollable_frame, text="Ver matriz de confusión", bg="#009688", fg="white", command=ver_matriz_confusion).pack(pady=10)

ventana.mainloop()