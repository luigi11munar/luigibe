import os

carpeta = r"C:\Users\sergi\Desktop\BackendTafur\li"
archivos = sorted(os.listdir(carpeta)) 
salida = "unificado.txt"

with open(salida, "w", encoding="utf-8") as archivo_salida:
    for nombre in archivos:
        if nombre.endswith(".txt"):
            ruta = os.path.join(carpeta, nombre)
            with open(ruta, "r", encoding="utf-8") as archivo_actual:
                contenido = archivo_actual.read()
                archivo_salida.write(contenido + "\n") 

print("Archivos unificados en", salida)