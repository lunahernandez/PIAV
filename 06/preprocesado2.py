# preprocesado2.py
# 2. Ecualizar y normalizar el histograma con el fin de mejorar el contraste.

import os
import cv2 as cv
import matplotlib.pyplot as plt

def main():
    sample_path = r"06/data"
    folders = os.listdir(sample_path)
    
    print("Carpetas encontradas:", folders)

    for folder in folders:
        folder_path = os.path.join(sample_path, folder)
        if not os.path.isdir(folder_path):
            continue

        roi_path = os.path.join(folder_path, "roi")
        if not os.path.exists(roi_path):
            print(f"La carpeta '{folder}' no tiene una subcarpeta 'roi'.")
            continue

        images_files = os.listdir(roi_path)
        image_files = [f for f in images_files if f.lower().endswith('.png')]

        if len(image_files) < 2:
            print(f"No hay suficientes imágenes en {roi_path}")
            continue

        # Corregido: leer desde roi_path
        img1_path = os.path.join(roi_path, image_files[0])
        img2_path = os.path.join(roi_path, image_files[1])

        img1 = cv.imread(img1_path)
        img2 = cv.imread(img2_path)

        # Convertir a escala de grises
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # Ecualizar el histograma
        eq1 = cv.equalizeHist(gray1)
        eq2 = cv.equalizeHist(gray2)

        # Guardar resultados
        out_folder = os.path.join(folder_path, "ecualizadas")
        os.makedirs(out_folder, exist_ok=True)

        cv.imwrite(os.path.join(out_folder, image_files[0]), eq1)
        cv.imwrite(os.path.join(out_folder, image_files[1]), eq2)
        print(f"Imágenes ecualizadas guardadas en '{out_folder}'.")

if __name__ == "__main__":
    main()
