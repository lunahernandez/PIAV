# preprocesado4.py
# 4. Aplicar un realce de crestas con un operador de bordes Sobel.

import os
import cv2 as cv
import matplotlib.pyplot as plt

def main():
    sample_path = r"06/data"
    folders = os.listdir(sample_path)

    print("Carpetas encontradas:", folders)

    # Acceder a la carpeta filtradas dentro de cada carpeta
    for folder in folders:
        folder_path = os.path.join(sample_path, folder)
        if not os.path.isdir(folder_path):
            continue

        filtradas_path = os.path.join(folder_path, "filtradas_bilateral")
        if not os.path.exists(filtradas_path):
            print(f"La carpeta '{folder}' no tiene una subcarpeta 'filtradas_bilateral'.")
            continue

        images_files = os.listdir(filtradas_path)
        image_files = [f for f in images_files if f.lower().endswith('.png')]

        # Leer las dos imágenes
        img1_filtrada_path = os.path.join(filtradas_path, image_files[0])
        img2_filtrada_path = os.path.join(filtradas_path, image_files[1])

        img1 = cv.imread(img1_filtrada_path)
        img2 = cv.imread(img2_filtrada_path)

        # Convertir a escala de grises
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # Aplicar el operador Sobel para realce de crestas
        sobel1 = cv.Sobel(gray1, cv.CV_64F, 0, 1, ksize=5)
        sobel2 = cv.Sobel(gray2, cv.CV_64F, 0, 1, ksize=5)

        # Mostrar resultados con originales y realce de crestas
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 2, 1)
        plt.title('Imagen 1 Original')
        plt.imshow(gray1, cmap='gray')
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.title('Imagen 2 Original')
        plt.imshow(gray2, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.title('Imagen 1 Realce Sobel')
        plt.imshow(sobel1, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.title('Imagen 2 Realce Sobel')
        plt.imshow(sobel2, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
