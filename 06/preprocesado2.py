# preprocesado2.py
# 2. Ecualizar y normalizar el histograma con el fin de mejorar el contraste.

import os
import cv2 as cv
import matplotlib.pyplot as plt

def main():
    sample_path = r"06/data"
    files = os.listdir(sample_path)

    print("Carpetas encontradas:", files)

    for folder in files:
        folder_path = os.path.join(sample_path, folder)
        if not os.path.isdir(folder_path):
            continue  # ignora si no es carpeta

        image_files = os.listdir(folder_path)
        image_files = [f for f in image_files if f.lower().endswith('.png')]

        if len(image_files) < 2:
            print(f"La carpeta '{folder}' no tiene 2 imágenes (tiene {len(image_files)}).")
            continue

        # Leer las dos imágenes
        img1_path = os.path.join(folder_path, image_files[0])
        img2_path = os.path.join(folder_path, image_files[1])

        img1 = cv.imread(img1_path)
        img2 = cv.imread(img2_path)

        # Convertir a escala de grises
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # Ecualizar el histograma
        eq1 = cv.equalizeHist(gray1)
        eq2 = cv.equalizeHist(gray2)

        # Mostrar resultados com originales y ecualizadas
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
        plt.title('Imagen 1 Ecualizada')
        plt.imshow(eq1, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.title('Imagen 2 Ecualizada')
        plt.imshow(eq2, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Guardar resultados
        out_folder = os.path.join(folder_path, "ecualizadas")
        os.makedirs(out_folder, exist_ok=True)

        cv.imwrite(os.path.join(out_folder, image_files[0]), eq1)
        cv.imwrite(os.path.join(out_folder, image_files[1]), eq2)
        print(f"Imágenes ecualizadas guardadas en '{out_folder}'.")

if __name__ == "__main__":
    main()
