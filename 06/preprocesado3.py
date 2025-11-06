# preprocesado3.py
# 3. Aplicar un filtro bilateral para eliminar el ruido pero no las texturas.

import os
import cv2 as cv
import matplotlib.pyplot as plt

def main():
    sample_path = r"06/data"
    folders = os.listdir(sample_path)

    print("Carpetas encontradas:", folders)

    # Acceder a la carpeta ecualizadas dentro de cada carpeta
    for folder in folders:
        folder_path = os.path.join(sample_path, folder)
        if not os.path.isdir(folder_path):
            continue

        ecualizadas_path = os.path.join(folder_path, "ecualizadas")
        if not os.path.exists(ecualizadas_path):
            print(f"La carpeta '{folder}' no tiene una subcarpeta 'ecualizadas'.")
            continue

        images_files = os.listdir(ecualizadas_path)
        image_files = [f for f in images_files if f.lower().endswith('.png')]

        # Leer las dos imágenes
        img1_ecualizada_path = os.path.join(ecualizadas_path, image_files[0])
        img2_ecualizada_path = os.path.join(ecualizadas_path, image_files[1])

        img1 = cv.imread(img1_ecualizada_path)
        img2 = cv.imread(img2_ecualizada_path)

        # Convertir a escala de grises
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # Aplicar filtro bilateral
        bilateral1 = cv.bilateralFilter(gray1, d=10, sigmaColor=7, sigmaSpace=11)
        bilateral2 = cv.bilateralFilter(gray2, d=10, sigmaColor=7, sigmaSpace=11)

        # Mostrar resultados con originales y filtradas
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
        plt.title('Imagen 1 Filtrada Bilateral')
        plt.imshow(bilateral1, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.title('Imagen 2 Filtrada Bilateral')
        plt.imshow(bilateral2, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Guardar resultados
        out_folder = os.path.join(folder_path, "filtradas_bilateral")
        os.makedirs(out_folder, exist_ok=True)

        cv.imwrite(os.path.join(out_folder, image_files[0]), bilateral1)
        cv.imwrite(os.path.join(out_folder, image_files[1]), bilateral2)
        print(f"Imágenes filtradas con bilateral guardadas en '{out_folder}'.")



if __name__ == "__main__":
    main()  