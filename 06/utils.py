import os
import cv2 as cv
import numpy as np
from pathlib import Path


# PREPROCESADO: Buscar y segmentar la ROI
def mejor_roi_por_negros(image, ventana=500, paso=20):
    H, W = image.shape
    w = min(ventana, W, H)
    h = min(ventana, W, H)
    mejor_ventana = (0, 0, w, h)
    mejor_suma = -1

    for y in range(0, H - h + 1, paso):
        for x in range(0, W - w + 1, paso):
            region = image[y:y+h, x:x+w]
            suma = np.sum(region == 0)

            if suma > mejor_suma:
                mejor_suma = suma
                mejor_ventana = (x, y, w, h)

    return mejor_ventana

def recortar_roi(db_path, out_path, ventana=500, paso=15):
    users = os.listdir(db_path)

    for user in users:
        user_folder = os.path.join(db_path, user)

        if not os.path.isdir(user_folder):
            continue

        image_files = [f for f in os.listdir(user_folder) if f.lower().endswith(".png")]
        if len(image_files) < 2:
            continue
        
        out_dir = os.path.join(out_path, user, "roi")
        os.makedirs(out_dir, exist_ok=True)

        for filename in image_files:
            img_path = os.path.join(user_folder, filename)
            img = cv.imread(img_path)

            if img is None:
                continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) # Otsu: sugerencia de GPT
            x, y, w, h = mejor_roi_por_negros(image, ventana=ventana, paso=paso)
            roi = img[y:y+h, x:x+w]
            cv.imwrite(os.path.join(out_dir, filename), roi)


# PREPROCESADO: Ecualizar y normalizar el histograma
def ecualizar_histograma(db_path, out_path):
    users = os.listdir(db_path)

    for user in users:
        user_folder = os.path.join(out_path, user, "roi")

        if not os.path.isdir(user_folder):
            continue

        image_files = [f for f in os.listdir(user_folder) if f.lower().endswith('.png')]
        if len(image_files) < 2:
            continue

        out_dir = os.path.join(out_path, user, "equalized")
        os.makedirs(out_dir, exist_ok=True)

        for filename in image_files:
            img_path = os.path.join(user_folder, filename)
            image = cv.imread(img_path)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            equalized = cv.equalizeHist(gray)
            cv.imwrite(os.path.join(out_dir, filename), equalized)


# PREPROCESADO: Aplicar un filtro bilateral
def aplicar_filtro_bilateral(db_path, out_path):
    users = os.listdir(db_path)
    for user in users:
        user_folder = os.path.join(out_path, user, "equalized")
        if not os.path.isdir(user_folder):
            print(f"No existe directorio: {user_folder}")
            continue

        image_files = [f for f in os.listdir(user_folder) if f.lower().endswith('.png')]
        if len(image_files) < 2:
            continue

        out_dir = os.path.join(out_path, user, "bilateral_filter")
        os.makedirs(out_dir, exist_ok=True)

        for filename in image_files:
            img_path = os.path.join(user_folder, filename)
            image = cv.imread(img_path)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            bilateral = cv.bilateralFilter(gray, d=10, sigmaColor=10, sigmaSpace=10)
            cv.imwrite(os.path.join(out_dir, filename), bilateral)


# PREPROCESADO: Aplicar un realce de crestas
def realzar_crestas(db_path, out_path):
    users = os.listdir(db_path)
    for user in users:
        user_folder = os.path.join(out_path, user, "bilateral_filter")
        if not os.path.isdir(user_folder):
            continue

        image_files = [f for f in os.listdir(user_folder) if f.lower().endswith('.png')]
        if len(image_files) < 2:
            continue

        out_dir = os.path.join(out_path, user, "sobel")
        os.makedirs(out_dir, exist_ok=True)

        for filename in image_files:
            img_path = os.path.join(user_folder, filename)
            image = cv.imread(img_path)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            sobel = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
            _, bw = cv.threshold(sobel, 0, 255, cv.THRESH_BINARY_INV)
            sobel_8u = cv.normalize(bw, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8) # Código de GPT para solventar warning: [ WARN:0@0.762] global loadsave.cpp:1063 cv::imwrite_ Unsupported depth image for selected encoder is fallbacked to CV_8U.
            cv.imwrite(os.path.join(out_dir, filename), sobel_8u)

# PREPROCESADO: Refinar la ROI