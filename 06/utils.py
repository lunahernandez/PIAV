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
def refinar_crestas(db_path, out_path):
    """
    Refina las imágenes generadas por Sobel eliminando bordes falsos y ruido periférico.
    Invierte los colores (fondo blanco, crestas negras) para mejor detección SIFT.
    Guarda los resultados en out_path/<user>/refinadas
    """
    users = os.listdir(db_path)
    for user in users:
        user_folder = os.path.join(out_path, user, "sobel")
        if not os.path.isdir(user_folder):
            continue

        image_files = [f for f in os.listdir(user_folder) if f.lower().endswith('.png')]
        if len(image_files) == 0:
            continue

        out_dir = os.path.join(out_path, user, "refinadas")
        os.makedirs(out_dir, exist_ok=True)

        for filename in image_files:
            img_path = os.path.join(user_folder, filename)
            image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            
            if image is None or image.size == 0:
                print(f"(i) Imagen no legible: {filename}")
                continue

            # --- PASO 1: Eliminar bordes de la imagen (recorte interno) ---
            h, w = image.shape
            margin = int(min(h, w) * 0.05)  # 5% de margen
            cropped = image[margin:h-margin, margin:w-margin]

            # --- PASO 2: Operaciones morfológicas para limpiar ruido ---
            # Apertura: elimina puntos blancos aislados (ruido)
            kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            opened = cv.morphologyEx(cropped, cv.MORPH_OPEN, kernel_open, iterations=1)
            
            # Cierre: conecta líneas de crestas rotas
            kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
            closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel_close, iterations=1)

            # --- PASO 3: Adelgazamiento alternativo (sin ximgproc) ---
            # Erosión suave para reducir grosor de crestas
            kernel_thin = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
            refined = cv.morphologyEx(closed, cv.MORPH_ERODE, kernel_thin, iterations=1)

            # --- PASO 4: Filtrado de componentes pequeños ---
            num_labels, labels, stats, _ = cv.connectedComponentsWithStats(refined, connectivity=8)
            min_size = 50  # píxeles mínimos para considerar válido
            
            mask = np.zeros_like(refined)
            for i in range(1, num_labels):  # Ignorar el fondo (label 0)
                if stats[i, cv.CC_STAT_AREA] >= min_size:
                    mask[labels == i] = 255

            # --- PASO 5: INVERTIR COLORES (fondo blanco, crestas negras) ---
            refinada = cv.bitwise_not(mask)

            # --- PASO 6 (OPCIONAL): Rellenar bordes con blanco ---
            # Crea un borde blanco alrededor para eliminar cualquier resto negro en márgenes
            border_size = 10
            refinada = cv.copyMakeBorder(
                refinada, 
                border_size, border_size, border_size, border_size,
                cv.BORDER_CONSTANT, 
                value=255  # Blanco
            )

            # --- PASO 7: Guardar resultado ---
            cv.imwrite(os.path.join(out_dir, filename), refinada)

        print(f"[{user}] refinadas guardadas en '{out_dir}'")