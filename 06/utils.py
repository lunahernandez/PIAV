import os
import cv2 as cv
import numpy as np
from pathlib import Path


# PREPROCESADO: Buscar y segmentar la ROI
def mejor_roi_por_negros(bw, ventana=500, paso=20, negros_val=255):
    H, W = bw.shape
    w = h = min(ventana, W, H)
    mask = (bw == negros_val).astype(np.uint8)
    integ = cv.integral(mask)

    def suma(x, y, w, h):
        x2, y2 = x + w, y + h
        return int(integ[y2, x2] - integ[y, x2] - integ[y2, x] + integ[y, x])

    best = (0, 0, w, h)
    best_sum = -1
    max_y = max(1, H - h + 1)
    max_x = max(1, W - w + 1)
    for y in range(0, max_y, paso):
        for x in range(0, max_x, paso):
            s = suma(x, y, w, h)
            if s > best_sum:
                best_sum = s
                best = (x, y, w, h)
    return best


def recorte_por_negros(img_path, ventana=500, paso=20, invert=False, out_dir=None):
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    flag = cv.THRESH_BINARY_INV if not invert else cv.THRESH_BINARY
    _, bw = cv.threshold(gray, 0, 255, flag + cv.THRESH_OTSU)

    total_pix = bw.size
    negros = int(np.sum(bw == (255 if not invert else 0)))
    blancos = total_pix - negros

    x, y, w, h = mejor_roi_por_negros(bw, ventana=ventana, paso=paso,
                                      negros_val=(255 if not invert else 0))
    roi = img[y:y+h, x:x+w]

    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        base = Path(img_path).stem
        roi_path = Path(out_dir) / f"{base}.png"
        cv.imwrite(str(roi_path), roi)

    return roi, bw, (x, y, w, h), (negros, blancos)


def recortar_roi(db_path, out_path, ventana=500, paso=15):
    users = os.listdir(db_path)
    for user in users:
        user_folder = os.path.join(db_path, user)
        if not os.path.isdir(user_folder):
            continue

        image_files = [f for f in os.listdir(user_folder) if f.lower().endswith(".png")]
        if len(image_files) < 2:
            continue
        image_files.sort()
        
        out_dir = os.path.join(out_path, user, "roi")
        os.makedirs(out_dir, exist_ok=True)

        for filename in image_files:
            img_path = os.path.join(user_folder, filename)
            recorte_por_negros(img_path, ventana=ventana, paso=paso, invert=False, out_dir=out_dir)


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

        image1 = cv.imread(os.path.join(user_folder, image_files[0]))
        image2 = cv.imread(os.path.join(user_folder, image_files[1]))

        gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

        equalized1 = cv.equalizeHist(gray1)
        equalized2 = cv.equalizeHist(gray2)

        out_dir = os.path.join(out_path, user, "equalized")
        os.makedirs(out_dir, exist_ok=True)

        cv.imwrite(os.path.join(out_dir, image_files[0]), equalized1)
        cv.imwrite(os.path.join(out_dir, image_files[1]), equalized2)


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

        image1 = cv.imread(os.path.join(user_folder, image_files[0]))
        image2 = cv.imread(os.path.join(user_folder, image_files[1]))

        gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

        bilateral1 = cv.bilateralFilter(gray1, d=10, sigmaColor=7, sigmaSpace=11)
        bilateral2 = cv.bilateralFilter(gray2, d=10, sigmaColor=7, sigmaSpace=11)

        out_dir = os.path.join(out_path, user, "bilateral_filter")
        os.makedirs(out_dir, exist_ok=True)

        cv.imwrite(os.path.join(out_dir, image_files[0]), bilateral1)
        cv.imwrite(os.path.join(out_dir, image_files[1]), bilateral2)


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

        image1 = cv.imread(os.path.join(user_folder, image_files[0]))
        image2 = cv.imread(os.path.join(user_folder, image_files[1]))

        gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

        sobel1 = cv.Sobel(gray1, cv.CV_64F, 0, 1, ksize=5)
        sobel2 = cv.Sobel(gray2, cv.CV_64F, 0, 1, ksize=5)

        out_dir = os.path.join(out_path, user, "sobel")
        os.makedirs(out_dir, exist_ok=True)

        # Código de GPT para solventar warning: [ WARN:0@0.762] global loadsave.cpp:1063 cv::imwrite_ Unsupported depth image for selected encoder is fallbacked to CV_8U.
        sobel1_8u = cv.normalize(sobel1, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        sobel2_8u = cv.normalize(sobel2, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        cv.imwrite(os.path.join(out_dir, image_files[0]), sobel1_8u)
        cv.imwrite(os.path.join(out_dir, image_files[1]), sobel2_8u)

# PREPROCESADO: Refinar la ROI