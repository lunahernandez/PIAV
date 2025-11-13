from fileinput import filename
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
    users = os.listdir(db_path)

    for user in users:
        user_folder = os.path.join(out_path, user, "sobel")
        if not os.path.isdir(user_folder):
            continue

        image_files = [f for f in os.listdir(user_folder)
                       if f.lower().endswith(".png")]
        if not image_files:
            continue

        out_dir = os.path.join(out_path, user, "refinadas")
        os.makedirs(out_dir, exist_ok=True)

        for filename in image_files:
            img_path = os.path.join(user_folder, filename)
            image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if image is None:
                continue

            h, w = image.shape
            pct = 0.03           # 3% de cada lado (ajustable)
            m = int(min(h, w) * pct)
            crop = image[m:h-m, m:w-m]
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            clean = cv.morphologyEx(crop, cv.MORPH_OPEN, kernel, iterations=1)
            clean = cv.medianBlur(clean, 3)
            refinada = cv.normalize(clean, None, 0, 255, cv.NORM_MINMAX)
            cv.imwrite(os.path.join(out_dir, filename), refinada)

# SIFT : Comparar huellas del mismo usuario
def comparar_huellas_mismo_usuario(
    db_path,
    out_path,
    nfeatures=6000,
    contrastThreshold=0.01,
    edgeThreshold=10,
    sigma=1.4,
):
    users = os.listdir(db_path)

    for user in users:
        folder = os.path.join(out_path, user, "refinadas")
        if not os.path.isdir(folder):
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
        if len(files) < 2:
            continue

        img1 = cv.imread(os.path.join(folder, files[0]), cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(os.path.join(folder, files[1]), cv.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue

        # Invertir imágenes
        img1_inv = cv.bitwise_not(img1)
        img2_inv = cv.bitwise_not(img2)

        # SIFT
        sift = cv.SIFT_create(
            nfeatures=nfeatures,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma,
        )

        kp1, des1 = sift.detectAndCompute(img1_inv, None)
        kp2, des2 = sift.detectAndCompute(img2_inv, None)

        if des1 is None or des2 is None:
            print(f"\nUsuario {user}: No hay suficientes descriptores")
            continue

        # Matcher
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe Ratio
        good = [m for m, n in matches if m.distance < 0.80 * n.distance]

        if len(good) < 8:
            print(f"\nUsuario {user}: Muy pocos matches")
            continue

        # RANSAC para filtrar inliers
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
        if mask is None:
            print(f"\nUsuario {user}: No homografía válida")
            continue

        inliers = mask.sum()
        ratio = inliers / len(good) * 100
        print(f"\nUsuario {user}: inliers={inliers}, ratio={ratio:.1f}%")

        # Decisión final
        if inliers >= 8 and ratio >= 40:
            print("MATCH - Huellas coinciden")
        else:
            print("NO MATCH - Pocos inliers")

        # Visualización de matches
        inlier_matches = [
            good[i] for i in range(len(good)) if mask[i]
        ]

        vis = cv.drawMatches(
            img1, kp1,
            img2, kp2,
            inlier_matches[:25],
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        cv.imshow(f"Matches entre huellas - {user}", vis)
        cv.waitKey(0)
        cv.destroyAllWindows()
