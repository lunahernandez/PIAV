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
            sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
            sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

            sobel = cv.magnitude(sobelx, sobely)
            # _, bw = cv.threshold(sobel, 0, 255, cv.THRESH_BINARY_INV)
            sobel_8u = cv.normalize(sobel, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8) # Código de GPT para solventar warning: [ WARN:0@0.762] global loadsave.cpp:1063 cv::imwrite_ Unsupported depth image for selected encoder is fallbacked to CV_8U.
            cv.imwrite(os.path.join(out_dir, filename), sobel_8u)

# PREPROCESADO: Refinar la ROI
def refinar_roi(db_path, out_path):
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
import os
import cv2 as cv
import numpy as np

# SIFT : Comparar huellas del mismo usuario
def comparar_huellas(
    db_path,
    out_path,
    nfeatures=1500,
    contrastThreshold=0.01,
    edgeThreshold=10,
    sigma=1.4,
):
    users = os.listdir(db_path)

    # Crear SIFT una sola vez
    sift = cv.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma,
    )

    for user in users:
        print(20 * "=")
        print(f"Usuario: {user}")
        print(20 * "=")

        folder = os.path.join(out_path, user, "sobel")
        if not os.path.isdir(folder):
            continue

        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])
        if len(files) < 2:
            continue

        for file in files:
            filename1 = file
            path1 = os.path.join(folder, file)
            img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)

            if img1 is None:
                print(f"(ERROR) {filename1}: no se pudo leer {path1}")
                continue

            # Invertir y calcular SIFT una vez por imagen 1
            img1_inv = cv.bitwise_not(img1)
            kp1, des1 = sift.detectAndCompute(img1_inv, None)
            if des1 is None:
                print(f"(ERROR) {filename1}: sin descriptores SIFT")
                continue

            for user2 in users:
                folder2 = os.path.join(out_path, user2, "sobel")
                if not os.path.isdir(folder2):
                    continue

                files2 = sorted([f for f in os.listdir(folder2) if f.lower().endswith(".png")])
                if len(files2) < 2:
                    continue

                for file2 in files2:
                    filename2 = file2
                    path2 = os.path.join(folder2, file2)

                    # Si no quieres comparar una imagen consigo misma, descomenta:
                    # if user == user2 and filename1 == filename2:
                    #     print(f"(INFO) {filename1}-{filename2}: comparación consigo misma, se omite")
                    #     continue

                    img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)
                    if img2 is None:
                        print(f"(ERROR) {filename1}-{filename2}: no se pudo leer {path2}")
                        continue

                    img2_inv = cv.bitwise_not(img2)
                    kp2, des2 = sift.detectAndCompute(img2_inv, None)
                    if des2 is None:
                        print(f"(ERROR) {filename1}-{filename2}: sin descriptores SIFT en la segunda imagen")
                        continue

                    # Matcher + Lowe ratio (seguro)
                    bf = cv.BFMatcher()
                    matches = bf.knnMatch(des1, des2, k=2)

                    good = []
                    for par in matches:
                        if len(par) < 2:
                            continue
                        m, n = par
                        if m.distance < 0.70 * n.distance:
                            good.append(m)

                    if len(good) < 4:
                        # Ya contamos esto como NO MATCH y lo marcamos como CORRECTO/ERROR según prefijo
                        mismo_prefijo = (filename1[:8] == filename2[:8])
                        if mismo_prefijo:
                            print(f"(ERROR) {filename1}-{filename2} -> NO MATCH (muy pocos matches buenos)")
                        else:
                            print(f"(CORRECTO) {filename1}-{filename2} -> NO MATCH (muy pocos matches buenos)")
                        continue

                    # RANSAC para filtrar inliers
                    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    M, mask = cv.findHomography(src, dst, cv.RANSAC, 8.0)
                    if mask is None:
                        mismo_prefijo = (filename1[:8] == filename2[:8])
                        if mismo_prefijo:
                            print(f"(ERROR) {filename1}-{filename2} -> NO MATCH (no se pudo estimar homografía)")
                        else:
                            print(f"(CORRECTO) {filename1}-{filename2} -> NO MATCH (no se pudo estimar homografía)")
                        continue

                    inliers = int(mask.sum())
                    ratio = inliers / len(good) * 100.0

                    # Decisión final
                    mismo_prefijo = (filename1[:8] == filename2[:8])
                    if inliers >= 4 and ratio >= 25 and mismo_prefijo:
                        print(f"(CORRECTO) {filename1}-{filename2} -> MATCH "
                              f"(inliers={inliers}, ratio={ratio:.1f}%)")
                    elif inliers >= 4 and ratio >= 25 and not mismo_prefijo:
                        print(f"(ERROR) {filename1}-{filename2} -> MATCH "
                              f"(inliers={inliers}, ratio={ratio:.1f}%)")
                    elif (inliers < 4 or ratio < 25) and not mismo_prefijo:
                        print(f"(CORRECTO) {filename1}-{filename2} -> NO MATCH "
                              f"(inliers={inliers}, ratio={ratio:.1f}%)")
                    else:
                        print(f"(ERROR) {filename1}-{filename2} -> NO MATCH "
                              f"(inliers={inliers}, ratio={ratio:.1f}%)")

                    # Visualización de matches (solo inliers)
                    inlier_matches = [
                        good[i] for i in range(len(good)) if mask[i, 0]
                    ]

                    vis = cv.drawMatches(
                        img1, kp1,
                        img2, kp2,
                        inlier_matches[:25],
                        None,
                        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                    )

                    #cv.imshow(f"Matches entre huellas - {user}", vis)
                    cv.waitKey(0)
                    cv.destroyAllWindows()
