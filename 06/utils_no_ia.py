from fileinput import filename
import os
import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm


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
    print("\n[INFO] Cargando imágenes y generando descriptores SIFT...")
    sift = cv.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma,
    )
    bf = cv.BFMatcher()
    dataset = []
    users = sorted(os.listdir(out_path))
    for user in users:
        if user.lower() == "test": continue

        folder = os.path.join(out_path, user, "sobel")
        if not os.path.isdir(folder): continue
        
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])
        for f in files:
            path = os.path.join(folder, f)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is None: continue
            
            img_inv = cv.bitwise_not(img)
            kp, des = sift.detectAndCompute(img_inv, None)
            
            if des is not None:
                user_id = f[:8] 
                dataset.append({'id': user_id, 'filename': f, 'kp': kp, 'des': des, 'path': path})

    num_imgs = len(dataset)
    print(f"[INFO] Se han cargado {num_imgs} huellas válidas.")

    scores_genuinos = []
    scores_impostores = []

    stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    print("\n" + "="*145)
    print(f"{'HUELLA A':<25} | {'HUELLA B':<25} | {'MATCH?':<8} | {'INLIERS':<8} | {'RATIO':<8} | {'RESULTADO':<25}")
    print("="*145)

    for i in range(num_imgs):
        for j in range(i, num_imgs):
            
            img_A = dataset[i]
            img_B = dataset[j]

            matches = bf.knnMatch(img_A['des'], img_B['des'], k=2)
            good = []
            for par in matches:
                if len(par) < 2: continue
                m, n = par
                if m.distance < 0.8 * n.distance:
                    good.append(m)

            inliers = 0
            ratio = 0.0
            match_algoritmo = False

            # RANSAC
            if len(good) >= 8:
                src = np.float32([img_A['kp'][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([img_B['kp'][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                try:
                    M, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
                    if mask is not None:
                        inliers = int(mask.sum())
                        if len(good) > 0:
                            ratio = inliers / len(good) * 100.0
                        
                        # Umbrales
                        if inliers >= 8 and ratio >= 15:
                            match_algoritmo = True
                except:
                    pass

            es_mismo_archivo = (i == j)
            es_misma_persona = (img_A['id'] == img_B['id'])
            
            #  Datos para la gráfica
            if not es_mismo_archivo:
                if es_misma_persona:
                    scores_genuinos.append(ratio)
                else:
                    scores_impostores.append(ratio)

            etiqueta = ""

            if es_mismo_archivo:
                if match_algoritmo:
                    etiqueta = "TP (Auto-Check)"
                    stats["TP"] += 1
                else:
                    etiqueta = "FN (Error SIFT)"
                    stats["FN"] += 1

            elif es_misma_persona:
                if match_algoritmo:
                    etiqueta = "TP (Acierto)"
                    stats["TP"] += 1
                else:
                    etiqueta = "FN (No concidio)"
                    stats["FN"] += 1
            
            else:
                if match_algoritmo:
                    etiqueta = "FP (Error)"
                    stats["FP"] += 1
                else:
                    etiqueta = "TN (Correcto)"
                    stats["TN"] += 1

            ratio_str = f"{ratio:.1f}%"
            print(f"{img_A['filename'][:23]:<25} | {img_B['filename'][:23]:<25} | {str(match_algoritmo):<8} | {inliers:<8} | {ratio_str:<8} | {etiqueta}")

    total = sum(stats.values())
    print("="*145)
    print("RESUMEN GLOBAL")
    print(f"Comparaciones totales: {total}")
    print(f"Aciertos Totales: {stats['TP'] + stats['TN']}")
    print(f"   - Match Correctos (TP): {stats['TP']}")
    print(f"   - Rechazos Correctos (TN): {stats['TN']}")
    print(f"Fallos Totales: {stats['FP'] + stats['FN']}")
    print(f"   - Falsos Positivos (Confundió personas): {stats['FP']}")
    print(f"   - Falsos Negativos (No vio la coincidencia): {stats['FN']}")
    print("="*145)
    return scores_genuinos, scores_impostores

# GRAFICA : Grafica para dibujar la distribución
def dibujar_curvas_error(scores_genuinos, scores_impostores, umbral=15):
    """
    Dibuja las distribuciones de puntuaciones para ver el solapamiento.
    """
    plt.figure(figsize=(10, 6))
    
    x_range = np.linspace(0, 100, 500)

    if len(scores_impostores) > 1:
        mu_imp, std_imp = norm.fit(scores_impostores)
        p_imp = norm.pdf(x_range, mu_imp, std_imp)
        plt.plot(x_range, p_imp, 'b-', lw=2, label='Impostores (Distinta Persona)')
        plt.fill_between(x_range, p_imp, alpha=0.2, color='blue')
    
    if len(scores_genuinos) > 1:
        mu_gen, std_gen = norm.fit(scores_genuinos)
        p_gen = norm.pdf(x_range, mu_gen, std_gen)
        plt.plot(x_range, p_gen, 'r-', lw=2, label='Genuinos (Misma Persona)')
        plt.fill_between(x_range, p_gen, alpha=0.2, color='red')

    plt.axvline(x=umbral, color='k', linestyle='--', label=f'Umbral ({umbral}%)')
    plt.title('Distribución de Puntuaciones (Ratio de Similitud)')
    plt.xlabel('Score (Ratio %)')
    plt.ylabel('Densidad de Probabilidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    print("[GRAFICA] Generando ventana de gráfico...")
    plt.show()

# TEST : Función para testear
def procesar_y_comparar_test(
    test_path, 
    db_processed_path, 
    out_test_path,
    ventana=500, 
    paso=15
):
    """
    AUTENTICACIÓN:
    1. Lee huellas desconocidas desde test_path.
    2. Las procesa (ROI -> Sobel) y las guarda.
    3. Compara cada huella procesada contra TODOS los usuarios REALES en db_processed_path (excluyendo la carpeta 'test').
    4. Devuelve la identidad más probable.
    """

    print(f"\n[TEST] Buscando imágenes en: {test_path}")

    test_files = [f for f in os.listdir(test_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    source_path = test_path
    if not test_files:
        possible_subfolder = os.path.join(test_path, "sobel")
        if os.path.isdir(possible_subfolder):
            print(f"[AVISO] No se encontraron imágenes en la raíz, buscando dentro de: {possible_subfolder}")
            source_path = possible_subfolder
            test_files = [f for f in os.listdir(source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not test_files:
        print("[ERROR] No se encontraron imágenes ni en 'test' ni en 'test/sobel'. Revisa la ruta.")
        return

    out_dir_sobel = os.path.join(out_test_path, "sobel")
    os.makedirs(out_dir_sobel, exist_ok=True)

    test_images_processed = {} # Aquí guardaremos las huellas listas para comparar

    print(f"[TEST] Procesando {len(test_files)} imágenes desconocidas...")

    for filename in test_files:
        img_path = os.path.join(source_path, filename)
        img = cv.imread(img_path)
        
        if img is None: continue
        
        try:
            # ROI
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, thresh_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            x, y, w, h = mejor_roi_por_negros(thresh_img, ventana=ventana, paso=paso)
            roi = img[y:y+h, x:x+w]
            
            # Ecualizar
            roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            equalized = cv.equalizeHist(roi_gray)
            
            # Bilateral
            bilateral = cv.bilateralFilter(equalized, d=10, sigmaColor=10, sigmaSpace=10)
            
            # Sobel
            sobelx = cv.Sobel(bilateral, cv.CV_64F, 1, 0, ksize=3)
            sobely = cv.Sobel(bilateral, cv.CV_64F, 0, 1, ksize=3)
            sobel_mag = cv.magnitude(sobelx, sobely)
            sobel_8u = cv.normalize(sobel_mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

            save_path = os.path.join(out_dir_sobel, filename)
            cv.imwrite(save_path, sobel_8u)

            test_images_processed[filename] = sobel_8u
            print(f"  -> Procesada: {filename}")
            
        except Exception as e:
            print(f"  (Saltada {filename}): Error en preprocesado ({e})")

    print(f"\n[AUTH] Iniciando autenticación contra la base de datos...")
    
    if not os.path.exists(db_processed_path):
        print(f"[ERROR] No existe la carpeta de datos procesados: {db_processed_path}")
        return

    sift = cv.SIFT_create(nfeatures=1500, contrastThreshold=0.01, edgeThreshold=10, sigma=1.4)
    bf = cv.BFMatcher()
    db_users = [u for u in os.listdir(db_processed_path) 
                if os.path.isdir(os.path.join(db_processed_path, u, "sobel")) and u.lower() != "test"]

    print(f"[AUTH] Comparando contra {len(db_users)} usuarios registrados.")

    for test_name, test_img in test_images_processed.items():
        print("\n" + "-"*60)
        print(f"Identificando huella desconocida: {test_name}")

        test_img_inv = cv.bitwise_not(test_img)
        kp_test, des_test = sift.detectAndCompute(test_img_inv, None)
        
        if des_test is None:
            print("  (Warning) Sin rasgos suficientes para identificar.")
            continue
            
        mejor_match_user = "Desconocido"
        mejor_inliers = 0
        mejor_ratio = 0.0
        candidato_fichero = ""

        for user in db_users:
            user_sobel_path = os.path.join(db_processed_path, user, "sobel")
            db_files = [f for f in os.listdir(user_sobel_path) if f.lower().endswith(".png")]
            
            for db_file in db_files:
                path_db = os.path.join(user_sobel_path, db_file)
                img_db = cv.imread(path_db, cv.IMREAD_GRAYSCALE)
                if img_db is None: continue
                
                img_db_inv = cv.bitwise_not(img_db)
                kp_db, des_db = sift.detectAndCompute(img_db_inv, None)
                if des_db is None: continue
                
                # KNN Match
                matches = bf.knnMatch(des_test, des_db, k=2)
                
                good = []
                for par in matches:
                    if len(par) < 2: continue
                    m, n = par
                    # Lowe Ratio estricto
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                
                # RANSAC para verificar geometría
                if len(good) >= 8:
                    src_pts = np.float32([kp_test[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_db[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    
                    try:
                        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                        
                        if mask is not None:
                            inliers = int(mask.sum())
                            ratio = inliers / len(good) * 100.0
                            
                            # Si este match es mejor que el que teníamos, lo guardamos como "Mejor Candidato"
                            # Umbral mínimo para considerar candidato: 8 inliers
                            if inliers > mejor_inliers and inliers >= 8:
                                mejor_inliers = inliers
                                mejor_ratio = ratio
                                mejor_match_user = user # Guardamos la carpeta del usuario (ej. crd_0811f)
                                candidato_fichero = db_file
                    except:
                        pass

        if mejor_inliers >= 12: 
            print(f"  >>> IDENTIFICADO: {mejor_match_user}")
            print(f"      (Evidencia muy fuerte: {mejor_inliers} inliers con {candidato_fichero})")
        elif mejor_inliers >= 8:
            print(f"  >>> POSIBLE IDENTIFICACIÓN: {mejor_match_user}")
            print(f"      (Evidencia moderada: {mejor_inliers} inliers con {candidato_fichero})")
        else:
            print(f"  >>> NO IDENTIFICADO (No coincide con nadie registrado)")