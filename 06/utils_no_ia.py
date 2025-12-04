from fileinput import filename
import os
import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm

# PREPROCESADO: Buscar y segmentar la ROI
def mejor_roi_por_negros(image, ventana=500, paso=20):
    """Encuentra la región de interés (ROI) en la imagen que contiene la mayor cantidad de píxeles negros."""

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
    """
    Detecta si db_path contiene carpetas de usuarios (Data) o imágenes sueltas (Test)
    y aplica el recorte ROI guardando en la estructura correspondiente.
    """

    if not os.path.exists(db_path):
        return

    items = os.listdir(db_path)

    images = [f for f in items if f.lower().endswith(('.png', '.jpg'))] # Datos de Test
    users = [d for d in items if os.path.isdir(os.path.join(db_path, d))] # Datos de Data

    tareas = [] # Lista de tuplas (origen, destino)

    if len(images) > 0:
        # MODO TEST
        dst_folder = os.path.join(out_path, "roi")
        tareas.append((db_path, dst_folder))
    else:
        # MODO DATA
        for user in users:
            src = os.path.join(db_path, user)
            dst = os.path.join(out_path, user, "roi")
            tareas.append((src, dst))

    for src_dir, dst_dir in tareas:
        os.makedirs(dst_dir, exist_ok=True)
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(".png")]
        
        for filename in files:
            img_path = os.path.join(src_dir, filename)
            img = cv.imread(img_path)

            if img is None: continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            x, y, w, h = mejor_roi_por_negros(image, ventana=ventana, paso=paso)
            roi = img[y:y+h, x:x+w]
            cv.imwrite(os.path.join(dst_dir, filename), roi)


# PREPROCESADO: Ecualizar y normalizar el histograma
def ecualizar_histograma(db_path, out_path):
    """Ecualiza y normaliza el histograma de las imágenes en db_path y guarda los resultados en out_path."""

    items = os.listdir(db_path)
    images = [f for f in items if f.lower().endswith(('.png', '.jpg'))]
    users = [d for d in items if os.path.isdir(os.path.join(db_path, d))]

    tareas = []

    if len(images) > 0:
        # MODO TEST
        src = os.path.join(out_path, "roi")
        dst = os.path.join(out_path, "equalized")
        if os.path.isdir(src):
            tareas.append((src, dst))
    else:
        # MODO DATA
        for user in users:
            src = os.path.join(out_path, user, "roi")
            dst = os.path.join(out_path, user, "equalized")
            if os.path.isdir(src):
                tareas.append((src, dst))

    for src_dir, dst_dir in tareas:
        os.makedirs(dst_dir, exist_ok=True)
        image_files = [f for f in os.listdir(src_dir) if f.lower().endswith('.png')]

        for filename in image_files:
            img_path = os.path.join(src_dir, filename)
            image = cv.imread(img_path)
            if image is None: continue
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            equalized = cv.equalizeHist(gray)
            cv.imwrite(os.path.join(dst_dir, filename), equalized)


# PREPROCESADO: Aplicar un filtro bilateral
def aplicar_filtro_bilateral(db_path, out_path):
    """Aplica un filtro bilateral a las imágenes en db_path y guarda los resultados en out_path."""

    items = os.listdir(db_path)
    images = [f for f in items if f.lower().endswith(('.png', '.jpg'))]
    users = [d for d in items if os.path.isdir(os.path.join(db_path, d))]

    tareas = []

    if len(images) > 0:
        # MODO TEST
        src = os.path.join(out_path, "equalized")
        dst = os.path.join(out_path, "bilateral_filter")
        if os.path.isdir(src):
            tareas.append((src, dst))
    else:
        # MODO DATA
        for user in users:
            src = os.path.join(out_path, user, "equalized")
            dst = os.path.join(out_path, user, "bilateral_filter")
            if os.path.isdir(src):
                tareas.append((src, dst))

    for src_dir, dst_dir in tareas:
        os.makedirs(dst_dir, exist_ok=True)
        image_files = [f for f in os.listdir(src_dir) if f.lower().endswith('.png')]

        for filename in image_files:
            img_path = os.path.join(src_dir, filename)
            image = cv.imread(img_path)
            if image is None: continue
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            bilateral = cv.bilateralFilter(gray, d=10, sigmaColor=10, sigmaSpace=10)
            cv.imwrite(os.path.join(dst_dir, filename), bilateral)


# PREPROCESADO: Aplicar un realce de crestas
def realzar_crestas(db_path, out_path):
    """Aplica un realce de crestas a las imágenes en db_path y guarda los resultados en out_path."""

    items = os.listdir(db_path)
    images = [f for f in items if f.lower().endswith(('.png', '.jpg'))]
    users = [d for d in items if os.path.isdir(os.path.join(db_path, d))]

    tareas = []

    if len(images) > 0:
        # MODO TEST
        src = os.path.join(out_path, "bilateral_filter")
        dst = os.path.join(out_path, "sobel")
        if os.path.isdir(src):
            tareas.append((src, dst))
    else:
        # MODO DATA
        for user in users:
            src = os.path.join(out_path, user, "bilateral_filter")
            dst = os.path.join(out_path, user, "sobel")
            if os.path.isdir(src):
                tareas.append((src, dst))

    for src_dir, dst_dir in tareas:
        os.makedirs(dst_dir, exist_ok=True)
        image_files = [f for f in os.listdir(src_dir) if f.lower().endswith('.png')]

        for filename in image_files:
            img_path = os.path.join(src_dir, filename)
            image = cv.imread(img_path)
            if image is None: continue
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
            sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

            sobel = cv.magnitude(sobelx, sobely)
            sobel_8u = cv.normalize(sobel, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            cv.imwrite(os.path.join(dst_dir, filename), sobel_8u)

# PREPROCESADO: Refinar la ROI
def refinar_roi(db_path, out_path):
    """Refina la ROI de las imágenes en db_path y guarda los resultados en out_path."""

    items = os.listdir(db_path)
    images = [f for f in items if f.lower().endswith(('.png', '.jpg'))]
    users = [d for d in items if os.path.isdir(os.path.join(db_path, d))]

    tareas = []

    if len(images) > 0:
        # MODO TEST
        src = os.path.join(out_path, "sobel")
        dst = os.path.join(out_path, "refinadas")
        if os.path.isdir(src):
            tareas.append((src, dst))
    else:
        # MODO DATA
        for user in users:
            src = os.path.join(out_path, user, "sobel")
            dst = os.path.join(out_path, user, "refinadas")
            if os.path.isdir(src):
                tareas.append((src, dst))

    for src_dir, dst_dir in tareas:
        os.makedirs(dst_dir, exist_ok=True)
        image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(".png")]

        for filename in image_files:
            img_path = os.path.join(src_dir, filename)
            image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if image is None: continue

            h, w = image.shape
            pct = 0.03
            m = int(min(h, w) * pct)
            crop = image[m:h-m, m:w-m]
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            clean = cv.morphologyEx(crop, cv.MORPH_OPEN, kernel, iterations=1)
            clean = cv.medianBlur(clean, 3)
            refinada = cv.normalize(clean, None, 0, 255, cv.NORM_MINMAX)
            cv.imwrite(os.path.join(dst_dir, filename), refinada)

# FUNCIÓN AUXILIAR: Calcular similitud entre dos descriptores
def calcular_similitud(bf, kp_a, des_a, kp_b, des_b):
    """Compara dos conjuntos de descriptores y devuelve inliers, ratio y si es match."""

    matches = bf.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in matches if m.distance < 0.8 * n.distance]

    inliers = 0
    ratio = 0.0
    es_match = False

    if len(good) >= 8:
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        try:
            _, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.sum())
                if len(good) > 0:
                    ratio = (inliers / len(good)) * 100.0
                
                if inliers >= 8 and ratio >= 15:
                    es_match = True
        except Exception:
            pass

    return inliers, ratio, es_match

# FUNCIÓN AUXILIAR: Cargar dataset y extraer descriptores SIFT
def cargar_dataset(out_path, sift):
    """Carga las imágenes procesadas y genera descriptores SIFT."""

    dataset = []
    print("\n[INFO] Cargando imágenes y generando descriptores SIFT...")
    
    users = sorted([u for u in os.listdir(out_path) if u.lower() != "test"])
    
    for user in users:
        folder = os.path.join(out_path, user, "refinadas")
        if not os.path.isdir(folder): continue
        
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])
        
        for f in files:
            path = os.path.join(folder, f)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is None: continue
            
            kp, des = sift.detectAndCompute(cv.bitwise_not(img), None)
            
            if des is not None:
                dataset.append({
                    'id': user,
                    'filename': f,
                    'kp': kp, 
                    'des': des
                })
                
    print(f"[INFO] Se han cargado {len(dataset)} huellas válidas.")
    return dataset

# COMPARAR HUELLAS: Función principal para comparar huellas en la base de datos
def comparar_huellas(db_path, out_path, nfeatures=1500):
    """Compara todas las huellas en la base de datos y devuelve listas de scores genuinos e impostores."""

    sift = cv.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.01, edgeThreshold=10, sigma=1.4)
    bf = cv.BFMatcher()
    
    dataset = cargar_dataset(out_path, sift)
    num_imgs = len(dataset)
    
    scores_genuinos = []
    scores_impostores = []
    stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    print("\n" + "="*125)
    print(f"{'HUELLA A':<25} | {'HUELLA B':<25} | {'MATCH?':<8} | {'INL':<4} | {'RATIO':<6} | {'RESULTADO'}")
    print("="*125)

    for i in range(num_imgs):
        for j in range(i, num_imgs):
            
            img_A = dataset[i]
            img_B = dataset[j]
            
            inliers, ratio, match_algoritmo = calcular_similitud(
                bf, img_A['kp'], img_A['des'], img_B['kp'], img_B['des']
            )

            es_mismo_archivo = (i == j)
            es_misma_persona = (img_A['id'] == img_B['id'])
            
            etiqueta = "TN"
            
            if es_mismo_archivo:
                etiqueta = "TP (Auto)" if match_algoritmo else "FN (Error)"
                stats["TP" if match_algoritmo else "FN"] += 1
            elif es_misma_persona:
                etiqueta = "TP" if match_algoritmo else "FN"
                stats["TP" if match_algoritmo else "FN"] += 1
                scores_genuinos.append(ratio)
            else:
                etiqueta = "FP" if match_algoritmo else "TN"
                stats["FP" if match_algoritmo else "TN"] += 1
                scores_impostores.append(ratio)

            print(f"{img_A['filename'][:23]:<25} | {img_B['filename'][:23]:<25} | {str(match_algoritmo):<8} | {inliers:<4} | {ratio:.1f}% | {etiqueta}")

    total = sum(stats.values())
    print("="*125)
    print(f"RESUMEN: Total {total} | Aciertos: {stats['TP']+stats['TN']} | Fallos: {stats['FP']+stats['FN']}")
    print(f"TP: {stats['TP']} | TN: {stats['TN']} | FP: {stats['FP']} | FN: {stats['FN']}")
    print("="*125)
    
    return scores_genuinos, scores_impostores

# GRAFICA : Función para dibujar las curvas de error
def dibujar_curvas_error(scores_genuinos, scores_impostores, umbral=15):
    """Dibuja las distribuciones de scores genuinos e impostores con un umbral indicado."""

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

# TEST : Función para testear (MODIFICADA: SIN PREPROCESAMIENTO INTERNO)
def procesar_y_comparar_test(
    test_path, 
    db_processed_path, 
    out_test_path,
    ventana=500, 
    paso=15
):
    """Procesa las imágenes de test ya preprocesadas y las compara contra la base de datos procesada."""

    source_path = os.path.join(out_test_path, "refinadas")
    print(f"\n[TEST] Buscando imágenes PROCESADAS en: {source_path}")
    
    if not os.path.isdir(source_path):
        print(f"[ERROR] No existe la carpeta procesada: {source_path}")
        return

    test_files = [f for f in os.listdir(source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not test_files:
        print("[ERROR] No se encontraron imágenes procesadas.")
        return

    test_images_processed = {} 

    print(f"[TEST] Cargando {len(test_files)} imágenes para identificar...")

    for filename in test_files:
        img_path = os.path.join(source_path, filename)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        
        if img is not None:
            test_images_processed[filename] = img
            print(f"Imagen Cargada: {filename}")

    print(f"\n[AUTH] Iniciando autenticación contra la base de datos...")
    
    if not os.path.exists(db_processed_path):
        print(f"[ERROR] No existe la carpeta de datos procesados: {db_processed_path}")
        return

    sift = cv.SIFT_create(nfeatures=1500, contrastThreshold=0.01, edgeThreshold=10, sigma=1.4)
    bf = cv.BFMatcher()

    dataset_db = cargar_dataset(db_processed_path, sift)
    
    print(f"[AUTH] Comparando contra {len(dataset_db)} huellas registradas.")

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

        for db_item in dataset_db:
            inliers, ratio, _ = calcular_similitud(
                bf, kp_test, des_test, db_item['kp'], db_item['des']
            )

            if inliers > mejor_inliers and inliers >= 8:
                mejor_inliers = inliers
                mejor_ratio = ratio
                mejor_match_user = db_item['id']
                candidato_fichero = db_item['filename']

        if mejor_inliers >= 12: 
            print(f"[IDENTIFICADO] {mejor_match_user}")
            print(f"Evidencia muy fuerte: {mejor_inliers} inliers con {candidato_fichero})")
        elif mejor_inliers >= 8:
            print(f"[POSIBLE IDENTIFICACIÓN] {mejor_match_user}")
            print(f"(Evidencia moderada: {mejor_inliers} inliers con {candidato_fichero})")
        else:
            print(f"[NO IDENTIFICADO] No coincide con nadie registrado")