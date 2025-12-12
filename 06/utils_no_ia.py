import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json
import itertools
from scipy.stats import norm

# REPROCESAMIENTO

def mejor_roi_por_negros(image, ventana=500, paso=20):
    """
    Esta función sirve para encontrar la región de interés (ROI) 
    en una imagen que contiene la mayor cantidad de píxeles negros.
    """
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
    Esta función recorta la región de interés (ROI) en las imágenes de una base de datos,
    utilizando la función mejor_roi_por_negros para encontrar la región con más píxeles negros.
    """
    if not os.path.exists(db_path): return
    items = os.listdir(db_path)
    images = [f for f in items if f.lower().endswith(('.png', '.jpg'))]
    users = [d for d in items if os.path.isdir(os.path.join(db_path, d))]
    tareas = []
    if images:
        dst_folder = os.path.join(out_path, "roi")
        tareas.append((db_path, dst_folder))
    else:
        for user in users:
            tareas.append((os.path.join(db_path, user), os.path.join(out_path, user, "roi")))
    for src_dir, dst_dir in tareas:
        os.makedirs(dst_dir, exist_ok=True)
        for filename in [f for f in os.listdir(src_dir) if f.lower().endswith(".png")]:
            img = cv.imread(os.path.join(src_dir, filename))
            if img is None: continue
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            x, y, w, h = mejor_roi_por_negros(image, ventana=ventana, paso=paso)
            cv.imwrite(os.path.join(dst_dir, filename), img[y:y+h, x:x+w])

def procesar_fase(fase_func, input_folder, output_folder, db_path, out_base):
    """Esta función procesa una fase específica del pipeline de procesamiento de imágenes."""
    items = os.listdir(db_path)
    users = [d for d in items if os.path.isdir(os.path.join(db_path, d))]
    tareas = []
    check_path = os.path.join(out_base, input_folder)
    if os.path.exists(check_path) and any(f.endswith('.png') for f in os.listdir(check_path)):
         tareas.append((check_path, os.path.join(out_base, output_folder)))
    else:
        for user in users:
            src = os.path.join(out_base, user, input_folder)
            if os.path.isdir(src):
                tareas.append((src, os.path.join(out_base, user, output_folder)))     
    for src_dir, dst_dir in tareas:
        os.makedirs(dst_dir, exist_ok=True)
        for filename in [f for f in os.listdir(src_dir) if f.lower().endswith('.png')]:
            fase_func(src_dir, dst_dir, filename)

def fase_ecualizar(src, dst, name):
    """Ecualiza el histograma de la imagen en escala de grises."""
    img = cv.imread(os.path.join(src, name))
    if img is not None:
        cv.imwrite(os.path.join(dst, name), cv.equalizeHist(cv.cvtColor(img, cv.COLOR_BGR2GRAY)))

def fase_bilateral(src, dst, name):
    """Aplica un filtro bilateral para reducir el ruido mientras se preservan los bordes."""
    img = cv.imread(os.path.join(src, name))
    if img is not None:
        cv.imwrite(os.path.join(dst, name), cv.bilateralFilter(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 10, 10, 10))

def fase_sobel(src, dst, name):
    """Aplica el operador Sobel para resaltar los bordes en la imagen."""
    img = cv.imread(os.path.join(src, name))
    if img is not None:
        g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sx = cv.Sobel(g, cv.CV_64F, 1, 0, ksize=3)
        sy = cv.Sobel(g, cv.CV_64F, 0, 1, ksize=3)
        m = cv.magnitude(sx, sy)
        cv.imwrite(os.path.join(dst, name), cv.normalize(m, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8))

def fase_refinar(src, dst, name):
    """Refina la imagen aplicando un filtro morfológico y un desenfoque mediano."""
    img = cv.imread(os.path.join(src, name), cv.IMREAD_GRAYSCALE)
    if img is not None:
        h, w = img.shape
        m = int(min(h, w) * 0.03)
        crop = img[m:h-m, m:w-m]
        cl = cv.morphologyEx(crop, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=1)
        cv.imwrite(os.path.join(dst, name), cv.normalize(cv.medianBlur(cl, 3), None, 0, 255, cv.NORM_MINMAX))

def ejecutar_pipeline_completo(raw_path, out_path):
    """Ejecuta todo el pipeline de preprocesamiento de imágenes."""
    print(f"Procesando: {raw_path} -> {out_path}")
    recortar_roi(raw_path, out_path)
    procesar_fase(fase_ecualizar, "roi", "equalized", raw_path, out_path)
    procesar_fase(fase_bilateral, "equalized", "bilateral_filter", raw_path, out_path)
    procesar_fase(fase_sobel, "bilateral_filter", "sobel", raw_path, out_path)
    procesar_fase(fase_refinar, "sobel", "refinadas", raw_path, out_path)

# --- SIFT Y MÉTRICAS ---

def cargar_dataset(out_path, sift):
    """Carga todas las imágenes de entrenamiento, ignorando la carpeta 'test'."""
    dataset = []
    users = sorted([u for u in os.listdir(out_path) 
                   if os.path.isdir(os.path.join(out_path, u)) and u.lower() != "test"])
    
    print(f"Cargando dataset de entrenamiento ({len(users)} usuarios)...")
    for user in users:
        folder = os.path.join(out_path, user, "refinadas")
        if not os.path.isdir(folder): continue
        for f in sorted([x for x in os.listdir(folder) if x.lower().endswith(".png")]):
            img = cv.imread(os.path.join(folder, f), cv.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = sift.detectAndCompute(cv.bitwise_not(img), None)
                if des is not None and len(kp) > 0:
                    dataset.append({'id': user, 'filename': f, 'kp': kp, 'des': des})
    return dataset

def calcular_similitud_raw(bf, kp_a, des_a, kp_b, des_b):
    """Calcula la similitud entre dos conjuntos de keypoints y descriptores usando KNNMatcher y RANSAC."""
    # DEBUG: Información de entrada
    print(f"   [DEBUG_MATH] Puntos detectados -> ImgA: {len(kp_a) if kp_a else 0}, ImgB: {len(kp_b) if kp_b else 0}")

    if des_a is None or des_b is None: return 0.0
    if len(kp_a) == 0 or len(kp_b) == 0: return 0.0
    
    try:
        matches = bf.knnMatch(des_a, des_b, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        # DEBUG: Matches tras filtro de Lowe
        print(f"   [DEBUG_MATH] Matches 'Good' (Lowe): {len(good)}")

        if len(good) >= 4:
            src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            _, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
            
            if mask is not None:
                inliers = int(mask.sum())
                score = (inliers / len(kp_a)) * 100.0
                # DEBUG: Resultado final de esta comparación
                print(f"   [DEBUG_MATH] Inliers (RANSAC): {inliers} -> Score: {score:.4f}%")
                return score

    except Exception as e:
        print(f"   [ERROR] Excepción matemática: {e}")
        pass
    
    return 0.0

# GESTIÓN DE HIPERPARÁMETROS

def gestionar_optimizacion_sift(out_path, params_file):
    """Gestiona la búsqueda de hiperparámetros con menú interactivo."""
    mejor_config = {'nfeatures': 0, 'contrastThreshold': 0.04, 'edgeThreshold': 10, 'sigma': 1.6}
    
    if os.path.exists(params_file):
        print(f"Archivo de configuración encontrado: {params_file}")
        resp = input("¿Recalcular hiperparámetros (s/n)? ").lower().strip()
        if resp != 's':
            with open(params_file, 'r') as f: return json.load(f)

    print("Iniciando búsqueda de mejores parámetros (GRID SEARCH)...")
    
    # Muestra pequeña para ir rápido
    users = sorted([u for u in os.listdir(out_path) if os.path.isdir(os.path.join(out_path, u)) and u != "test"])
    sample_imgs = []
    for u in users[:3]: 
        d = os.path.join(out_path, u, "refinadas")
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.png')]
            sample_imgs.extend(files[:2]) 
    
    param_grid = {
        'nfeatures': [0, 2000, 5000],
        'contrastThreshold': [0.03, 0.04],
        'edgeThreshold': [10],
        'sigma': [1.6]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_eer = 100.0
    
    print(f"Se probarán {len(combinations)} combinaciones diferentes...")

    for idx, params in enumerate(combinations):
        print(f"Probando conf {idx+1}/{len(combinations)}: {params} ... ", end="")
        
        # DEBUG: Aviso sobre nfeatures=0
        if params['nfeatures'] == 0:
            print("\n   [INFO] nfeatures=0 detectado: Se cogerán TODAS las características posibles (sin límite).")
        
        sift = cv.SIFT_create(**params)
        bf = cv.BFMatcher()
        gen, imp = [], []
        
        kps_descs = []
        for p in sample_imgs:
            im = cv.imread(p, cv.IMREAD_GRAYSCALE)
            kp, des = sift.detectAndCompute(cv.bitwise_not(im), None)
            kps_descs.append((kp, des))
            
        for i in range(len(kps_descs)):
            for j in range(i+1, len(kps_descs)):
                kp1, des1 = kps_descs[i]
                kp2, des2 = kps_descs[j]
                score = calcular_similitud_raw(bf, kp1, des1, kp2, des2)
                
                u1 = sample_imgs[i].split(os.sep)[-3]
                u2 = sample_imgs[j].split(os.sep)[-3]
                if u1 == u2: gen.append(score)
                else: imp.append(score)
        
        if not gen or not imp:
            print("Sin datos.")
            continue
            
        u_mean = (np.mean(gen) + np.mean(imp)) / 2
        frr = np.mean(np.array(gen) < u_mean)
        far = np.mean(np.array(imp) >= u_mean)
        eer = (frr + far) / 2
        
        print(f"EER aprox: {eer*100:.2f}%")
        if eer < best_eer:
            best_eer = eer
            mejor_config = params
            
    print(f"Mejor configuración guardada: {mejor_config}")
    with open(params_file, 'w') as f: json.dump(mejor_config, f)
    return mejor_config

def generar_scores_entrenamiento(out_path, params):
    """Genera los scores de similitud para todas las imágenes de entrenamiento."""
    sift = cv.SIFT_create(**params)
    bf = cv.BFMatcher()
    
    dataset = cargar_dataset(out_path, sift)
    if len(dataset) < 2:
        print("Error: No hay suficientes datos para entrenar.")
        return [], []

    genuinos = []
    impostores = []
    
    print("Ejecutando comparaciones Todos contra Todos...", end="")
    count = 0
    for i in range(len(dataset)):
        for j in range(i + 1, len(dataset)):
            img_A = dataset[i]
            img_B = dataset[j]
            
            # DEBUG: Ver qué archivos se comparan
            print(f"\n[DEBUG] Comparando {img_A['filename']} vs {img_B['filename']}")
            
            score = calcular_similitud_raw(bf, img_A['kp'], img_A['des'], img_B['kp'], img_B['des'])
            
            if img_A['id'] == img_B['id']:
                genuinos.append(score)
            else:
                impostores.append(score)
            
            count += 1
            if count % 500 == 0: print(".", end="", flush=True)
            
    print(f" Hecho. {len(genuinos)} Genuinos, {len(impostores)} Impostores.")
    return np.array(genuinos), np.array(impostores)

def calcular_umbral_optimo(genuinos, impostores):
    """Calcula el umbral óptimo que minimiza la diferencia entre FAR y FRR."""
    if len(genuinos) == 0 or len(impostores) == 0: return 0.5
    
    max_score = max(np.max(genuinos), np.max(impostores))
    if max_score == 0: return 0.0
    
    umbrales = np.linspace(0, max_score, 1000)
    
    mejor_diff = float('inf')
    mejor_umbral = 0.0
    
    for u in umbrales:
        frr = np.mean(genuinos < u)
        far = np.mean(impostores >= u)
        diff = abs(frr - far)
        
        if diff < mejor_diff:
            mejor_diff = diff
            mejor_umbral = u
            
    return mejor_umbral

def dibujar_curvas_entrenamiento(genuinos, impostores, umbral):
    """Dibuja las curvas de distribución de scores y la curva DET."""
    if len(genuinos) == 0 or len(impostores) == 0: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Resultados del Entrenamiento (Todos contra Todos)', fontsize=16)

    # GAUSSIANAS
    x_max = max(np.max(genuinos), np.max(impostores)) * 1.2
    x = np.linspace(0, x_max, 500)
    
    mu_g, std_g = norm.fit(genuinos)
    if std_g == 0: std_g = 0.001
    p_g = norm.pdf(x, mu_g, std_g)
    ax1.plot(x, p_g, 'g-', lw=2, label='Genuinos')
    ax1.fill_between(x, p_g, where=(x < umbral), color='green', alpha=0.3, label='FRR')
    
    mu_i, std_i = norm.fit(impostores)
    if std_i == 0: std_i = 0.001
    p_i = norm.pdf(x, mu_i, std_i)
    ax1.plot(x, p_i, 'r-', lw=2, label='Impostores')
    ax1.fill_between(x, p_i, where=(x >= umbral), color='red', alpha=0.3, label='FAR')
    
    ax1.axvline(umbral, color='k', linestyle='--', label=f'Umbral: {umbral:.2f}')
    ax1.set_title('Distribución de Scores')
    ax1.set_xlabel('Score de Similitud (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # CURVA DET
    far_list, frr_list = [], []
    thresholds = np.linspace(0, x_max, 1000)
    
    for t in thresholds:
        far = np.mean(impostores >= t)
        frr = np.mean(genuinos < t)
        far_list.append(far)
        frr_list.append(frr)
        
    ax2.plot(far_list, frr_list, 'b-', lw=3, label='Curva DET')
    
    far_op = np.mean(impostores >= umbral)
    frr_op = np.mean(genuinos < umbral)
    ax2.plot(far_op, frr_op, 'ro', markersize=10, label=f'Punto Operativo (Umbral {umbral:.2f})')
    
    ax2.set_title('Curva DET')
    ax2.set_xlabel('Falsa Aceptación (FAR)')
    ax2.set_ylabel('Falso Rechazo (FRR)')
    ax2.grid(True, which='both', alpha=0.5)
    ax2.set_xscale('log') 
    ax2.set_yscale('log')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def test_final_contrastar(test_path, out_path_train, params, umbral):
    """Realiza la verificación final en modo test, comparando imágenes de test contra referencias de un usuario seleccionado."""
    users = sorted([d for d in os.listdir(out_path_train) if os.path.isdir(os.path.join(out_path_train, d)) and d.lower() != "test"])
    
    print("\n--- SELECCIÓN DE USUARIO PARA VERIFICACIÓN ---")
    for i, u in enumerate(users): print(f"[{i}] {u}")
    sel = input(f"Elige usuario (Enter={users[0]}): ").strip()
    
    ref_user = users[int(sel)] if sel.isdigit() and int(sel) < len(users) else users[0]
    
    print(f"Usuario Referencia: {ref_user}")
    print(f"INFO: Se aplicará corte estricto > {umbral:.4f}") 

    sift = cv.SIFT_create(**params)
    bf = cv.BFMatcher()
    
    ref_dir = os.path.join(out_path_train, ref_user, "refinadas")
    refs = []
    
    if os.path.exists(ref_dir):
        for f in os.listdir(ref_dir):
            img = cv.imread(os.path.join(ref_dir, f), cv.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = sift.detectAndCompute(cv.bitwise_not(img), None)
                if des is not None and len(kp) > 0:
                    refs.append((kp, des)) 

    if not refs:
        print("Error: No se encontraron referencias para este usuario.")
        return

    test_dir = os.path.join(test_path, "refinadas")
    print(f"\n{'FICHERO TEST':<30} | {'SCORE':<8} | {'DECISIÓN':<10}")
    print("-" * 55)
    
    if not os.path.exists(test_dir):
        print("La carpeta test no existe.")
        return

    for f in sorted(os.listdir(test_dir)):
        if not f.endswith('.png'): continue
        
        img = cv.imread(os.path.join(test_dir, f), cv.IMREAD_GRAYSCALE)
        if img is None: continue

        kp_t, des_t = sift.detectAndCompute(cv.bitwise_not(img), None)
        
        mejor_score = 0.0

        for i, (kp_r, des_r) in enumerate(refs):
            
            # DEBUG: Ver comparación
            print(f"  [DEBUG] Contrastando {f} vs Referencia_{i}")
            
            score = calcular_similitud_raw(bf, kp_t, des_t, kp_r, des_r)
            if score > mejor_score: mejor_score = score
        
        decision = "ACEPTADO" if mejor_score > umbral else "RECHAZADO"
        
        print(f"{f:<30} | {mejor_score:<8.2f} | {decision:<10}")