import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json
import itertools
from scipy.stats import norm

# PREPROCESAMIENTO

def recorrer_y_procesar(input_base, output_base, funcion_procesamiento, **kwargs):
    """Recorre input_base (posible estructura de usuarios) y aplica funcion_procesamiento a cada imagen."""

    if not os.path.exists(input_base): return
    items = os.listdir(input_base)
    es_directorio_usuarios = any(os.path.isdir(os.path.join(input_base, i)) for i in items)
    
    tareas = []
    if es_directorio_usuarios:
        users = [d for d in items if os.path.isdir(os.path.join(input_base, d))]
        for u in users:
            # Crea la ruta destino: output/refinadas/Usuario
            tareas.append((os.path.join(input_base, u), os.path.join(output_base, u)))
    else:
        tareas.append((input_base, output_base))

    for src_dir, dst_dir in tareas:
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if not f.lower().endswith(('.png', '.jpg')): continue
            img = cv.imread(os.path.join(src_dir, f))
            if img is not None:
                funcion_procesamiento(img, os.path.join(dst_dir, f), **kwargs)

def mejor_roi_por_negros(image, ventana=500, paso=20):
    """Encuentra el mejor ROI basado en la cantidad de píxeles negros en una ventana deslizante."""

    H, W = image.shape
    w, h = min(ventana, W, H), min(ventana, W, H)
    mejor_roi = (0, 0, w, h)
    max_negros = -1
    for y in range(0, H - h + 1, paso):
        for x in range(0, W - w + 1, paso):
            negros = np.sum(image[y:y+h, x:x+w] == 0)
            if negros > max_negros:
                max_negros = negros
                mejor_roi = (x, y, w, h)
    return mejor_roi

def aplicar_recorte(img, save_path, ventana=500, paso=15):
    """Aplica un recorte basado en el mejor ROI encontrado por cantidad de píxeles negros."""

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    x, y, w, h = mejor_roi_por_negros(binary, ventana, paso)
    cv.imwrite(save_path, img[y:y+h, x:x+w])

def aplicar_ecualizacion(img, save_path):
    """Aplica ecualización de histograma a la imagen."""

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite(save_path, cv.equalizeHist(gray))

def aplicar_bilateral(img, save_path):
    """Aplica un filtro bilateral para suavizar la imagen manteniendo los bordes."""

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    cv.imwrite(save_path, cv.bilateralFilter(gray, 10, 10, 10))

def aplicar_sobel(img, save_path):
    """Aplica el filtro Sobel para la detección de bordes."""

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    m = cv.magnitude(sx, sy)
    cv.imwrite(save_path, cv.normalize(m, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8))

def aplicar_refinado(img, save_path):
    """Aplica un refinado final mediante apertura morfológica y mediana."""

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape
    m = int(min(h, w) * 0.03)
    crop = gray[m:h-m, m:w-m]
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    cl = cv.morphologyEx(crop, cv.MORPH_OPEN, kernel, iterations=1)
    cv.imwrite(save_path, cv.normalize(cv.medianBlur(cl, 3), None, 0, 255, cv.NORM_MINMAX))

def ejecutar_pipeline_completo(raw_path, out_path):
    """Ejecuta el pipeline completo de preprocesamiento sobre raw_path y guarda en out_path."""

    root_output = os.path.dirname(out_path.rstrip(os.sep)) 
    if not root_output: root_output = "output"

    print(f"--- Procesando Pipeline: {raw_path} ---")
    p_roi = os.path.join(root_output, "roi")
    p_equ = os.path.join(root_output, "equalized")
    p_bil = os.path.join(root_output, "bilateral_filter")
    p_sob = os.path.join(root_output, "sobel")
    p_ref = out_path 

    print("1. Recortando...")
    recorrer_y_procesar(raw_path, p_roi, aplicar_recorte)
    print("2. Ecualizando...")
    recorrer_y_procesar(p_roi, p_equ, aplicar_ecualizacion)
    print("3. Filtrando...")
    recorrer_y_procesar(p_equ, p_bil, aplicar_bilateral)
    print("4. Detectando bordes...")
    recorrer_y_procesar(p_bil, p_sob, aplicar_sobel)
    print("5. Refinando (Salida final)...")
    recorrer_y_procesar(p_sob, p_ref, aplicar_refinado)

# UTILIDADES DE DIRECTORIOS
def obtener_usuarios_validos(out_path):
    """
    Lista las carpetas de usuario dentro de out_path (ej: output/refinadas).
    Busca directamente las imágenes dentro de la carpeta del usuario.
    """

    validos = []
    if not os.path.exists(out_path): return []

    posibles = sorted([d for d in os.listdir(out_path) if os.path.isdir(os.path.join(out_path, d))])
    
    for user in posibles:
        if user.lower() == "test": continue 
        user_path = os.path.join(out_path, user)
        tiene_imgs = any(f.lower().endswith('.png') for f in os.listdir(user_path))
        if tiene_imgs:
            validos.append(user)
    return validos

# SIFT Y MÉTRICAS
def cargar_dataset(out_path, sift):
    """Carga el dataset desde out_path, extrayendo keypoints y descriptores SIFT."""

    dataset = []
    users = obtener_usuarios_validos(out_path)
    
    print(f"Cargando dataset de entrenamiento ({len(users)} usuarios válidos)...")
    for user in users:
        folder = os.path.join(out_path, user)
        
        files = sorted([x for x in os.listdir(folder) if x.lower().endswith(".png")])
        for f in files:
            img = cv.imread(os.path.join(folder, f), cv.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv.bitwise_not(img) 
                kp, des = sift.detectAndCompute(img, None)
                if des is not None and len(kp) > 0:
                    dataset.append({'id': user, 'filename': f, 'kp': kp, 'des': des, 'num_kp': len(kp)})
    return dataset

def calcular_similitud_raw(bf, kp_a, des_a, kp_b, des_b):
    if des_a is None or des_b is None or len(kp_a) == 0 or len(kp_b) == 0: return 0.0
    try:
        matches = bf.knnMatch(des_a, des_b, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 4: return 0.0
        src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
        if mask is not None:
            inliers = int(mask.sum())
            return (inliers / len(kp_a)) * 100.0
    except Exception:
        pass
    return 0.0

# GESTIÓN DE PARÁMETROS
def _evaluar_params(params, paths_imgs):
    """Evalúa una configuración de parámetros SIFT mediante EER en un conjunto de imágenes de muestra."""

    if params['nfeatures'] == 0:
        print("\n   [INFO] nfeatures=0: Se usarán TODAS las características.")
    sift = cv.SIFT_create(**params)
    bf = cv.BFMatcher()
    
    data = []
    for p in paths_imgs:
        img = cv.imread(p, cv.IMREAD_GRAYSCALE)
        if img is None: continue
        kp, des = sift.detectAndCompute(cv.bitwise_not(img), None)
        if des is not None and len(kp) > 0:
            user_id = p.split(os.sep)[-2]
            data.append({'kp': kp, 'des': des, 'id': user_id, 'num_kp': len(kp)})

    if len(data) < 2: return 100.0
    gen, imp = [], []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            score = calcular_similitud_raw(bf, data[i]['kp'], data[i]['des'], data[j]['kp'], data[j]['des'])
            if data[i]['id'] == data[j]['id']: gen.append(score)
            else: imp.append(score)

    if not gen or not imp: return 100.0
    thresh = (np.mean(gen) + np.mean(imp)) / 2
    frr = np.mean(np.array(gen) < thresh)
    far = np.mean(np.array(imp) >= thresh)
    return (frr + far) / 2

def gestionar_optimizacion_sift(out_path, params_file):
    """Gestiona la optimización de hiperparámetros SIFT mediante Grid Search o carga desde JSON."""

    if os.path.exists(params_file):
        print(f"Configuración encontrada: {params_file}")
        if input("¿Recalcular hiperparámetros (s/n)? ").strip().lower() != 's':
            with open(params_file, 'r') as f: return json.load(f)

    print("Iniciando Grid Search...")
    users = obtener_usuarios_validos(out_path)
    sample_imgs = []
    
    for u in users[:3]:
        d = os.path.join(out_path, u)
        sample_imgs.extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.png')][:2])

    param_grid = {
        'nfeatures': [0, 2000, 5000],
        'contrastThreshold': [0.03, 0.04],
        'edgeThreshold': [10],
        'sigma': [1.6]
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_eer = 100.0
    mejor_config = {'nfeatures': 0}

    print(f"Probando {len(combinations)} combinaciones...")
    for i, params in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Conf: {params} ... ", end="")
        eer = _evaluar_params(params, sample_imgs)
        print(f"EER: {eer*100:.2f}%")
        if eer < best_eer:
            best_eer = eer
            mejor_config = params

    print(f"\n>>> MEJOR CONFIGURACIÓN: {mejor_config}")
    with open(params_file, 'w') as f: json.dump(mejor_config, f)
    return mejor_config

def generar_scores_entrenamiento(out_path, params):
    """Genera los scores de comparación entre todas las imágenes del dataset de entrenamiento."""

    sift = cv.SIFT_create(**params)
    bf = cv.BFMatcher()
    
    dataset = cargar_dataset(out_path, sift)
    if len(dataset) < 2: return [], []

    genuinos, impostores = [], []
    total = (len(dataset) * (len(dataset) - 1)) // 2
    print(f"Ejecutando {total} comparaciones...", end="")
    
    for i, (img_A, img_B) in enumerate(itertools.combinations(dataset, 2)):
        score = calcular_similitud_raw(bf, img_A['kp'], img_A['des'], img_B['kp'], img_B['des'])
        if img_A['id'] == img_B['id']: genuinos.append(score)
        else: impostores.append(score)
        if i % 500 == 0: print(".", end="", flush=True)
            
    print(f" Hecho.\nGenuinos: {len(genuinos)} | Impostores: {len(impostores)}")
    return np.array(genuinos), np.array(impostores)

def calcular_umbral_optimo(genuinos, impostores):
    """Calcula el umbral óptimo que minimiza la diferencia entre FAR y FRR."""

    if len(genuinos) == 0 or len(impostores) == 0: return 0.5
    max_score = max(np.max(genuinos), np.max(impostores))
    umbrales = np.linspace(0, max_score, 1000)
    t = umbrales[:, np.newaxis]
    frr = np.mean(genuinos < t, axis=1)
    far = np.mean(impostores >= t, axis=1)
    return umbrales[np.argmin(np.abs(frr - far))]

# GRÁFICAS
def _plot_gauss_aux(ax, x_range, data, color, label, fill_cond, fill_label):
    """Auxiliar para graficar distribución gaussiana y áreas bajo la curva."""

    if len(data) == 0: return
    mu, std = norm.fit(data)
    std = max(std, 1e-6)
    y = norm.pdf(x_range, mu, std)
    ax.plot(x_range, y, color=color, lw=2, label=label)
    ax.fill_between(x_range, y, where=fill_cond, color=color, alpha=0.3, label=fill_label)

def dibujar_curvas_entrenamiento(genuinos, impostores, umbral):
    """Dibuja las curvas de distribución gaussiana y DET con el umbral óptimo."""

    if len(genuinos) == 0 or len(impostores) == 0: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    max_val = max(np.max(genuinos), np.max(impostores))
    x_range = np.linspace(0, max_val * 1.2, 500)
    
    _plot_gauss_aux(ax1, x_range, genuinos, 'green', 'Genuinos', x_range < umbral, 'FRR')
    _plot_gauss_aux(ax1, x_range, impostores, 'red', 'Impostores', x_range >= umbral, 'FAR')
    ax1.axvline(umbral, color='k', linestyle='--', label=f'Umbral: {umbral:.2f}')
    ax1.set_title('Distribución de Scores'); ax1.legend(); ax1.grid(True, alpha=0.3)

    t = np.linspace(0, max_val * 1.2, 1000)[:, np.newaxis]
    far_list = np.mean(impostores >= t, axis=1)
    frr_list = np.mean(genuinos < t, axis=1)
    ax2.plot(far_list, frr_list, 'b-', lw=3, label='Curva DET')
    f_op = np.mean(impostores >= umbral)
    r_op = np.mean(genuinos < umbral)
    ax2.plot(f_op, r_op, 'ro', markersize=10, label=f'Op (U={umbral:.2f})')
    ax2.set_title('Curva DET'); ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.grid(True, which='both', alpha=0.5); ax2.legend()
    plt.tight_layout(); plt.show()

# TEST FINAL
def test_final_contrastar(test_path, out_path_train, params, umbral):
    """Realiza el test final de verificación con un usuario seleccionado."""

    sift = cv.SIFT_create(**params)
    bf = cv.BFMatcher()

    users = obtener_usuarios_validos(out_path_train)
    
    if not users:
        print("ERROR: No se encontraron usuarios válidos. Revisa el preprocesamiento.")
        return

    print("\n--- SELECCIÓN DE USUARIO ---")
    for i, u in enumerate(users): print(f"[{i}] {u}")
    
    try:
        sel = int(input(f"Elige ID (0-{len(users)-1}): ").strip())
        ref_user = users[sel]
    except (ValueError, IndexError):
        ref_user = users[0]

    print(f"Usuario: {ref_user} | Umbral: > {umbral:.4f}")

    ref_dir = os.path.join(out_path_train, ref_user)
    refs = []
    
    if os.path.exists(ref_dir):
        files = [f for f in os.listdir(ref_dir) if f.lower().endswith('.png')]
        for f in files:
            img = cv.imread(os.path.join(ref_dir, f), cv.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv.bitwise_not(img)
            kp, des = sift.detectAndCompute(img, None)
            if des is not None and len(kp) > 0:
                refs.append((kp, des))

    if not refs:
        print("¡Error! No hay referencias válidas para este usuario.")
        return

    if not os.path.exists(test_path): 
        print(f"Error: No existe {test_path}")
        return
        
    print(f"\n{'FICHERO TEST':<30} | {'SCORE':<8} | {'DECISIÓN':<10}")
    print("-" * 55)

    test_files = sorted([f for f in os.listdir(test_path) if f.lower().endswith('.png')])

    for f in test_files:
        img = cv.imread(os.path.join(test_path, f), cv.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv.bitwise_not(img)

        kp_t, des_t = sift.detectAndCompute(img, None)
        if des_t is None: continue
        
        mejor_score = 0.0
        for i, (kp_r, des_r) in enumerate(refs):
            score = calcular_similitud_raw(bf, kp_t, des_t, kp_r, des_r)
            if score > mejor_score: mejor_score = score
        
        res = "ACEPTADO" if mejor_score > umbral else "RECHAZADO"
        print(f"{f:<30} | {mejor_score:<8.2f} | {res:<10}")