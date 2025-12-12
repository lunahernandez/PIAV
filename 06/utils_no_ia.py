import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import norm

# --- PREPROCESAMIENTO ---
# (Las funciones de preprocesamiento se mantienen igual, las omito para ahorrar espacio)
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
    img = cv.imread(os.path.join(src, name))
    if img is not None:
        cv.imwrite(os.path.join(dst, name), cv.equalizeHist(cv.cvtColor(img, cv.COLOR_BGR2GRAY)))

def fase_bilateral(src, dst, name):
    img = cv.imread(os.path.join(src, name))
    if img is not None:
        cv.imwrite(os.path.join(dst, name), cv.bilateralFilter(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 10, 10, 10))

def fase_sobel(src, dst, name):
    img = cv.imread(os.path.join(src, name))
    if img is not None:
        g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sx = cv.Sobel(g, cv.CV_64F, 1, 0, ksize=3)
        sy = cv.Sobel(g, cv.CV_64F, 0, 1, ksize=3)
        m = cv.magnitude(sx, sy)
        cv.imwrite(os.path.join(dst, name), cv.normalize(m, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8))

def fase_refinar(src, dst, name):
    img = cv.imread(os.path.join(src, name), cv.IMREAD_GRAYSCALE)
    if img is not None:
        h, w = img.shape
        m = int(min(h, w) * 0.03)
        crop = img[m:h-m, m:w-m]
        cl = cv.morphologyEx(crop, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=1)
        cv.imwrite(os.path.join(dst, name), cv.normalize(cv.medianBlur(cl, 3), None, 0, 255, cv.NORM_MINMAX))

def ejecutar_pipeline_completo(raw_path, out_path):
    print(f"Procesando: {raw_path} -> {out_path}")
    recortar_roi(raw_path, out_path)
    procesar_fase(fase_ecualizar, "roi", "equalized", raw_path, out_path)
    procesar_fase(fase_bilateral, "equalized", "bilateral_filter", raw_path, out_path)
    procesar_fase(fase_sobel, "bilateral_filter", "sobel", raw_path, out_path)
    procesar_fase(fase_refinar, "sobel", "refinadas", raw_path, out_path)

# --- SIFT Y MÉTRICAS ---

def cargar_dataset(out_path, sift):
    dataset = []
    users = sorted([u for u in os.listdir(out_path) 
                   if os.path.isdir(os.path.join(out_path, u)) and u.lower() != "test"])
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
    if des_a is None or des_b is None: return 0, 0.0
    try:
        matches = bf.knnMatch(des_a, des_b, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) >= 4:
            src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
            if mask is not None:
                return int(mask.sum()), (int(mask.sum()) / len(good)) * 100.0
    except: pass
    return 0, 0.0

def evaluar_configuracion_sift(out_path, params):
    sift = cv.SIFT_create(**params)
    bf = cv.BFMatcher()
    dataset = cargar_dataset(out_path, sift)
    gen, imp = [], []
    step = 1 if len(dataset) < 50 else 2
    for i in range(0, len(dataset), step):
        for j in range(i + 1, len(dataset), step):
            img_A, img_B = dataset[i], dataset[j]
            _, ratio = calcular_similitud_raw(bf, img_A['kp'], img_A['des'], img_B['kp'], img_B['des'])
            if img_A['id'] == img_B['id']: gen.append(ratio)
            else: imp.append(ratio)
    return gen, imp

def gestionar_optimizacion_sift(out_path, params_file):
    mejor_config = {}
    buscar = True
    if os.path.exists(params_file):
        print(f"Archivo de configuración encontrado: {params_file}")
        if input("¿Recalcular hiperparámetros (s/n)? ").lower().strip() != 's':
            buscar = False
            with open(params_file, 'r') as f: mejor_config = json.load(f)
            print(f"Configuración cargada: {mejor_config}")
    if buscar:
        print("Iniciando búsqueda de mejores parámetros...")
        param_grid = {'nfeatures': [1000, 2000], 'contrastThreshold': [0.01, 0.03], 'edgeThreshold': [10, 15], 'sigma': [1.4, 1.6]}
        mejor_eer = 100.0
        import itertools
        keys, values = zip(*param_grid.items())
        total_combs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for idx, params in enumerate(total_combs):
            print(f"Probando [{idx+1}/{len(total_combs)}]: {params} ...", end="")
            gen, imp = evaluar_configuracion_sift(out_path, params)
            if not gen or not imp: 
                print(" Sin datos.")
                continue
            umbrales = np.linspace(0, 100, 50)
            diffs = [abs((np.sum(np.array(gen) < u)/len(gen)) - (np.sum(np.array(imp) >= u)/len(imp))) for u in umbrales]
            eer_local = min(diffs) * 100
            print(f" EER aprox: {eer_local:.2f}")
            if eer_local < mejor_eer:
                mejor_eer = eer_local
                mejor_config = params
        print(f"Mejor configuración guardada con EER ~{mejor_eer:.2f}")
        with open(params_file, 'w') as f: json.dump(mejor_config, f)
    return mejor_config

def calcular_umbral_optimo(genuinos, impostores):
    umbrales = np.linspace(0, 100, 500)
    gen_arr, imp_arr = np.array(genuinos), np.array(impostores)
    diffs = []
    for u in umbrales:
        frr = np.sum(gen_arr < u) / len(gen_arr) if len(gen_arr) else 0
        far = np.sum(imp_arr >= u) / len(imp_arr) if len(imp_arr) else 0
        diffs.append(abs(frr*100 - far*100))
    return umbrales[np.argmin(diffs)]

def dibujar_gaussiana(genuinos, impostores, mejor_umbral):
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, 100, 500)
    plt.gca().set_facecolor('#f0f0f0')
    if len(impostores) > 1:
        mu, std = norm.fit(impostores)
        y = norm.pdf(x, mu, std)
        plt.plot(x, y, color='tab:blue', lw=3, label='Impostores')
        plt.fill_between(x, y, 0, where=(x >= mejor_umbral), facecolor='tab:blue', alpha=0.4, hatch='///', label='FAR')
    if len(genuinos) > 1:
        mu, std = norm.fit(genuinos)
        y = norm.pdf(x, mu, std)
        plt.plot(x, y, color='tab:orange', lw=3, label='Genuinos')
        plt.fill_between(x, y, 0, where=(x < mejor_umbral), facecolor='tab:orange', alpha=0.4, hatch='///', label='FRR')
    plt.axvline(x=mejor_umbral, color='k', linestyle='--', label=f'Umbral {mejor_umbral:.1f}%')
    plt.title('1. Zonas de Error (Entrenamiento)')
    plt.xlabel('Score (Similitud %)')
    plt.ylabel('Densidad')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def dibujar_det(genuinos, impostores, mejor_umbral):
    """Dibuja la curva DET. Protegida contra falta de datos."""
    plt.figure(figsize=(8, 6))
    
    # PROTECCIÓN: Si no hay datos (porque en el test solo hay 2 fotos iguales), avisar.
    if len(genuinos) == 0 and len(impostores) == 0:
        plt.text(0.5, 0.5, "NO HAY DATOS SUFICIENTES\nPARA GENERAR CURVA DET", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.title('2. Curva DET (Sin datos)')
        plt.show()
        return

    far_l, frr_l = [], []
    g_arr, i_arr = np.array(genuinos), np.array(impostores)
    
    for u in np.linspace(0, 100, 1000):
        # Evitamos división por cero si alguna lista está vacía
        frr = (np.sum(g_arr < u) / len(g_arr) * 100) if len(g_arr) > 0 else 0
        far = (np.sum(i_arr >= u) / len(i_arr) * 100) if len(i_arr) > 0 else 0
        frr_l.append(frr)
        far_l.append(far)

    plt.plot(far_l, frr_l, 'k-', lw=3, label='Rendimiento del Test')
    plt.plot([0, 100], [0, 100], 'k--', alpha=0.2)
    
    # Punto operativo actual
    f_op = (np.sum(i_arr >= mejor_umbral) / len(i_arr) * 100) if len(i_arr) > 0 else 0
    r_op = (np.sum(g_arr < mejor_umbral) / len(g_arr) * 100) if len(g_arr) > 0 else 0
    
    plt.plot(f_op, r_op, 'rx', markersize=14, markeredgewidth=3, label=f'Resultados (Umb={mejor_umbral:.1f})')
    
    plt.title('2. Curva DET (Resultados del TEST)')
    plt.xlabel('Falsa Aceptación (FAR %)')
    plt.ylabel('Falso Rechazo (FRR %)')
    plt.legend()
    plt.grid(True)
    
    # Zoom automático seguro
    xlim_max = max(10, f_op*4) if f_op > 0 else 100
    ylim_max = max(10, r_op*4) if r_op > 0 else 100
    plt.xlim([0, xlim_max])
    plt.ylim([0, ylim_max])
    plt.tight_layout()
    plt.show()

def test_autenticacion_usuario(test_path, out_path_db, params, umbral):
    """
    Menú interactivo y ejecución del test.
    AHORA RETORNA LAS LISTAS DE SCORES DEL TEST (GENUINOS E IMPOSTORES).
    """
    users = sorted([d for d in os.listdir(out_path_db) if os.path.isdir(os.path.join(out_path_db, d)) and d.lower() != "test"])
    if not users:
        print("No hay usuarios en la BD.")
        return [], []

    print("\n--- SELECCIÓN DE USUARIO REFERENCIA ---")
    for i, u in enumerate(users): print(f"[{i}] {u}")
    
    sel = input(f"Elige usuario (Enter={users[0]}): ").strip()
    ref_user = users[0]
    
    if sel.isdigit() and int(sel) < len(users): ref_user = users[int(sel)]
    elif sel in users: ref_user = sel
    
    print(f"\nVerificando contra: {ref_user} (Umbral: {umbral:.2f})")
    
    sift = cv.SIFT_create(**params)
    bf = cv.BFMatcher()
    ref_dir = os.path.join(out_path_db, ref_user, "refinadas")
    refs = []
    
    if os.path.exists(ref_dir):
        for f in os.listdir(ref_dir):
            img = cv.imread(os.path.join(ref_dir, f), cv.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = sift.detectAndCompute(cv.bitwise_not(img), None)
                if des is not None: refs.append((kp, des))
    
    test_dir = os.path.join(test_path, "refinadas")
    print(f"{'ARCHIVO':<30} | {'SCORE':<6} | {'RESULTADO'}")
    print("-" * 50)
    
    # Listas para guardar los resultados DE ESTE TEST
    test_genuinos = []
    test_impostores = []

    if os.path.exists(test_dir):
        for f in os.listdir(test_dir):
            img = cv.imread(os.path.join(test_dir, f), cv.IMREAD_GRAYSCALE)
            if img is None: continue
            kp, des = sift.detectAndCompute(cv.bitwise_not(img), None)
            
            best = 0.0
            for r_kp, r_des in refs:
                _, ratio = calcular_similitud_raw(bf, kp, des, r_kp, r_des)
                if ratio > best: best = ratio
            
            res = "ACEPTADO" if best >= umbral else "RECHAZADO"
            print(f"{f:<30} | {best:<6.2f} | {res}")

            # CLASIFICACIÓN PARA LA DET:
            # Asumimos que si el nombre del archivo contiene el ID del usuario referencia, es genuino.
            # Si no, asumimos que es un impostor.
            if ref_user in f:
                test_genuinos.append(best)
            else:
                test_impostores.append(best)
                
    return test_genuinos, test_impostores