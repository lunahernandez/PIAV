import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def cargar_imagen(img_path, size=224):
    """ Carga una imagen en escala de grises y la convierte en tensor normalizado"""

    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"No se pudo cargar: {img_path}")

    img = cv.resize(img, (size, size))
    tensor = torch.tensor(img, dtype=torch.float32) / 255.0

    return tensor.unsqueeze(0) # Añade dimensión canal: [1, H, W]

# Modelo Siamesa
class SiameseCNN(nn.Module):
    """ Red Siamesa para extracción de características de huellas dactilares."""

    def __init__(self, embedding_dim=128):
        super().__init__()
        
        # Bloque Convolucional (Extrae características visuales)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Bloque Fully Connected (Genera el vector de identidad)
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
    
    def forward_once(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # --- CAMBIO CLAVE: Normalización L2 ---
        # Esto obliga a que todos los vectores vivan en una hiperesfera.
        # Hace que la Distancia Euclídea y Coseno sean equivalentes.
        return F.normalize(x, p=2, dim=1)
    
    def forward(self, img1, img2):
        return self.forward_once(img1), self.forward_once(img2)

# Dataset de pares de huellas
class FingerprintPairsDataset(Dataset):
    """ Dataset de pares de huellas para entrenamiento del modelo siamesa."""

    def __init__(self, out_path):
        self.pairs = []
        self.labels = []
        
        usuarios = sorted(os.listdir(out_path))
        
        # Pares positivos (mismo usuario)
        for user in usuarios:
            folder = os.path.join(out_path, user, "sobel")
            if not os.path.isdir(folder):
                continue
            
            imgs = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
            if len(imgs) >= 2:
                self.pairs.append((os.path.join(folder, imgs[0]), os.path.join(folder, imgs[1])))
                self.labels.append(1)
        
        # Pares negativos (usuarios diferentes)
        for i, u1 in enumerate(usuarios):
            for u2 in usuarios[i+1:]:
                f1 = os.path.join(out_path, u1, "sobel")
                f2 = os.path.join(out_path, u2, "sobel")

                if not os.path.isdir(f1) or not os.path.isdir(f2):
                    continue

                imgs1 = sorted([f for f in os.listdir(f1) if f.endswith(".png")])
                imgs2 = sorted([f for f in os.listdir(f2) if f.endswith(".png")])

                if imgs1 and imgs2:
                    self.pairs.append((os.path.join(f1, imgs1[0]), os.path.join(f2, imgs2[0])))
                    self.labels.append(0)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1, img2 = self.pairs[idx]
        t1 = cargar_imagen(img1)
        t2 = cargar_imagen(img2)
        return t1, t2, torch.tensor(self.labels[idx], dtype=torch.float32)

# Entrenamiento del modelo siamesa
def entrenar_siamese(modelo, out_path, epochs=5, lr=1e-4):
    """ Entrena el modelo siamesa con pares de huellas. """

    dataset = FingerprintPairsDataset(out_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
    
    def contrastive_loss(e1, e2, y, margin=1.0):
        d = F.pairwise_distance(e1, e2)
        return torch.mean(y * d.pow(2) + (1 - y) * F.relu(margin - d).pow(2))
    
    print("\n===== ENTRENAMIENTO =====")
    
    modelo.train()
    for epoch in range(epochs):
        total_loss = 0
        for t1, t2, y in dataloader:
            e1, e2 = modelo(t1, t2)
            loss = contrastive_loss(e1, e2, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
    
    print("Entrenamiento terminado.\n")
    modelo.eval()

# FUNCION AUXILIAR: Cálculo de distancia Coseno
def distancia_coseno(e1, e2):
    """Calcula la distancia coseno entre dos embeddings."""

    sim = F.cosine_similarity(e1, e2)
    return 1 - sim.item()

# COMPARADOR : Comparación para datos de la BD
def comparar_huellas_ia(modelo, out_path, umbral=0.35):
    """ Realiza una comparación masiva "todos contra todos" de las huellas en out_path."""

    dataset = []
    usuarios = sorted(os.listdir(out_path))

    for user in usuarios:
        if user.lower() == "test": continue

        folder = os.path.join(out_path, user, "sobel")
        if not os.path.isdir(folder): continue
        
        imgs = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
        for f in imgs:
            path = os.path.join(folder, f)
            try:
                t = cargar_imagen(path).unsqueeze(0) 
                with torch.no_grad():
                    emb = modelo.forward_once(t)
                
                dataset.append({
                    'id': user,
                    'filename': f,
                    'emb': emb,
                    'path': path
                })
            except Exception as e:
                print(f"Error cargando {path}: {e}")

    num_imgs = len(dataset)
    scores_genuinos = []
    scores_impostores = []
    stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    print("\n" + "="*125)
    print(f"{'HUELLA A':<25} | {'HUELLA B':<25} | {'MATCH?':<8} | {'DISTANCIA':<10} | {'RESULTADO':<25}")
    print("="*125)

    for i in range(num_imgs):
        for j in range(i, num_imgs):
            
            img_A = dataset[i]
            img_B = dataset[j]

            dist = distancia_coseno(img_A['emb'], img_B['emb'])
            match_algoritmo = (dist < umbral)

            es_mismo_archivo = (i == j)
            es_misma_persona = (img_A['id'] == img_B['id'])

            if not es_mismo_archivo:
                if es_misma_persona:
                    scores_genuinos.append(dist)
                else:
                    scores_impostores.append(dist)

            etiqueta = ""
            if es_mismo_archivo:
                if match_algoritmo:
                    etiqueta = "TP (Auto-Check)"
                    stats["TP"] += 1
                else:
                    etiqueta = "FN (Error Modelo)" 
                    stats["FN"] += 1

            elif es_misma_persona:
                if match_algoritmo:
                    etiqueta = "TP (Acierto)"
                    stats["TP"] += 1
                else:
                    etiqueta = "FN (No coincidió)"
                    stats["FN"] += 1
            
            else:
                if match_algoritmo:
                    etiqueta = "FP (Error)"
                    stats["FP"] += 1
                else:
                    etiqueta = "TN (Correcto)"
                    stats["TN"] += 1

            print(f"{img_A['filename'][:23]:<25} | {img_B['filename'][:23]:<25} | {str(match_algoritmo):<8} | {dist:.4f}     | {etiqueta}")

    total = sum(stats.values())
    print("="*125)
    print("RESUMEN GLOBAL (IA)")
    print(f"Comparaciones totales: {total}")
    print(f"Aciertos Totales: {stats['TP'] + stats['TN']}")
    print(f"   - Match Correctos (TP): {stats['TP']}")
    print(f"   - Rechazos Correctos (TN): {stats['TN']}")
    print(f"Fallos Totales: {stats['FP'] + stats['FN']}")
    print(f"   - Falsos Positivos (FP): {stats['FP']}")
    print(f"   - Falsos Negativos (FN): {stats['FN']}")
    print("="*125)

    return scores_genuinos, scores_impostores

def dibujar_curvas_error_ia(scores_genuinos, scores_impostores, umbral=0.35):
    """ Dibuja las distribuciones de distancias para genuinos e impostores. """

    plt.figure(figsize=(10, 6))
    x_range = np.linspace(0, 1.5, 500)

    if len(scores_impostores) > 1:
        mu_imp, std_imp = norm.fit(scores_impostores)
        p_imp = norm.pdf(x_range, mu_imp, std_imp)
        plt.plot(x_range, p_imp, 'b-', lw=2, label='Impostores (Diferentes)')
        plt.fill_between(x_range, p_imp, alpha=0.2, color='blue')
    
    if len(scores_genuinos) > 1:
        mu_gen, std_gen = norm.fit(scores_genuinos)
        p_gen = norm.pdf(x_range, mu_gen, std_gen)
        plt.plot(x_range, p_gen, 'r-', lw=2, label='Genuinos (Mismos)')
        plt.fill_between(x_range, p_gen, alpha=0.2, color='red')

    plt.axvline(x=umbral, color='k', linestyle='--', label=f'Umbral ({umbral})')
    plt.title('Distribución de Distancias (Siamese CNN)')
    plt.xlabel('Distancia Coseno (Menor es más similar)')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def procesar_y_comparar_test_ia(modelo, test_path, db_processed_path, umbral=0.35):
    """ Procesa imágenes de test y las compara contra la base de datos procesada. """

    db_embeddings = []
    users = [u for u in os.listdir(db_processed_path) if u.lower() != "test"]

    for user in users:
        folder = os.path.join(db_processed_path, user, "refinadas")
        if not os.path.isdir(folder): continue
        
        for f in os.listdir(folder):
            if f.endswith(".png"):
                path = os.path.join(folder, f)
                try:
                    t = cargar_imagen(path).unsqueeze(0)
                    with torch.no_grad():
                        emb = modelo.forward_once(t)
                    db_embeddings.append({'user': user, 'file': f, 'emb': emb})
                except: pass
    
    test_files = [f for f in os.listdir(test_path) if f.lower().endswith('.png')]
    if not test_files and os.path.isdir(os.path.join(test_path, "refinadas")):
         test_path = os.path.join(test_path, "refinadas")
         test_files = [f for f in os.listdir(test_path) if f.lower().endswith('.png')]

    if not test_files:
        print("[ERROR] No se encontraron imágenes de test.")
        return

    for f in test_files:
        print(f"\nIdentificando: {f}")
        try:
            t_test = cargar_imagen(os.path.join(test_path, f)).unsqueeze(0)
            with torch.no_grad():
                emb_test = modelo.forward_once(t_test)
            
            mejor_dist = 999.0
            mejor_user = "Desconocido"
            match_file = ""

            for db_item in db_embeddings:
                d = distancia_coseno(emb_test, db_item['emb'])
                if d < mejor_dist:
                    mejor_dist = d
                    mejor_user = db_item['user']
                    match_file = db_item['file']
            
            if mejor_dist < umbral:
                print(f"[IDENTIFICADO] {mejor_user}")
                print(f"[Match con {match_file}, Distancia: {mejor_dist:.4f}]")
            else:
                print(f"[NO IDENTIFICADO] (Más cercano: {mejor_user}, Dist: {mejor_dist:.4f})")

        except Exception as e:
            print(f"Error procesando {f}: {e}")