# utils_ia.py
import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Cargar imagen
def cargar_imagen(img_path, size=224):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"No se pudo cargar: {img_path}")

    img = cv.resize(img, (size, size))
    tensor = torch.tensor(img, dtype=torch.float32) / 255.0

    return tensor.unsqueeze(0)

# Modelo Siamesa
class SiameseCNN(nn.Module):

    def __init__(self, embedding_dim=128):
        super().__init__()
        
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
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
    
    def forward_once(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def forward(self, img1, img2):
        return self.forward_once(img1), self.forward_once(img2)

# Dataset de pares de huellas
class FingerprintPairsDataset(Dataset):

    def __init__(self, out_path):
        self.pairs = []
        self.labels = []
        
        usuarios = sorted(os.listdir(out_path))
        
        # Pares positivos (mismo usuario)
        for user in usuarios:
            folder = os.path.join(out_path, user, "refinadas")
            if not os.path.isdir(folder):
                continue
            
            imgs = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
            if len(imgs) >= 2:
                self.pairs.append((os.path.join(folder, imgs[0]), os.path.join(folder, imgs[1])))
                self.labels.append(1)
        
        # Pares negativos (usuarios diferentes)
        for i, u1 in enumerate(usuarios):
            for u2 in usuarios[i+1:]:
                f1 = os.path.join(out_path, u1, "refinadas")
                f2 = os.path.join(out_path, u2, "refinadas")

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

# Distancia coseno
def distancia_coseno(e1, e2):
    sim = F.cosine_similarity(e1, e2)
    return 1 - sim.item()


# Comparación con huellas del mismo usuario
def comparar_mismo_usuario_ia(modelo, out_path, umbral=0.35):
    
    print("============================================================")
    print("COMPARACIÓN MISMO USUARIO (IA)")
    print("============================================================")
    
    usuarios = sorted(os.listdir(out_path))
    
    for user in usuarios:
        folder = os.path.join(out_path, user, "refinadas")
        if not os.path.isdir(folder):
            continue
        
        imgs = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
        if len(imgs) < 2:
            continue
        
        t1 = cargar_imagen(os.path.join(folder, imgs[0]))
        t2 = cargar_imagen(os.path.join(folder, imgs[1]))

        # Añadimos dimensión batch -> [1, 1, H, W]
        t1 = t1.unsqueeze(0)
        t2 = t2.unsqueeze(0)

        with torch.no_grad():
            e1, e2 = modelo(t1, t2)
        
        dist = distancia_coseno(e1, e2)
        match = "MATCH" if dist < umbral else "NO MATCH"
        
        print(f"\nUsuario {user}: Distancia={dist:.4f} - {match}")

# Comparación entre usuarios
def comparar_entre_usuarios_ia(modelo, out_path, ref_user, umbral=0.35):

    usuarios = sorted(os.listdir(out_path))

    if ref_user not in usuarios:
        print(f"ERROR: El usuario '{ref_user}' no existe en {out_path}")
        print("Usuarios disponibles:", usuarios)
        return

    ref_folder = os.path.join(out_path, ref_user, "refinadas")
    ref_imgs = sorted([f for f in os.listdir(ref_folder) if f.endswith(".png")])

    if not ref_imgs:
        print(f"No hay imágenes en {ref_folder}")
        return

    ref_tensor = cargar_imagen(os.path.join(ref_folder, ref_imgs[0]))
    ref_tensor = ref_tensor.unsqueeze(0)

    with torch.no_grad():
        ref_emb = modelo.forward_once(ref_tensor)

    print(f"\nUsuario de referencia: {ref_user}\n")
    print("Comparaciones con otros usuarios:")

    # Comparar con otros usuarios
    for user in usuarios:
        if user == ref_user:
            continue

        folder = os.path.join(out_path, user, "refinadas")
        imgs = sorted([f for f in os.listdir(folder) if f.endswith(".png")])

        if not imgs:
            continue

        t = cargar_imagen(os.path.join(folder, imgs[0]))
        t = t.unsqueeze(0)

        with torch.no_grad():
            emb = modelo.forward_once(t)

        dist = distancia_coseno(ref_emb, emb)
        match = "MATCH" if dist < umbral else "NO MATCH"

        print(f"{ref_user} vs {user}: Distancia={dist:.4f} - {match}")
