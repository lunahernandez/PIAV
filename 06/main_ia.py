import os
from utils_ia import (
    SiameseCNN,
    entrenar_siamese,
    comparar_mismo_usuario_ia,
    comparar_entre_usuarios_ia
)

def main():

    out_path = "06/output"

    if not os.path.isdir(out_path):
        raise ValueError(f"La ruta {out_path} no existe. Coloca OUT en el mismo nivel que el main.")

    print("\n===== CREANDO MODELO SIAMESA =====")
    print("Paso 1. Crear el modelo siamesa")
    modelo = SiameseCNN(embedding_dim=128)

    print("Paso 2. Entrenar el modelo siamesa")
    entrenar_siamese(
        modelo=modelo,
        out_path=out_path,
        epochs=10,
        lr=1e-4
    )

    print("Paso 3. Comparar imágenes del mismo usuario")
    comparar_mismo_usuario_ia(
        modelo=modelo,
        out_path=out_path,
        umbral=0.35
    )

    print("Paso 4. Comparar entre usuarios")
    comparar_entre_usuarios_ia(
        modelo=modelo,
        out_path=out_path,
        ref_user="crd_0811f",
        umbral=0.05
    )

if __name__ == "__main__":
    main()
