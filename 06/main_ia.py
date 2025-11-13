import os
import torch

from utils_ia import (
    SiameseCNN,
    entrenar_siamese,
    comparar_mismo_usuario_ia,
    comparar_entre_usuarios_ia
)

# ============================================================
# MAIN
# ============================================================

def main():

    # ------------------------------
    # Rutas
    # ------------------------------
    out_path = "06/output"    # carpeta donde tienes las huellas refinadas por usuario

    if not os.path.isdir(out_path):
        raise ValueError(f"La ruta {out_path} no existe. Coloca OUT en el mismo nivel que el main.")

    # ------------------------------
    # Crear el modelo
    # ------------------------------
    print("\n===== CREANDO MODELO SIAMESA =====")
    modelo = SiameseCNN(embedding_dim=128)

    # ------------------------------
    # ENTRENAR EL MODELO
    # ------------------------------
    print("\n===== INICIANDO ENTRENAMIENTO =====")
    entrenar_siamese(
        modelo=modelo,
        out_path=out_path,
        epochs=10,        # puedes subirlo
        lr=1e-4
    )

    # ------------------------------
    # COMPARAR HUELLAS DEL MISMO USUARIO
    # ------------------------------
    print("\n===== COMPARACIÓN MISMO USUARIO =====")
    comparar_mismo_usuario_ia(
        modelo=modelo,
        out_path=out_path,
        umbral=0.35      # ajustable tras entrenar
    )

    # ------------------------------
    # COMPARAR HUELLAS ENTRE USUARIOS
    # ------------------------------
    print("\n===== COMPARACIÓN ENTRE USUARIOS =====")
    comparar_entre_usuarios_ia(
        modelo=modelo,
        out_path=out_path,
        umbral=0.05
    )


# Ejecutar
if __name__ == "__main__":
    main()
