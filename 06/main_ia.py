import os
import torch
from utils_ia import (
    SiameseCNN,
    entrenar_siamese,
    comparar_huellas_ia,
    dibujar_curvas_error_ia,
    procesar_y_comparar_test_ia
)

def main():
    OUT_PATH = "06/output"
    TEST_PATH = "06/output/test/sobel"

    if not os.path.isdir(OUT_PATH):
        raise ValueError(f"La ruta {OUT_PATH} no existe. Verifica las carpetas.")

    print("\n===== CREANDO MODELO SIAMESA =====")
    modelo = SiameseCNN(embedding_dim=128)

    entrenar_siamese(
        modelo=modelo,
        out_path=OUT_PATH,
        epochs=10,
        lr=1e-4
    )

    genuinos, impostores = comparar_huellas_ia(
        modelo=modelo,
        out_path=OUT_PATH,
        umbral=0.15
    )

    print("Generando gráficas de error...")
    dibujar_curvas_error_ia(genuinos, impostores, umbral=0.35)

    procesar_y_comparar_test_ia(
        modelo=modelo,
        test_path=TEST_PATH,
        db_processed_path=OUT_PATH,
        umbral=0.15
    )

if __name__ == "__main__":
    main()