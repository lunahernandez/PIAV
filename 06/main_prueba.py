from utils_no_ia import (
    ejecutar_pipeline_completo
    
)
import os


def main():
    # --- CONFIGURACIÓN ---
    DB_PATH = "data"
    OUT_PATH = "output"
    TEST_PATH = "test"
    OUT_TEST_PATH = "output/test"
    PARAMS_FILE = "mejor_config_sift.json"

    # 1. PREPROCESAMIENTO
    print("\n--- FASE 1: PREPROCESAMIENTO ---")
    ejecutar_pipeline_completo(DB_PATH, OUT_PATH)

if __name__ == "__main__":
    main()