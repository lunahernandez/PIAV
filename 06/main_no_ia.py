import numpy as np
from utils_no_ia import (
    ejecutar_pipeline_completo,
    gestionar_optimizacion_sift,
    generar_scores_entrenamiento,
    calcular_umbral_optimo,
    dibujar_curvas_entrenamiento,
    test_final_contrastar
)

def main():
    DB_PATH = "data"          
    OUT_PATH = "output"       
    TEST_PATH = "test"        
    OUT_TEST_PATH = "output/test"
    PARAMS_FILE = "mejor_config_sift.json"

    print("\n--- FASE 1: PREPROCESAMIENTO ---")
    ejecutar_pipeline_completo(DB_PATH, OUT_PATH)
    ejecutar_pipeline_completo(TEST_PATH, OUT_TEST_PATH)

    print("\n--- FASE 2: CONFIGURACIÓN E HIPERPARÁMETROS ---")
    config_sift = gestionar_optimizacion_sift(OUT_PATH, PARAMS_FILE)

    print("\n--- FASE 3: GENERACIÓN DE MÉTRICAS DE ENTRENAMIENTO ---")
    genuinos, impostores = generar_scores_entrenamiento(OUT_PATH, config_sift)
    
    mejor_umbral = calcular_umbral_optimo(genuinos, impostores)
    print(f"\n>>> UMBRAL ÓPTIMO (CALCULADO): {mejor_umbral:.4f}% <<<")
    
    dibujar_curvas_entrenamiento(genuinos, impostores, mejor_umbral)

    print("\n--- FASE 4: MODO TEST (VERIFICACIÓN FINAL) ---")
    test_final_contrastar(OUT_TEST_PATH, OUT_PATH, config_sift, mejor_umbral)

if __name__ == "__main__":
    main()