import numpy as np
import json
from utils_no_ia import (
    ejecutar_pipeline_completo,
    gestionar_optimizacion_sift,
    generar_scores_entrenamiento,
    calcular_umbral_optimo,
    dibujar_curvas_entrenamiento,
    test_final_contrastar
)

def main():
    # --- RUTAS ---
    DB_PATH = "data"          
    OUT_PATH = "output/refinadas"
    OUT_TEST_PATH = "output/test/refinadas"       
    TEST_PATH = "test"        
    PARAMS_FILE = "mejor_config_sift.json"

    # PREPROCESAMIENTO
    print("\n=== FASE 1: PREPROCESAMIENTO ===")
    ejecutar_pipeline_completo(DB_PATH, OUT_PATH)
    ejecutar_pipeline_completo(TEST_PATH, OUT_TEST_PATH)

    # CONFIGURACIÓN
    print("\n=== FASE 2: CONFIGURACIÓN SIFT ===")
    config_sift = gestionar_optimizacion_sift(OUT_PATH, PARAMS_FILE)

    # ENTRENAMIENTO REAL Y GRÁFICAS
    print("\n=== FASE 3: GENERACIÓN DE DATOS Y GRÁFICAS ===")
    genuinos, impostores = generar_scores_entrenamiento(OUT_PATH, config_sift)
    
    # Calcular umbral con los datos reales
    mejor_umbral = calcular_umbral_optimo(genuinos, impostores)
    print(f"\n>>> UMBRAL ÓPTIMO (CALCULADO): {mejor_umbral:.4f}% <<<")
    
    # Dibujar
    dibujar_curvas_entrenamiento(genuinos, impostores, mejor_umbral)

    # TEST FINAL
    print("\n=== FASE 4: MODO TEST (VERIFICACIÓN) ===")
    test_final_contrastar(OUT_TEST_PATH, OUT_PATH, config_sift, mejor_umbral)

if __name__ == "__main__":
    main()