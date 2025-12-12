from utils_no_ia import (
    ejecutar_pipeline_completo,
    evaluar_configuracion_sift,
    gestionar_optimizacion_sift,
    calcular_umbral_optimo,
    dibujar_gaussiana,
    dibujar_det,
    test_autenticacion_usuario
)

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
    ejecutar_pipeline_completo(TEST_PATH, OUT_TEST_PATH)

    # 2. OPTIMIZACIÓN SIFT (ENTRENAMIENTO)
    print("\n--- FASE 2: ENTRENAMIENTO Y PARÁMETROS ---")
    mejor_config = gestionar_optimizacion_sift(OUT_PATH, PARAMS_FILE)
    
    # Calculamos métricas de entrenamiento para decidir el umbral
    print("Calculando métricas de entrenamiento...")
    genuinos_train, impostores_train = evaluar_configuracion_sift(OUT_PATH, mejor_config)
    
    mejor_umbral = calcular_umbral_optimo(genuinos_train, impostores_train)
    
    # Visualización de entrenamiento: Distribución
    print(f"Umbral óptimo calculado: {mejor_umbral:.2f}%")
    dibujar_gaussiana(genuinos_train, impostores_train, mejor_umbral)

    # 3. TEST (EVALUACIÓN REAL)
    print("\n--- FASE 3: MODO TEST Y EVALUACIÓN ---")
    
    # Ejecutamos el test y RECOGEMOS los datos resultantes
    test_gen, test_imp = test_autenticacion_usuario(OUT_TEST_PATH, OUT_PATH, mejor_config, mejor_umbral)

    # Gráfica de Rendimiento (Con los datos del test recién ejecutado)
    print("\nGenerando Curva DET de los resultados del TEST...")
    
    # Aviso si hay pocos datos
    if len(test_gen) + len(test_imp) < 5:
        print("NOTA: La curva DET puede verse incompleta o vacía porque hay muy pocas imágenes en el Test.")
        
    dibujar_det(test_gen, test_imp, mejor_umbral)

if __name__ == "__main__":
    main()