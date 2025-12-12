from utils_no_ia import (
    recortar_roi,
    ecualizar_histograma,
    aplicar_filtro_bilateral,
    realzar_crestas,
    refinar_roi,
    comparar_huellas,
    dibujar_curvas_error,
    procesar_y_comparar_test
)
import os

def main():

    DB_PATH = r"06/data"
    OUT_PATH = "06/output"
    
    TEST_PATH = "06/test"
    OUT_TEST_PATH = "06/output/test"

    print(20*"=")
    print("PREPROCESADO BASE DE DATOS (DATA)")
    print(20*"=")
    print("Paso 1. Buscar y segmentar la ROI")
    recortar_roi(DB_PATH, OUT_PATH, ventana=500, paso=15)  
    print("Paso 2. Ecualizar y normalizar el histograma")
    ecualizar_histograma(DB_PATH, OUT_PATH)
    print("Paso 3. Aplicar un filtro bilateral")
    aplicar_filtro_bilateral(DB_PATH, OUT_PATH)
    print("Paso 4. Aplicar un realce de crestas")
    realzar_crestas(DB_PATH, OUT_PATH)
    print("Paso 5. Refinar la ROI eliminando bordes falsos")
    refinar_roi(DB_PATH, OUT_PATH)

    print(20*"=")
    print("PREPROCESADO IMÁGENES DE TEST")
    print(20*"=")
    print("Paso 1. ROI (Test)")
    recortar_roi(TEST_PATH, OUT_TEST_PATH, ventana=500, paso=15)
    print("Paso 2. Ecualizar (Test)")
    ecualizar_histograma(TEST_PATH, OUT_TEST_PATH)
    print("Paso 3. Bilateral (Test)")
    aplicar_filtro_bilateral(TEST_PATH, OUT_TEST_PATH)
    print("Paso 4. Sobel (Test)")
    realzar_crestas(TEST_PATH, OUT_TEST_PATH)
    print("Paso 5. Refinar (Test)")
    refinar_roi(TEST_PATH, OUT_TEST_PATH)

    genuinos, impostores = comparar_huellas(DB_PATH, OUT_PATH)
    print("Generando gráficas de error (FAR/FRR)...")
    dibujar_curvas_error(genuinos, impostores, umbral=15)

    print(20*"=")
    print("FASE DE TEST: Identificando huellas desconocidas")
    print(20*"=")
    DB_PROCESSED_PATH = OUT_PATH 
    procesar_y_comparar_test(TEST_PATH, DB_PROCESSED_PATH, OUT_TEST_PATH)

if __name__ == "__main__":
    main()