from utils import (
    recortar_roi,
    ecualizar_histograma,
    aplicar_filtro_bilateral,
    realzar_crestas,
    refinar_crestas,
)

def main():
    DB_PATH = r"06/data"
    OUT_PATH = "06/output"
    print(20*"=")
    print("PREPROCESADO")
    print(20*"=")

    print("Paso 1. Buscar y segmentar la ROI")
    recortar_roi(DB_PATH, OUT_PATH, ventana=500, paso=15)

    print("Paso 2. Ecualizar y normalizar el histograma")
    ecualizar_histograma(DB_PATH, OUT_PATH)

    print("Paso 3. Aplicar un filtro bilateral")
    aplicar_filtro_bilateral(DB_PATH, OUT_PATH)

    print("Paso 4. Aplicar un realce de crestas")
    realzar_crestas(DB_PATH, OUT_PATH)

    print("Paso 5. Refinar las crestas eliminando bordes falsos")
    refinar_crestas(DB_PATH, OUT_PATH)

    print(20*"-")
    print("Preprocesado finalizado")
    print(20*"-")

if __name__ == "__main__":
    main()
