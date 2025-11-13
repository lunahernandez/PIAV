# Hipótesis no IA
## Descripción 
Comparar las huellas digitales de diferentes imágenes del mismo usuario para evaluar la consistencia y precisión del sistema de reconocimiento de huellas digitales. Se utilizarán imágenes de huellas digitales capturadas en diferentes condiciones (ángulos, presión, humedad) para verificar si el sistema puede identificar correctamente al mismo usuario. Se usarán diferentes preprocesamientos y parámetros del algoritmo SIFT para optimizar la comparación.

## Código
### Utilidades
En el archivo `utils.py`, se añade toda la parte de utilidades necesarias para llevar a cabo la comparación de huellas digitales del mismo usuario y el preprocesamiento de las imágenes. Se va a explicar el uso de cada función añadida según el orden en el que aparecen en el código.

**recortar_roi** : Esta función permite recortar una región de interés (ROI) de una imagen dada. Se utiliza para aislar la parte relevante de la huella digital antes de procesarla.

**mejor_roi_por_negros** : Esta función selecciona la mejor región de interés (ROI) basada en la cantidad de píxeles negros presentes, lo que indica la calidad y relevancia de la región para el análisis, ya que se considera que son las regiones donde aparecen más píxeles negros. Devuelve una región óptima.

**ecualizar_histograma** : Esta función es un preprocesa que lo que hace es mejorar el contraste resaltantes de las crestas y sus valles, con el objetivo de tener imágenes uniformes y más fáciles de comparar con SIFT. Reparte las zonas oscuras y claras.

**aplicar_filtro_bilateral** : Esta función aplica un filtro bilateral a la imagen para reducir el ruido mientras se preservan los bordes, lo que es útil para mantener los detalles importantes de las crestas y valles en la huella digital durante el preprocesamiento.

**realzar_crestas** : Esta función realza las crestas de la huella digital para mejorar la visibilidad y facilitar la comparación, eso lo hace usando Sovel con el objetivo de marcarc con mayor fuerza los bordes verticales de la imagen, pero en la práctica esto puede dificultar el trabajo de SIFT.

A partir de que empezamos a usar SIFT al comparar las huellass nos dimos cuenta que las imágenes bimarizadas y con poca textura no son adecuadas para SIFT, y nos dimos cuentas que habría 2 caminos a seguir.

1. Cambiar la estructura del preprocesado para que las imágenes tengan más textura y SIFT pueda trabajar mejor.
2. Cambiar el SIFT por un detector de minucias, utilizando `kernels` para detectar comparativas entre las crestas y valles.

**refinar_crestas**  : Esta función refina las crestas aplicando un procesado morfológico para mejorar la definición de las crestas y valles en la huella digital,, también tiene un proceso de recorte de borde, y apertura morfológica, filtrado de mediana y normalización. Aunque mejora la apariencia visual del resultado, estas imágenes no son adeciadas para SIFT sino más bien para detectores de minucias.

- Morfológico en apertura: elimina pequeñas imperfecciones en las crestas y valles.

**comparar_huellas_mismo_usuario** : Esta función se encarga de comparar dos huellas del mismo usuario utilizando el algoritmo SIFT. Además, se incorpora RANSAC, que mejora la robustez del proceso al eliminar coincidencias erróneas mediante una validación geométrica. Gracias a esto, solo se consideran válidos los puntos que mantienen una correspondencia coherente entre ambas imágenes.
Si el número de coincidencias válidas (inliers) supera un umbral definido, se considera que las huellas pertenecen al mismo usuario.

También se aplica el Lowe Ratio, una técnica utilizada para filtrar coincidencias débiles o ambiguas entre descriptores SIFT. Este criterio ayuda a quedarse únicamente con los matches más confiables, incrementando la precisión de la comparación.

## Ejecución
Para ejecutar se utiliza el `main.py` con todas las funciones necesarias para llevar a cabo la comparación de huellas digitales del mismo usuario, utilizando las diferentes funciones añadidas en `utils.py`. En la carpeta `output` se guardan las imagenes filtradas según cada preprocesado y luego por ventana de OpenCV y por consola se ve todo lo relacionado con las comparaciones.

