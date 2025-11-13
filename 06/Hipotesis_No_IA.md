# Hipótesis A: sin IA
**Autores**: Luna Yue Hernández Guerra y Jorge Lorenzo Lorenzo
## Preprocesado
A continuación, se exponen los diferentes pasos del preprocesado de las imágenes llevados a cabo. Estos pasos son secuenciales, es decir, cada uno se aplica a los resultados del paso anterior. Los pasos son los siguientes:
1. Buscar y segmentar la ROI
2. Ecualizar y normalizar el histograma
3. Aplicar un filtro bilateral
4. Aplicar un realce de crestas
5. Refinar la ROI

### Buscar y segmentar la ROI
El objetivo del primer paso del preprocesado es obtener el núcleo de la huella. Según el estudio de Simón-Zorita, este paso es realmente importante para conseguir un buen preprocesado de huellas.

Para ello, creamos dos funciones: `mejor_roi()` y `recortar_roi()`.
- `mejor_roi()`: Esta función obtiene la región de interés de una imagen. Dicha región de interés se define como la región de tamaño `ventana`x`ventana` donde más píxeles negros hay. Para ello, se recorre la imagen con un kernel de tamaño `ventana`x`ventana` dando pasos de `paso` píxeles y se obtiene la suma de píxeles negros. Una vez recorrida toda la imagen, se devuelven las coordenadas donde inicia la ventana y el ancho y alto de la misma, con lo que se pueden calcular las coordenadas de la región con mayor número de píxeles negros.
- `recortar_roi()`: Esta función guarda las regiones de interés de todas las imágenes de la base de datos que se encuentra en la ruta `db_path` en disco, en un directorio `out_path` especificado. Para ello, hace uso de la función `mejor_roi()` tras convertir las imágenes a escala de grises y binarizarlas usando `cv.THRESH_BINARY + cv.THRESH_OTSU`. La binarización combinada es sugerencia de ChatGPT para que el umbral se establezca dinámicamente.

### Ecualizar y normalizar el histograma
El siguiente paso a realizar es ecualizar y normalizar el histograma. Con esto, queremos mejorar el contraste de la imagen. Esta técnica la aprendimos en prácticas anteriores donde ecualizar el histograma de una imagen nos permitía encontrar un equilibrio entre brillos y sombras. Para ello,  usamos la función `ecualizar_histograma()`.
- `ecualizar_histograma()`: Esta función, para cada imagen de la base de datos que se encuentra en `db_path`, convierte la imagen a escala de grises, ecualiza el histograma con `cv.equalizeHist()` y la guarda en disco en el directorio `out_path` especificado.

### Aplicar un filtro bilateral
A continuación, procedemos con la aplicación del filtro bilateral, cuyo objetivo es eliminar ruido pero no las texturas. Este procedimiento lo aprendimos en prácticas anteriores donde vimos la efectividad de los distintos tipos de filtros. De dicha práctica, sacamos la conclusión de que si no queremos difuminar demasiado la imagen, debíamos utilizar valores bajos de *sigma*. Para aplicarlo, definimos la función `aplicar_filtro_bilateral()`.
- `aplicar_filtro_bilateral()`: Esta función convierte cada imagen de la base de datos ubicada en `db_path` en escala de grises y le aplica el filtro bilateral con la función `cv.bilateralFilter`. Los resultados los guarda en disco en el directorio `out_path` especificado.

### Aplicar un realce de crestas
El siguiente paso, es realizar el realce de crestas con operadores de bordes Sobel con el fin de poder discriminar mejor entre crestas y valles. Este paso es una implementación de la teoría explicada en la primera parte de la asignatura. Para ello, creamos la función `realzar_crestas()`.
- `realzar_crestas()`:  Esta función convierte a escala de grises cada imagen de la base de datos ubicada en `db_path` y le aplica los operadores de Sobel, tanto verticales como horizontales. La ponderación de cada gradiente la obtuvimos de la [documentación de OpenCV](#https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html). Por último, se normaliza la imagen resultante y se pasa a `CV_8U` para solventar el *warning* `[ WARN:0@0.762] global loadsave.cpp:1063 cv::imwrite_ Unsupported depth image for selected encoder is fallbacked to CV_8U.`. Esto fue la solución al *warning* que nos dio ChatGPT. Los resultados los guarda en disco en el directorio `out_path` especificado.

### Refinar la ROI
El último paso del preprocesado es refinar la región de interés. El objetivo es eliminar los bordes falsos que se hayan podido detectar en el paso anterior. Este procedimiento es una idea del artículo de Simón-Zorita y se lo pedimos al ChatGPT. La función obtenida es `refinar_roi()`.
- `refinar_roi()`: Esta función recorta el 8% de los márgenes de la imagen con el fin de eliminar los bordes que no pertenecen a la huella. Seguidamente, elimina el posible ruido más pequeño que el `kernel` y luego aplica un filtro mediano, que trata de eliminar el ruido sal-pimienta. Finalmente, normaliza la imagen. Este procedimiento lo aplica a todas las imágenes de la base de datos que se encuentra en `db_path` y guarda los resultados en disco en el directorio `out_path` especificado. 

## Extracción de características
Trabajando en ello...

## Ejecución
```
python main.py
```