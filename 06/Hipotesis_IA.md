# Hipótesis B: con IA
**Autores**: Luna Yue Hernández Guerra y Jorge Lorenzo Lorenzo
## Preprocesado
Se reutiliza el preprocesado desarrollado en la Hipótesis A (sin IA).

### Cargar las imágenes
Esto función fse encarga de cargar la imagen en escala de grises desde una ruta dada y luego se redimenciona a un tamaño que se le pasa como parámetro, por último se convierten a tensores de PyTorch y se normaliza dividendo entre 255 para que los valores estén entre 0 y 1. Se devuelve el tensor resultante aplicando `unsqueeze(0)` para añadir una dimensión adicional al tensor, para que la red convolucional pueda procesarlo correctamente.

### Modelo Siamesa
Usamos este modelo ya que son dos redes neuronales convolucionales que comparten pesos y se utilizan para aprender una función de similitud entre dos entradas. En nuestro caso, las entradas son las imágenes de las huellas. La estructura del modelo consta de dos bloques el primero tiene toda la parte convolucional y el segundo bloque es la parte lineal que obtiene el embedding de la imagen. Luego están los métodos `forward_once` y `forward`. El primero procesa una sola imagen y el segundo procesa dos imágenes y devuelve sus embeddings.

### Dataset de pares de huellas (Implementación hecha por GPT)
Este dataset personalizado que se encarga de generar pares de imágenes de huellas dactilares, donde tenemos las siguientes etiquetas `1` siendo pares positivos del mismo usuario y `0` que son de usuarios diferentes, esto se hace para poder generar la perdida. Lueog están los métodos `__len__` y `__getitem__`. El primero devuelve el tamaño del dataset y el segegundo devuelve los tensores de las dos imágenes y su etiqueta correspondiente, siendo esto lo que la red necesita.

### Distancia Coseno
Calcula la distancia coseno entre los dos vectores embeddings de las huellas para comparar si son del mismo usuario o no.

### Función de comparación con el mismo usuario
Esta función que implementamos necesita el modelo, la ruta de salida donde están las imágenes refinadas y luego el umbral de acpetación. Se obtienen las lista de usuarios e reocrren y se obtienen las imágenes, carga las 2 imágenes, se añade al tensor una dimensión más para luego pasarlas por el modelo y obtener sus embeddings. Luego se calcula la distancia coseno entre los embeddings y se compara con el umbral para ver si son el mismo usuario o no.

### Función de comparación entre usuarios
Esta función la utilizamos para comparar un usuario elegido con el resto de imagenes para ver como funciona la red siamesa, la dínamica es similar a la función anterior, la lógica es que no debería de haber matches entre usuarios diferentes.