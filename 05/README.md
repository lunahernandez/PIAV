# Práctica 5. Detección de características
Autores: Luna Yue Hernández Guerra y Jorge Lorenzo Lorenzo
# Resumen
En esta práctica se profundiza en la detección de características con la
herramienta SIFT. Para ello, se crea una aplicación web mediante Stream-
lit que permite configurar los parámetros SIFT, delimitar una región de
interés, crear transformaciones afines y distorsiones en tiempo real y de-
tectar la ROI en las imágenes transformadas.

# Requisitos
Para poder ejecutar es necesario tener las siguientes librerías instaladas:
```bash
streamlit
streamlit-cropper
numpy
pandas
pillow
opencv-python
```
Se pueden instalar ejecutando:
```bash
pip install -r requeriments.txt
```
# Cómo ejecutar
```bash
streamlit run app.py
```