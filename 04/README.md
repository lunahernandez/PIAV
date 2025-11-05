# Práctica 4. Procesamiento audio
## Ejercicio 1
En el jupyter notebook `P04_E1.ipynb`.

Los datos utilizados provienen del dataset **TinySOL** (Zenodo), disponible en: https://zenodo.org/records/3685367#.Y7FJqdJBwUEc 

Una vez descargado el archivo, deben extraerse los datos en la siguiente ruta dentro del proyecto: `PIAV/04/data/E1/TinySOL/`



## Ejercicio 2
**Scripts**: `P04_E2.py` y `utils.py`.

### Requisitos (requirements.txt)
- streamlit
- scipy
- numpy
- matplotlib
- soundfile
- streamlit-audiorec

Instalación (desde el directorio principal `PIAV/`):
```bash
pip install -r 04/requirements.txt
```

### Cómo ejecutar (desde el directorio principal `PIAV/`)
```bash
streamlit run 04/P04_E2.py
```