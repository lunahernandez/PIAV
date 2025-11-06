import cv2 as cv
import os
import numpy as np
from pathlib import Path

def mejor_roi_por_negros(bw, ventana=500, paso=20, negros_val=255):
    H, W = bw.shape
    w = h = min(ventana, W, H)
    mask = (bw == negros_val).astype(np.uint8)
    integ = cv.integral(mask)

    def suma(x, y, w, h):
        x2, y2 = x + w, y + h
        return int(integ[y2, x2] - integ[y, x2] - integ[y2, x] + integ[y, x])

    best = (0, 0, w, h)
    best_sum = -1
    max_y = max(1, H - h + 1)
    max_x = max(1, W - w + 1)
    for y in range(0, max_y, paso):
        for x in range(0, max_x, paso):
            s = suma(x, y, w, h)
            if s > best_sum:
                best_sum = s
                best = (x, y, w, h)
    return best

def recorte_por_negros(img_path, ventana=500, paso=20, invert=False, out_dir=None):
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    flag = cv.THRESH_BINARY_INV if not invert else cv.THRESH_BINARY
    _, bw = cv.threshold(gray, 0, 255, flag + cv.THRESH_OTSU)

    total_pix = bw.size
    negros = int(np.sum(bw == (255 if not invert else 0)))
    blancos = total_pix - negros

    x, y, w, h = mejor_roi_por_negros(bw, ventana=ventana, paso=paso,
                                      negros_val=(255 if not invert else 0))
    roi = img[y:y+h, x:x+w]

    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        base = Path(img_path).stem
        roi_path = Path(out_dir) / f"{base}_roi.png"

        cv.imwrite(str(roi_path), roi)

    return roi, bw, (x, y, w, h), (negros, blancos)

def main():
    sample_path = r"06/data"
    folders = os.listdir(sample_path)

    print("Carpetas encontradas:", folders)
    for folder in folders:
        folder_path = os.path.join(sample_path, folder)
        if not os.path.isdir(folder_path):
            continue

        out_dir = os.path.join(folder_path, "roi")

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]
        if len(image_files) == 0:
            print(f"(i) '{folder}' no tiene imágenes .png")
            continue

        image_files.sort()
        for fname in image_files:
            img_path = os.path.join(folder_path, fname)
            recorte_por_negros(img_path, ventana=500, paso=15, invert=False, out_dir=out_dir)

if __name__ == "__main__":
    main()
