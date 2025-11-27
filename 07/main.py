import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# NOTA: Asegúrate de que esta ruta sea correcta (ej. "07/videos_with_ground_truth" 
# si ejecutas desde PIAV/). Si no, ajusta solo esta línea:
GT_DIR = "videos_with_ground_truth" 
IMG1_PATH = Path(GT_DIR) / "00001_img1.ppm"
IMG2_PATH = Path(GT_DIR) / "00001_img2.ppm"
FLOW_GT_PATH = Path(GT_DIR) / "00001_flow.flo"

# --- UTILIDADES ---

def read_flow(f):
    """Lee el archivo .flo y devuelve el array (H, W, 2)."""
    with open(f, 'rb') as f:
        # Verifica el 'magic number' y lee dimensiones
        np.fromfile(f, np.float32, count=1) 
        w, h = np.fromfile(f, np.int32, count=2)
        flow = np.fromfile(f, np.float32, count=2 * w * h)
    return flow.reshape((h, w, 2))

def get_epe(pred, gt):
    """Calcula el End-Point-Error (EPE)."""
    # El EPE es la distancia euclidiana promedio entre el flujo predicho y el GT.
    diff = pred - gt
    epe = np.mean(np.sqrt(diff[:,:, 0]**2 + diff[:,:, 1]**2))
    return epe

def lk_flow(img1, img2):
    """Calcula el flujo LK (OpenCV) y genera la imagen de visualización."""
    old_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    H, W = old_gray.shape

    # Parámetros mínimos para LK
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
    flow = np.zeros(img1.shape[:2] + (2,), dtype=np.float32)
    img_vis, mask = img2.copy(), np.zeros_like(img2)

    if p0 is not None:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, 
                                               winSize=(15, 15), maxLevel=2, 
                                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        if p1 is not None:
            # Dibujar y almacenar el flujo para los puntos rastreados
            for new_pt, old_pt in zip(p1[st == 1], p0[st == 1]):
                x_new, y_new = new_pt.ravel()
                x_old, y_old = old_pt.ravel()
                
                flow[int(y_old), int(x_old)] = [x_new - x_old, y_new - y_old]
                mask = cv2.line(mask, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (0, 255, 0), 2)
                cv2.circle(img_vis, (int(x_new), int(y_new)), 3, (0, 0, 255), -1)

            img_vis = cv2.add(img_vis, mask)

    return flow, img_vis

# --- MAIN EXECUTION ---

def main():
    # 1. Carga de datos (convertir BGR a RGB)
    gt_flow = read_flow(FLOW_GT_PATH)
    img1 = cv2.cvtColor(cv2.imread(str(IMG1_PATH)), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(str(IMG2_PATH)), cv2.COLOR_BGR2RGB)

    # 2. Calcular Flujo LK
    flow_pred, img_vis = lk_flow(img1, img2)

    # 3. Preparar GT para EPE (enmascarar el GT para comparar solo en puntos predichos)
    flow_gt_masked = gt_flow.copy()
    flow_gt_masked[flow_pred[:, :, 0] == 0] = 0
    flow_gt_masked[flow_pred[:, :, 1] == 0] = 0

    # 4. Calcular EPE
    epe = get_epe(flow_pred, flow_gt_masked) 

    # 5. Mostrar Resultados (Integrado en el main)
    
    # Visualización
    img_display = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 4))
    plt.imshow(img_display)
    plt.title(f"LK OpenCV Flow (EPE: {epe:.4f})")
    plt.axis('off')
    plt.show()
    
    # Valor EPE
    print(f"Resultado EPE (Lucas-Kanade OpenCV): {epe:.4f} píxeles/frame.")

if __name__ == "__main__":
    main()