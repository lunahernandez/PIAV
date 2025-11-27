import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F

# --- CONFIGURACIÓN Y RUTAS ---
GT_DIR = "videos_with_ground_truth" 
IMG1_PATH = Path(GT_DIR) / "00001_img1.ppm"
IMG2_PATH = Path(GT_DIR) / "00001_img2.ppm"
GT_FLOW_PATH = Path(GT_DIR) / "00001_flow.flo"

# --- UTILIDADES ---

def read_flow_file(filename: Path) -> np.ndarray:
    """Lee un archivo de flujo óptico en formato .flo."""
    with open(filename, 'rb') as f:
        header = np.fromfile(f, np.float32, count=1)[0]
        if header != 202021.25: raise Exception("Formato inválido.")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        flow = np.fromfile(f, np.float32, count=2 * w * h)
    return flow.reshape((h, w, 2))

def calculate_epe(flow_pred, flow_gt):
    """Calcula el End-Point-Error (EPE). Retorna (EPE_valor, flujo_redimensionado)."""
    if isinstance(flow_pred, torch.Tensor): flow_pred = flow_pred.cpu().numpy()
    if flow_pred.shape[0] == 2: flow_pred = np.transpose(flow_pred, (1, 2, 0))
    if flow_gt.shape[0] == 2: flow_gt = np.transpose(flow_gt, (1, 2, 0))
    
    # Redimensionar y escalar si las resoluciones no coinciden (aquí solo se usa si LK es disperso)
    if flow_pred.shape[:2] != flow_gt.shape[:2]:
        scale_x = flow_gt.shape[1] / flow_pred.shape[1]
        scale_y = flow_gt.shape[0] / flow_pred.shape[0]
        flow_pred = cv2.resize(flow_pred, (flow_gt.shape[1], flow_gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        flow_pred[..., 0] *= scale_x
        flow_pred[..., 1] *= scale_y
    
    diff = flow_pred - flow_gt
    epe_map = np.sqrt(diff[..., 0]**2 + diff[..., 1]**2)
    epe = np.mean(epe_map)
    return epe, flow_pred.copy()

def plot_optical_flow(img_ref: np.ndarray, flow_data: np.ndarray, title: str, is_sparse: bool = False):
    """Visualiza el frame de referencia con el flujo óptico."""
    if is_sparse:
        img_display = cv2.cvtColor(flow_data, cv2.COLOR_BGR2RGB)
    else:
        if flow_data.shape[0] == 2: flow_data = np.transpose(flow_data, (1, 2, 0))
        mag, ang = cv2.cartToPolar(flow_data[..., 0], flow_data[..., 1])
        hsv = np.zeros_like(img_ref, dtype=np.uint8)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_img_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        H_flow, W_flow = flow_img_color.shape[:2]
        img_ref_display = cv2.resize(cv2.cvtColor(img_ref, cv2.COLOR_RGB2BGR), (W_flow, H_flow), interpolation=cv2.INTER_LINEAR)
        combined_img = cv2.addWeighted(img_ref_display, 0.5, flow_img_color, 0.5, 0)
        img_display = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(img_display)
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- ALGORITMO LUCAS-KANADE ---

def calculate_lk_opencv_flow(img1_np, img2_np):
    """Calcula el flujo óptico disperso usando cv2.calcOpticalFlowPyrLK."""
    old_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
    H, W = old_gray.shape

    feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    flow_lk_opencv_dense = np.zeros((H, W, 2), dtype=np.float32)
    img_vis = img2_np.copy()
    mask = np.zeros_like(img_vis)

    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for (new_pt, old_pt) in zip(good_new, good_old):
                x_new, y_new = new_pt.ravel()
                x_old, y_old = old_pt.ravel()
                
                flow_lk_opencv_dense[int(y_old), int(x_old), 0] = x_new - x_old
                flow_lk_opencv_dense[int(y_old), int(x_old), 1] = y_new - y_old
                
                mask = cv2.line(mask, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (0, 255, 0), 2)
                cv2.circle(img_vis, (int(x_new), int(y_new)), 3, (0, 0, 255), -1)

            img_vis = cv2.add(img_vis, mask)

    return flow_lk_opencv_dense, img_vis

# --- MAIN EXECUTION ---

def main():
    # Cargar datos
    gt_flow_raw = read_flow_file(GT_FLOW_PATH)
    img1_np = cv2.cvtColor(cv2.imread(str(IMG1_PATH)), cv2.COLOR_BGR2RGB)
    img2_np = cv2.cvtColor(cv2.imread(str(IMG2_PATH)), cv2.COLOR_BGR2RGB)

    # 1. Ejecutar Lucas-Kanade (OpenCV)
    flow_lk_opencv_dense, img_lk_opencv_vis = calculate_lk_opencv_flow(img1_np, img2_np)

    # 2. Preparar Ground Truth (GT) para la métrica EPE (solo en puntos rastreados)
    flow_gt_at_features = gt_flow_raw.copy()
    flow_gt_at_features[flow_lk_opencv_dense[:, :, 0] == 0] = 0
    flow_gt_at_features[flow_lk_opencv_dense[:, :, 1] == 0] = 0

    # 3. Calcular EPE
    epe_lk_opencv, _ = calculate_epe(flow_lk_opencv_dense, flow_gt_at_features) 

    # 4. Mostrar Resultados
    
    # Imagen de Trayectorias
    plot_optical_flow(
        img2_np,
        img_lk_opencv_vis, 
        "Lucas-Kanade (OpenCV) - Trayectorias",
        is_sparse=True
    )

    # Valor EPE
    print(f"Resultado EPE (Lucas-Kanade OpenCV): {epe_lk_opencv:.4f} píxeles/frame.")

if __name__ == "__main__":
    main()