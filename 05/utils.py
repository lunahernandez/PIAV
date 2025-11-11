# utils.py
import cv2 as cv
import numpy as np
from PIL import Image

# ----------------------------
# Parámetros por defecto SIFT
# ----------------------------
DEFAULT_SIFT = dict(
    nfeatures=1000,
    nOctaveLayers=3,
    contrastThreshold=0.04,
    edgeThreshold=10.0,
    sigma=1.6,
)

# ----------------------------
# Utilidades de imagen
# ----------------------------
def file_to_bgr(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    arr = np.array(image)[:, :, ::-1]  # RGB->BGR
    return arr

def bgr_to_rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def to_gray(img_bgr):
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

def draw_text(img, text, org=(10, 30), color=(0, 255, 0)):
    vis = img.copy()
    cv.putText(vis, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(vis, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
    return vis

def norm_int(v, lo, hi):
    return max(lo, min(hi, v))

# ----------------------------
# Transformaciones “por defecto”
# ----------------------------
def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv.warpAffine(img, M, (new_w, new_h), borderValue=(255, 255, 255))

def scale_image(img, scale):
    h, w = img.shape[:2]
    scaled = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
    if scale > 1.0:
        new_h, new_w = scaled.shape[:2]
        y0, x0 = (new_h - h) // 2, (new_w - w) // 2
        return scaled[y0:y0+h, x0:x0+w]
    else:
        new_h, new_w = scaled.shape[:2]
        result = np.full((h, w, 3), 255, dtype=np.uint8)
        y0, x0 = (h - new_h) // 2, (w - new_w) // 2
        result[y0:y0+new_h, x0:x0+new_w] = scaled
        return result

def translate_image(img, tx, ty):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

def perspective_transform(img, intensity="leve"):
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    base_offset = min(w, h) // 20
    factor = {"leve": 1, "moderada": 2, "fuerte": 3}.get(intensity, 1)
    offset = base_offset * factor
    pts2 = np.float32([
        [offset, offset//2],
        [w - offset*2, offset],
        [offset//2, h - offset],
        [w - offset, h - offset//2]
    ])
    M = cv.getPerspectiveTransform(pts1, pts2)
    return cv.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))

def deform_radial(img, k=0.00001):
    h, w = img.shape[:2]
    fx = fy = 1.0
    cx, cy = w / 2, h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    D = np.array([k, 0, 0, 0], dtype=np.float32)
    map1, map2 = cv.initUndistortRectifyMap(K, D, None, K, (w, h), cv.CV_32FC1)
    return cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderValue=(255, 255, 255))

def deform_barrel(img, k=0.00001):
    return deform_radial(img, k=k)

def deform_pincushion(img, k=-0.00001):
    return deform_radial(img, k=k)

# ----------------------------
# SIFT y matching
# ----------------------------
def create_sift(params):
    return cv.SIFT_create(
        nfeatures=int(params["nfeatures"]),
        nOctaveLayers=int(params["nOctaveLayers"]),
        contrastThreshold=float(params["contrastThreshold"]),
        edgeThreshold=float(params["edgeThreshold"]),
        sigma=float(params["sigma"]),
    )

def extract_sift(gray, params):
    sift = create_sift(params)
    return sift.detectAndCompute(gray, None), sift


def estimate_geom(kp_q, kp_t, good, ransac_thresh=5.0, prefer_affine=False):
    if len(good) < 4:
        return None, None, 0, None
    src = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    A, maskA = None, None
    inliers_aff = -1
    if prefer_affine:
        A, maskA = cv.estimateAffinePartial2D(src, dst, method=cv.RANSAC, ransacReprojThreshold=ransac_thresh)
        inliers_aff = int(maskA.sum()) if maskA is not None else -1

    H, maskH = cv.findHomography(src, dst, cv.RANSAC, ransac_thresh)
    inliers_H = int(maskH.sum()) if maskH is not None else -1

    use_aff = prefer_affine and inliers_aff >= max(4, inliers_H)
    inliers = inliers_aff if use_aff else inliers_H
    if use_aff and A is not None and inliers >= 4:
        return A, maskA, inliers, "affine"
    if (not use_aff) and H is not None and inliers >= 4:
        return H, maskH, inliers, "homography"
    return None, None, 0, None

def project_box(M, model_kind, roi_shape):
    h, w = roi_shape
    box = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    if model_kind == "homography":
        proj = cv.perspectiveTransform(box, M).reshape(-1, 2)
    else:
        pts = box.reshape(-1, 2)
        proj = cv.transform(pts[None, :, :], M)[0]
    return proj.astype(np.int32)

def is_valid_quad(poly, img_shape, min_area_ratio=1e-4, max_area_ratio=0.95):
    if poly is None or len(poly) != 4:
        return False
    h, w = img_shape[:2]
    poly_int = np.int32(poly.reshape(-1, 2))
    if not cv.isContourConvex(poly_int):
        return False
    if (poly_int[:, 0].min() < -0.05 * w or poly_int[:, 0].max() > 1.05 * w or
        poly_int[:, 1].min() < -0.05 * h or poly_int[:, 1].max() > 1.05 * h):
        return False
    area = cv.contourArea(poly_int)
    area_ratio = area / float(w * h)
    if not (min_area_ratio <= area_ratio <= max_area_ratio):
        return False
    lens = np.linalg.norm(np.diff(np.vstack([poly_int, poly_int[0]]), axis=0), axis=1)
    if lens.min() < 5:
        return False
    return True

# ----------------------------
# Motor de transformaciones personalizadas
# ----------------------------

def affine_matrix_with_pivot(angle_deg, sx, sy, cx, cy, tx, ty):
    """Construye M 2x3 tal que x' = A x + b, con pivote (cx,cy) y traslación (tx,ty)."""
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[ c, -s],
                  [ s,  c]], dtype=np.float32)
    S = np.array([[sx, 0.0],
                  [0.0, sy]], dtype=np.float32)
    A = (R @ S).astype(np.float32)
    I = np.eye(2, dtype=np.float32)
    c_vec = np.array([cx, cy], dtype=np.float32)
    t_vec = np.array([tx, ty], dtype=np.float32)
    b = (t_vec + (I - A) @ c_vec).astype(np.float32)  # b = t + (I - A) c
    M = np.hstack([A, b.reshape(2, 1)])  # 2x3
    return M

def warp_affine_on_canvas(img, scale_factor, tx, ty, angle, cx, cy, sx, sy):
    """Aplica afin con pivote a la imagen centrada en un canvas ampliado."""
    canvas, (ox, oy), (CW, CH) = make_canvas_centered(img, scale_factor)
    M = affine_matrix_with_pivot(angle, sx, sy, cx, cy, tx, ty)
    out = cv.warpAffine(canvas, M, (CW, CH), borderValue=(255, 255, 255))
    # recorte de vista central (mismo tamaño que la imagen original)
    view = out[oy:oy+img.shape[0], ox:ox+img.shape[1]].copy()
    return out, view, (CW, CH)

def apply_distortion_full(image, k1, k2, p1, p2, k3, center=None, focal=10.0):
    """Aplica distorsión radial/tangencial (como la demo), devolviendo imagen del mismo tamaño."""
    h, w = image.shape[:2]
    cam = np.eye(3, dtype=np.float32)
    cam[0, 0] = focal
    cam[1, 1] = focal
    if center is None:
        cam[0, 2] = w / 2.0
        cam[1, 2] = h / 2.0
    else:
        cam[0, 2] = float(center[0])
        cam[1, 2] = float(center[1])

    dist = np.zeros((5, 1), np.float64)
    dist[0, 0] = float(k1)
    dist[1, 0] = float(k2)
    dist[2, 0] = float(p1)
    dist[3, 0] = float(p2)
    dist[4, 0] = float(k3)

    # undistort aplica el modelo inverso: para simular distorsión “hacia fuera”
    # puedes usar signos positivos/negativos apropiados en k1,k2,k3.
    out = cv.undistort(image, cam, dist)
    return out

def apply_transform_spec(base_img, spec):
    """
    Aplica una transformación según un spec:
      - {'type':'affine', 'scale_factor':2.0, 'tx':.., 'ty':.., 'angle':.., 'cx':.., 'cy':.., 'sx':..,'sy':..}
      - {'type':'distortion','k1':..,'k2':..,'p1':..,'p2':..,'k3':..,'cx':.. or None,'cy':.. or None,'focal':..}
    Devuelve (nombre, imagen_transformada).
    """
    t = spec.get("type")
    name = spec.get("name", t)
    if t == "affine":
        sf = float(spec.get("scale_factor", 2.0))
        tx = float(spec.get("tx", 0.0))
        ty = float(spec.get("ty", 0.0))
        ang = float(spec.get("angle", 0.0))
        cx  = float(spec.get("cx", 0.0))
        cy  = float(spec.get("cy", 0.0))
        sx  = float(spec.get("sx", 1.0))
        sy  = float(spec.get("sy", 1.0))
        out, view, _ = warp_affine_on_canvas(base_img, sf, tx, ty, ang, cx, cy, sx, sy)
        return name, view
    elif t == "distortion":
        k1 = float(spec.get("k1", 0.0))
        k2 = float(spec.get("k2", 0.0))
        p1 = float(spec.get("p1", 0.0))
        p2 = float(spec.get("p2", 0.0))
        k3 = float(spec.get("k3", 0.0))
        focal = float(spec.get("focal", 10.0))
        if spec.get("cx") is None or spec.get("cy") is None:
            center = None
        else:
            center = (float(spec["cx"]), float(spec["cy"]))
        out = apply_distortion_full(base_img, k1, k2, p1, p2, k3, center=center, focal=focal)
        return name, out
    else:
        raise ValueError(f"Tipo de transformación no soportado: {t}")

import cv2 as cv
import numpy as np

def match_bf_crosscheck(des1, des2):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    return sorted(matches, key=lambda x: x.distance)

def match_knn_ratio(des1, des2, ratio=0.75, k=2):
    """
    Empareja con KNN y aplica el ratio test de Lowe.
    Devuelve una lista plana de DMatch (solo el mejor por par filtrado).
    """
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=k)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < ratio * n.distance:
            good.append(m)
    # ordenar por distancia, igual que en BF clásico
    return sorted(good, key=lambda x: x.distance)



def draw_matches_panel(img_left, img_right, kp_left, kp_right, matches, topN=100, poly_right=None, banner=None):
    left = cv.cvtColor(img_left, cv.COLOR_GRAY2BGR) if img_left.ndim == 2 else img_left.copy()
    right = cv.cvtColor(img_right, cv.COLOR_GRAY2BGR) if img_right.ndim == 2 else img_right.copy()

    h = max(left.shape[0], right.shape[0])
    left_p = cv.copyMakeBorder(left, 0, h-left.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(0,0,0))
    right_p = cv.copyMakeBorder(right, 0, h-right.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(0,0,0))
    vis = cv.hconcat([left_p, right_p])

    offset = left_p.shape[1]
    n = min(topN, len(matches))
    for m in matches[:n]:
        pt1 = tuple(np.int32(kp_left[m.queryIdx].pt))
        pt2 = tuple(np.int32(kp_right[m.trainIdx].pt))
        pt2_shift = (pt2[0] + offset, pt2[1])
        color = tuple(int(c) for c in np.random.randint(60, 255, 3))
        cv.circle(vis, pt1, 3, color, -1, cv.LINE_AA)
        cv.circle(vis, pt2_shift, 3, color, -1, cv.LINE_AA)
        cv.line(vis, pt1, pt2_shift, color, 1, cv.LINE_AA)

    if poly_right is not None and poly_right.size == 8:
        poly = poly_right.reshape(-1, 2).astype(int)
        poly[:, 0] += offset
        cv.polylines(vis, [poly.reshape(-1,1,2)], True, (0,0,255), 3, cv.LINE_AA)

    if banner is not None:
        txt, col = banner
        cv.putText(vis, txt, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv.LINE_AA)

    return vis

def to_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img

def bgr_to_rgb(img):
    return img[:, :, ::-1] if img.ndim == 3 else img

import cv2 as cv
import numpy as np

# --- Matrices A = R @ S y la identidad I ---
def affine_matrix_RS(angle_deg: float, sx: float, sy: float):
    """Devuelve (A, I) con A = R(ang) @ S(sx,sy)."""
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[ c, -s],
                  [ s,  c]], dtype=np.float32)
    S = np.array([[sx, 0.0],
                  [0.0, sy]], dtype=np.float32)
    A = (R @ S).astype(np.float32)
    I = np.eye(2, dtype=np.float32)
    return A, I

def affine_update_t_for_pivot(t_state: np.ndarray, prev_c: np.ndarray, c_now: np.ndarray, A: np.ndarray, I: np.ndarray):
    """
    Mantiene constante b = t + (I - A)·c al cambiar c:  t' = b_keep - (I - A)·c_now
    donde b_keep = t_state + (I - A)·prev_c
    """
    b_keep = t_state + (I - A) @ prev_c
    t_new  = b_keep - (I - A) @ c_now
    return t_new.astype(np.float32)

def affine_build_2x3(A: np.ndarray, t: np.ndarray, c: np.ndarray, I: np.ndarray):
    """Devuelve (M 2x3, b) con x' = A x + b,  b = t + (I - A)·c."""
    b = (t + (I - A) @ c).astype(np.float32)
    M = np.hstack([A, b.reshape(2,1)])
    return M, b

def affine_preview_on_canvas(img: np.ndarray, scale_factor: float, M: np.ndarray, A: np.ndarray, b: np.ndarray, c_now: np.ndarray):
    """
    Aplica warp sobre un lienzo centrado y devuelve:
      - view: recorte central del tamaño original
      - pc: pivote transformado (A·c + b)
      - CW, CH, ox, oy: datos del lienzo/offset
    """
    h, w = img.shape[:2]
    CW, CH = int(w*scale_factor), int(h*scale_factor)
    canvas = np.full((CH, CW, 3), 255, dtype=np.uint8)
    ox, oy = (CW - w)//2, (CH - h)//2
    canvas[oy:oy+h, ox:ox+w] = img

    out = cv.warpAffine(canvas, M, (CW, CH), borderValue=(255,255,255))
    pc  = (A @ c_now + b).astype(int)

    # Marcador del pivote (opcional, el app dibuja sobre view si quiere)
    disp = out  # si quieres dibujar aquí, haz una copia: out.copy()

    view = disp[oy:oy+h, ox:ox+w].copy()
    return view, pc, CW, CH, ox, oy


def make_canvas_centered(img, scale_factor=2): 
    """Devuelve un canvas blanco (CH,CW,3) y el offset (ox,oy) para situar la imagen centrada.""" 
    h, w = img.shape[:2] 
    CW, CH = int(w * scale_factor), int(h * scale_factor) 
    canvas = np.full((CH, CW, 3), 255, dtype=np.uint8) 
    ox, oy = (CW - w) // 2, (CH - h) // 2 
    canvas[oy:oy+h, ox:ox+w] = img 
    return canvas, (ox, oy), (CW, CH)

