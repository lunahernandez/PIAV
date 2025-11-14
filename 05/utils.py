# utils.py
import cv2 as cv
import numpy as np
from PIL import Image


# Parámetros por defecto de SIFT
DEFAULT_SIFT = dict(
    nfeatures=1000,
    nOctaveLayers=3,
    contrastThreshold=0.04,
    edgeThreshold=10.0,
    sigma=1.6,
)


# Utilidades
def imagen_to_bgr(file):
    """Convierte un archivo de imagen a formato BGR."""
    image = Image.open(file).convert("RGB")
    array = np.array(image)
    bgr = cv.cvtColor(array, cv.COLOR_RGB2BGR)
    return bgr


def bgr_to_rgb(img):
    """Convierte una imagen BGR en RGB."""
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def to_gray(img):
    """Convierte una imagen BGR en RGB."""
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img


# Transformaciones predefinidas
def make_canvas(img):
    """
    Devuelve un lienzo 2x blanco, el desplazamiento para centrar
    la imagen y las dimensiones del lienzo.
    """
    h, w = img.shape[:2]
    CW, CH = int(w * 2), int(h * 2)
    canvas = np.full((CH, CW, 3), 255, dtype=np.uint8)
    ox, oy = (CW - w) // 2, (CH - h) // 2
    canvas[oy:oy+h, ox:ox+w] = img
    return canvas, (ox, oy), (CW, CH)


def affine_matrix(ang, sx, sy, cx, cy, tx, ty):
    """
    Construye una matriz 2x3 tal que x' = A x + b, 
    con pivote (cx,cy) y traslación (tx,ty).
    """
    rad = np.deg2rad(ang)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[ c, -s],
                  [ s,  c]], dtype=np.float32)
    S = np.array([[sx, 0.0],
                  [0.0, sy]], dtype=np.float32)
    A = (R @ S).astype(np.float32)
    I = np.eye(2, dtype=np.float32)
    pivot = np.array([cx, cy], dtype=np.float32)
    t = np.array([tx, ty], dtype=np.float32)
    b = (t + (I - A) @ pivot).astype(np.float32)  # b = t + (I - A) pivot
    M = np.hstack([A, b.reshape(2, 1)]) # 2x3
    return M


def apply_affine(img, tx, ty, angle, cx, cy, sx, sy):
    """
    Aplica la transformación afín a la imagen y devuelve 
    tanto el lienzo completo como la vista recortada.
    """
    canvas, (ox, oy), (CW, CH) = make_canvas(img)
    cx_canvas = ox + cx
    cy_canvas = oy + cy
    M = affine_matrix(angle, sx, sy, cx_canvas, cy_canvas, tx, ty)
    out = cv.warpAffine(canvas, M, (CW, CH), borderValue=(255, 255, 255))
    view = out[oy:oy+img.shape[0], ox:ox+img.shape[1]].copy()
    return out, view


def apply_distortion(image, k1, k2, p1, p2, k3, center=None, focal=10.0):
    """Aplica distorsión radial/tangencial y devuelve la imagen distorsionada."""
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

    out = cv.undistort(image, cam, dist)
    return out


def apply_transform(base_img, params):
    """
    Aplica una transformación según unos parámetros del tipo:
      - {'type':'affine', 'tx':.., 'ty':.., 'angle':.., 'cx':.., 'cy':.., 'sx':..,'sy':..}
      - {'type':'distortion','k1':..,'k2':..,'p1':..,'p2':..,'k3':..,'cx':.. or None,'cy':.. or None,'focal':..}
    Devuelve el nombre de la transformación y la imagen transformada.
    """
    type = params.get("type")
    name = params.get("name", type)
    if type == "affine":
        tx = float(params.get("tx", 0.0))
        ty = float(params.get("ty", 0.0))
        ang = float(params.get("angle", 0.0))
        cx  = float(params.get("cx", 0.0))
        cy  = float(params.get("cy", 0.0))
        sx  = float(params.get("sx", 1.0))
        sy  = float(params.get("sy", 1.0))
        out, view = apply_affine(base_img, tx, ty, ang, cx, cy, sx, sy)
        return name, view
    elif type == "distortion":
        k1 = float(params.get("k1", 0.0))
        k2 = float(params.get("k2", 0.0))
        p1 = float(params.get("p1", 0.0))
        p2 = float(params.get("p2", 0.0))
        k3 = float(params.get("k3", 0.0))
        focal = float(params.get("focal", 10.0))
        if params.get("cx") is None or params.get("cy") is None:
            center = None
        else:
            center = (float(params["cx"]), float(params["cy"]))
        out = apply_distortion(base_img, k1, k2, p1, p2, k3, center=center, focal=focal)
        return name, out
    else:
        raise ValueError(f"Tipo de transformación no soportado: {type}")


# Transformaciones personalizadas 
def affine_matrix_RS(ang, sx, sy):
    """Devuelve las matrices A (escalado y rotación) y la matriz identidad I."""
    rad = np.deg2rad(ang)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[ c, -s],
                  [ s,  c]], dtype=np.float32)
    S = np.array([[sx, 0.0],
                  [0.0, sy]], dtype=np.float32)
    A = (R @ S).astype(np.float32)
    I = np.eye(2, dtype=np.float32)
    return A, I


def affine_update_t_for_pivot(t_state, prev_c, c_now, A, I):
    """
    Devuelve el vector de traslación t. 
    Mantiene b = t + (I - A)·c constante al cambiar c.
    """
    b_keep = t_state + (I - A) @ prev_c
    t_new  = b_keep - (I - A) @ c_now
    return t_new.astype(np.float32)


def affine_matrix_realtime(A, t, c, I):
    """
    Devuelve una matriz M 2x3 tal que x' = A x + b,
    y el vector de traslación b, con compensación.
    """
    b = (t + (I - A) @ c).astype(np.float32)
    M = np.hstack([A, b.reshape(2,1)])
    return M, b


def apply_affine_realtime(img, M, A, b, c_now):
    """
    Aplica la transformación afín a la imagen y devuelve 
    la vista recortada, el pivote, las dimensiones del
    lienzo y los desplazamientos para centrar la imagen.
    """
    h, w = img.shape[:2]
    CW, CH = int(w*2), int(h*2)
    canvas = np.full((CH, CW, 3), 255, dtype=np.uint8)
    ox, oy = (CW - w)//2, (CH - h)//2
    canvas[oy:oy+h, ox:ox+w] = img

    out = cv.warpAffine(canvas, M, (CW, CH), borderValue=(255,255,255))
    pivot  = (A @ c_now + b).astype(int)

    view = out[oy:oy+h, ox:ox+w].copy()
    return view, pivot, CW, CH, ox, oy


# SIFT
def create_sift(params):
    """Crea un detector SIFT con los parámetros especificados."""
    return cv.SIFT_create(
        nfeatures=int(params["nfeatures"]),
        nOctaveLayers=int(params["nOctaveLayers"]),
        contrastThreshold=float(params["contrastThreshold"]),
        edgeThreshold=float(params["edgeThreshold"]),
        sigma=float(params["sigma"]),
    )


def extract_sift(gray, params):
    """Extrae los puntos claves y los descriptores SIFT de una imagen en escala de grises."""
    sift = create_sift(params)
    return sift.detectAndCompute(gray, None), sift


def match_bf_crosscheck(des1, des2):
    """Empareja descriptores con BFMatcher crossCheck y devuelve los matches ordenados."""
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    return sorted(matches, key=lambda x: x.distance)


def match_knn_ratio(des1, des2, ratio=0.75, k=2):
    """
    Empareja descriptores con KNN, aplica el test de ratio de Lowe
    y devuelve los matches ordenados.
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

    return sorted(good, key=lambda x: x.distance)


def estimate_geom(kp_src, kp_dst, matches, ransac_thresh=5.0, prefer_affine=False):
    """
    Estima la transformación afín o homografía entre dos conjuntos de puntos 
    claves mediante RANSAC. Devuelve la matriz de transformación, la máscara 
    de inliers, el número de inliers y el tipo de transformación.
    """
    if len(matches) < 4:
        return None, None, 0, None
    
    src = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    A, mask_inliers_A = None, None
    inliers_aff = -1

    if prefer_affine:
        A, mask_inliers_A = cv.estimateAffinePartial2D(src, dst, method=cv.RANSAC, ransacReprojThreshold=ransac_thresh)
        inliers_aff = int(mask_inliers_A.sum()) if mask_inliers_A is not None else -1

    H, mask_inliers_H = cv.findHomography(src, dst, cv.RANSAC, ransac_thresh)
    inliers_H = int(mask_inliers_H.sum()) if mask_inliers_H is not None else -1

    use_aff = prefer_affine and inliers_aff >= max(4, inliers_H)
    inliers = inliers_aff if use_aff else inliers_H

    if use_aff and A is not None and inliers >= 4:
        return A, mask_inliers_A, inliers, "affine"
    if (not use_aff) and H is not None and inliers >= 4:
        return H, mask_inliers_H, inliers, "homography"
    return None, None, 0, None


def transform_roi_box(M, model_type, roi_shape):
    """
    Proyecta el recuadro de la ROI usando la matriz M y devuelve sus 4 vértices
    transformados como polígono.
    """
    h, w = roi_shape
    box = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    if model_type == "homography":
        box_transformed = cv.perspectiveTransform(box, M).reshape(-1, 2)
    else:
        pts = box.reshape(-1, 2)
        box_transformed = cv.transform(pts[None, :, :], M)[0]

    return box_transformed.astype(np.int32)


def is_valid_poly(poly, img_shape, min_area_ratio=1e-4, max_area_ratio=0.95):
    """
    Comprueba si el polígono dibujado es válido según si esta dentro de la imagen,
    si es convexo, si su área es adecuada y si sus lados no son demasiado pequeños
    """
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

# Matches
def draw_matches(img_left, img_right, kp_left, kp_right, matches, topN=100, poly=None, banner=None):
    """
    Dibuja los matches entre las dos imagenes, dibujando las líneas entre los puntos coincidentes.
    """
    left = cv.cvtColor(img_left, cv.COLOR_GRAY2BGR) if img_left.ndim == 2 else img_left.copy()
    right = cv.cvtColor(img_right, cv.COLOR_GRAY2BGR) if img_right.ndim == 2 else img_right.copy()

    h = max(left.shape[0], right.shape[0])
    left_padded = cv.copyMakeBorder(left, 0, h-left.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(0,0,0))
    right_padded = cv.copyMakeBorder(right, 0, h-right.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(0,0,0))
    view = cv.hconcat([left_padded, right_padded])

    offset = left_padded.shape[1]
    n = min(topN, len(matches))
    for m in matches[:n]:
        pt1 = tuple(np.int32(kp_left[m.queryIdx].pt))
        pt2 = tuple(np.int32(kp_right[m.trainIdx].pt))
        pt2_shift = (pt2[0] + offset, pt2[1])
        color = tuple(int(c) for c in np.random.randint(60, 255, 3))
        cv.circle(view, pt1, 3, color, -1, cv.LINE_AA)
        cv.circle(view, pt2_shift, 3, color, -1, cv.LINE_AA)
        cv.line(view, pt1, pt2_shift, color, 1, cv.LINE_AA)

    if poly is not None and poly.size == 8:
        poly = poly.reshape(-1, 2).astype(int)
        poly[:, 0] += offset
        cv.polylines(view, [poly.reshape(-1,1,2)], True, (0,0,255), 3, cv.LINE_AA)

    if banner is not None:
        txt, col = banner
        cv.putText(view, txt, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv.LINE_AA)

    return view
