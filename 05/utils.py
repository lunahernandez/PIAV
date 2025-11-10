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
# Transformaciones (C y D)
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
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([k, 0, 0, 0])
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

def match_flann(des_q, des_t, ratio=0.7, crosscheck=False):
    if des_q is None or des_t is None or len(des_q) == 0 or len(des_t) == 0:
        return []
    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    knn = flann.knnMatch(des_q.astype(np.float32), des_t.astype(np.float32), k=2)
    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    if crosscheck and good:
        flann_back = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        knn_back = flann_back.knnMatch(des_t.astype(np.float32), des_q.astype(np.float32), k=2)
        back = {m[0].queryIdx: m[0].trainIdx for m in knn_back if len(m) == 1}
        good = [m for m in good if back.get(m.trainIdx, -1) == m.queryIdx]
    return good

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
