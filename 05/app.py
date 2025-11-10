# app.py
import io
import zipfile
from pathlib import Path
from enum import Enum

import cv2 as cv
import numpy as np
from PIL import Image
import streamlit as st

# Intentamos usar canvas para ROI
try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

# ----------------------------
# Tipos e infraestructura
# ----------------------------
class ImageType(Enum):
    LIBRE = "LIBRE"
    MEDICA = "MEDICA"
    TELEMETRICA = "TELEMETRICA"

DEFAULT_SIFT = dict(
    nfeatures=1000,
    nOctaveLayers=3,
    contrastThreshold=0.04,
    edgeThreshold=10.0,
    sigma=1.6,
)

def ensure_ss():
    if "images" not in st.session_state:
        st.session_state.images = {}            # {ImageType: np.ndarray BGR}
    if "current_type" not in st.session_state:
        st.session_state.current_type = ImageType.LIBRE.name  # 'LIBRE' (string)
    if "sift" not in st.session_state:
        st.session_state.sift = DEFAULT_SIFT.copy()
    if "rois" not in st.session_state:
        st.session_state.rois = {}              # {ImageType: dict(image=np.ndarray, bbox=(x,y,w,h))}
    if "transforms_cache" not in st.session_state:
        st.session_state.transforms_cache = {}


ensure_ss()

# --- Normalización de claves de session_state a strings ---
def _normalize_dict_keys_to_str(d):
    changed = False
    keys = list(d.keys())
    for k in keys:
        if isinstance(k, ImageType):
            d[k.name] = d.pop(k)   # Enum -> 'LIBRE'
            changed = True
    return changed

_normalize = False
if isinstance(st.session_state.current_type, ImageType):
    st.session_state.current_type = st.session_state.current_type.name
    _normalize = True

# Rehacer claves de images/rois a strings si traen Enums de recargas previas
if any(isinstance(k, ImageType) for k in st.session_state.images.keys()):
    _normalize |= _normalize_dict_keys_to_str(st.session_state.images)

if any(isinstance(k, ImageType) for k in st.session_state.rois.keys()):
    _normalize |= _normalize_dict_keys_to_str(st.session_state.rois)


def current_enum():
    # Convierte el nombre guardado en Enum
    return ImageType[st.session_state.current_type]

def current_name() -> str:
    return st.session_state.current_type  # 'LIBRE', ...

def to_enum(name: str) -> ImageType:
    return ImageType[name]

def get_img(name: str):
    return st.session_state.images.get(name)

def get_roi(name: str):
    return st.session_state.rois.get(name)



# ----------------------------
# Utilidades de imagen
# ----------------------------
def file_to_bgr(uploaded):
    image = Image.open(uploaded).convert("RGB")
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

def norm_int(v, lo, hi):  # clamp
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

def deform_barrel(img, k=0.00001):   # barril
    return deform_radial(img, k=k)

def deform_pincushion(img, k=-0.00001):  # cojín
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
        knn_back = flann_back.knnMatch(des_t.astype(np.float32), des_q.astype(np.float32), k=1)
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
# ----------------------------
# Sidebar: carga y SIFT
# ----------------------------
st.sidebar.title("Parámetros y datos")

# Selección de tipo actual usando STRINGS (no Enums)
type_options = [t.name for t in ImageType]  # ['LIBRE','MEDICA','TELEMETRICA']
try:
    idx = type_options.index(st.session_state.current_type)
except ValueError:
    idx = 0
selected_name = st.sidebar.selectbox(
    "Tipo de imagen actual",
    options=type_options,
    index=idx,
)
st.session_state.current_type = selected_name  # guardamos string

# Carga de imágenes (widgets con clave estable basada en string)
st.sidebar.subheader("Subir imágenes")
for t in ImageType:
    upl = st.sidebar.file_uploader(
        f"{t.value}", type=["png", "jpg", "jpeg", "bmp"], key=f"upl_{t.name}"
    )
    if upl is not None:
        st.session_state.images[t.name] = file_to_bgr(upl)  # <-- string


# Puerta de entrada: exigir las 3 imágenes
missing = [t for t in ImageType if t.name not in st.session_state.images]
if missing:
    faltan = ", ".join([t.value for t in missing])
    st.sidebar.error(f"Faltan por subir: {faltan}")
    st.title("Detección de características SIFT")
    st.info("Sube las tres imágenes (LIBRE, MEDICA, TELEMETRICA) en el sidebar para continuar.")
    st.stop()

# Parámetros SIFT (solo si hay 3 imágenes)
st.sidebar.subheader("SIFT")
nfeatures = st.sidebar.slider("nfeatures", 0, 10000, int(st.session_state.sift["nfeatures"]), 50)
nOctaveLayers = st.sidebar.slider("nOctaveLayers", 1, 10, int(st.session_state.sift["nOctaveLayers"]), 1)
contrast = st.sidebar.slider("contrastThreshold", 0.001, 0.1, float(st.session_state.sift["contrastThreshold"]), 0.001)
edge = st.sidebar.slider("edgeThreshold", 1.0, 100.0, float(st.session_state.sift["edgeThreshold"]), 1.0)
sigma = st.sidebar.slider("sigma", 0.5, 5.0, float(st.session_state.sift["sigma"]), 0.05)

if st.sidebar.button("Guardar parámetros SIFT"):
    st.session_state.sift = dict(
        nfeatures=int(nfeatures),
        nOctaveLayers=int(nOctaveLayers),
        contrastThreshold=float(contrast),
        edgeThreshold=float(edge),
        sigma=float(sigma),
    )
if st.sidebar.button("Restaurar valores por defecto"):
    st.session_state.sift = DEFAULT_SIFT.copy()

# ----------------------------
# Tabs principales
# ----------------------------
st.title("Detección de características SIFT")
tab1, tab2, tab3 = st.tabs(["Características SIFT", "Seleccionar ROI", "Detectar ROI"])

# ----------------------------
# Tab 1: Características SIFT
# ----------------------------
with tab1:
    st.subheader("Vista de keypoints SIFT")
    cname = current_name()              # 'LIBRE'
    img = get_img(cname)                # dict por string
    if img is None:
        st.info("Sube la imagen del tipo actual en el sidebar.")
    else:
        gray = to_gray(img)
        (kp, des), _ = extract_sift(gray, st.session_state.sift)
        vis = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        st.write(f"Keypoints detectados: {len(kp)}")
        st.image(bgr_to_rgb(vis), caption=f"{cname}: {len(kp)} puntos", use_container_width=True)


# ----------------------------
# Tab 2: Seleccionar ROI (2 modos: cropper y coordenadas)
# ----------------------------
try:
    from streamlit_cropper import st_cropper
    HAS_CROPPER = True
except Exception:
    HAS_CROPPER = False

with tab2:
    st.subheader("Definir ROI manualmente")
    cname = current_name()               # 'LIBRE'
    img = get_img(cname)                 # dict por string

    if img is None:
        st.info("Sube la imagen del tipo actual en el sidebar.")
        st.stop()

    rgb = bgr_to_rgb(img)
    h, w = rgb.shape[:2]

    st.caption("Previsualización de la imagen cargada:")
    st.image(rgb, use_container_width=True)

    modo = st.radio(
        "Modo de selección",
        options=["Arrastrar (cropper)", "Coordenadas"],
        help="Elige cómo quieres indicar la ROI"
    )

    roi_img, bbox = None, None

    # --- MODO 1: Arrastrar (cropper) ---
    if modo == "Arrastrar (cropper)":
        if not HAS_CROPPER:
            st.error("Instala streamlit-cropper: pip install streamlit-cropper")
        else:
            pil = Image.fromarray(rgb)
            cropped = st_cropper(
                pil,
                box_color="red",
                realtime_update=True,
                aspect_ratio=None,
                return_type="image",
            )
            if st.button("Guardar ROI"):
                if cropped is not None:
                    roi_bgr = np.array(cropped)[:, :, ::-1]  # RGB->BGR
                    h2, w2 = roi_bgr.shape[:2]
                    roi_img = roi_bgr
                    bbox = (0, 0, w2, h2)  # ROI ya recortada
                else:
                    st.error("No se obtuvo recorte válido.")

    # --- MODO 2: Coordenadas numéricas ---
    elif modo == "Coordenadas":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            x = st.number_input("x", min_value=0, max_value=max(0, w-1), value=0, step=1)
        with c2:
            y = st.number_input("y", min_value=0, max_value=max(0, h-1), value=0, step=1)
        with c3:
            ww = st.number_input("ancho", min_value=1, max_value=w, value=min(100, w), step=1)
        with c4:
            hh = st.number_input("alto", min_value=1, max_value=h, value=min(100, h), step=1)

        if st.button("Guardar ROI"):
            x = int(norm_int(x, 0, w-1))
            y = int(norm_int(y, 0, h-1))
            ww = int(norm_int(ww, 1, w - x))
            hh = int(norm_int(hh, 1, h - y))
            bbox = (x, y, ww, hh)
            roi_img = img[y:y+hh, x:x+ww]

    # --- Guardado y visualización final comunes ---
    if roi_img is not None and bbox is not None:
        st.session_state.rois[cname] = dict(image=roi_img, bbox=bbox)
        st.image(bgr_to_rgb(roi_img), caption="ROI", use_container_width=False)

    if get_roi(cname):
        st.caption("ROI actual:")
        st.image(bgr_to_rgb(get_roi(cname)["image"]), use_container_width=False)




# ----------------------------
# Tab 3: Detectar ROI
# ----------------------------
with tab3:
    st.subheader("Detección de ROI en transformaciones y deformaciones")

    # Tipos disponibles: aquellos cuyo nombre está en ambos diccionarios
    tipos_disp = [t for t in ImageType if (t.name in st.session_state.images) and (t.name in st.session_state.rois)]
    if not tipos_disp:
        st.info("Necesitas al menos una imagen subida y su ROI guardada.")
    else:
        tsel = st.selectbox("Tipo a evaluar", options=tipos_disp, format_func=lambda t: t.value)
        base = st.session_state.images[tsel.name]
        roi  = st.session_state.rois[tsel.name]["image"]
        # Opciones de transformaciones
        st.markdown("#### Transformaciones a generar")
        c1, c2, c3 = st.columns(3)
        with c1:
            rot30 = st.checkbox("Rotación +30°", value=True)
            rotm45 = st.checkbox("Rotación -45°", value=True)
            esc15 = st.checkbox("Escala 1.5×", value=True)
            esc07 = st.checkbox("Escala 0.7×", value=True)
        with c2:
            tr1 = st.checkbox("Traslación (50,100)", value=True)
            tr2 = st.checkbox("Traslación (-30,80)", value=True)
            per_l = st.checkbox("Perspectiva leve", value=True)
            per_m = st.checkbox("Perspectiva moderada", value=True)
        with c3:
            per_f = st.checkbox("Perspectiva fuerte", value=True)
            def_b = st.checkbox("Deformación barril", value=True)
            def_c = st.checkbox("Deformación cojín", value=True)

        modo = st.selectbox("Modo de detección", options=["rigido", "deformacion"])
        min_matches = st.slider("Mínimo de coincidencias", 4, 40, 10 if modo == "rigido" else 8, 1)
        ransac = st.slider("Umbral RANSAC (px)", 1, 15, 5 if modo == "rigido" else 8, 1)
        ratio = st.slider("Ratio test", 0.55, 0.95, 0.70 if modo == "rigido" else 0.80, 0.01)

        run = st.button("Ejecutar detección")
        if run:
            roi_gray = to_gray(roi)
            (kp_roi, des_roi), sift = extract_sift(roi_gray, st.session_state.sift)
            if des_roi is None or len(kp_roi) == 0:
                st.error("La ROI no tiene descriptores con los parámetros SIFT actuales. Ajusta SIFT en el sidebar.")
            else:
                # Generamos lista de transformaciones
                todo = []
                if rot30: todo.append(("rotacion_30", rotate_image(base, 30)))
                if rotm45: todo.append(("rotacion_-45", rotate_image(base, -45)))
                if esc15: todo.append(("escala_1.5", scale_image(base, 1.5)))
                if esc07: todo.append(("escala_0.7", scale_image(base, 0.7)))
                if tr1:   todo.append(("traslacion_50_100", translate_image(base, 50, 100)))
                if tr2:   todo.append(("traslacion_-30_80", translate_image(base, -30, 80)))
                if per_l: todo.append(("perspectiva_leve", perspective_transform(base, "leve")))
                if per_m: todo.append(("perspectiva_moderada", perspective_transform(base, "moderada")))
                if per_f: todo.append(("perspectiva_fuerte", perspective_transform(base, "fuerte")))
                if def_b: todo.append(("deformacion_barril", deform_barrel(base, 0.00001)))
                if def_c: todo.append(("deformacion_cojin", deform_pincushion(base, -0.00001)))

                if not todo:
                    st.warning("Selecciona al menos una transformación.")
                else:
                    results = []
                    imgs_out = []
                    prefer_affine = (modo == "deformacion")
                    crosscheck = (modo == "deformacion")
                    for name, img_t in todo:
                        gray_t = to_gray(img_t)
                        kp_t, des_t = sift.detectAndCompute(gray_t, None)
                        status = "NO DETECTADA"
                        good_n = 0
                        inliers = 0
                        kind = None
                        poly = None
                        vis = img_t.copy()
                        if des_t is not None and len(kp_t) > 1:
                            good = match_flann(des_roi, des_t, ratio=ratio, crosscheck=crosscheck)
                            good_n = len(good)
                            if good_n >= min_matches:
                                M, mask, inl, kind = estimate_geom(kp_roi, kp_t, good, ransac_thresh=ransac, prefer_affine=prefer_affine)
                                inliers = inl
                                if M is not None and inliers >= 4:
                                    poly = project_box(M, kind, roi_gray.shape[:2])
                                    if is_valid_quad(poly, img_t.shape):
                                        cv.polylines(vis, [poly.reshape(-1, 1, 2)], True, (0, 0, 255), 3, cv.LINE_AA)
                                        cv.putText(vis, f"ROI DETECTADA ({kind}, good={good_n}, inliers={inliers})",
                                                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv.LINE_AA)
                                        status = "DETECTADA"
                                    else:
                                        vis = draw_text(vis, "Homografía/Afín inválida", (10, 30), (0, 165, 255))
                                else:
                                    vis = draw_text(vis, "Homografía/Afín no estimable", (10, 30), (0, 0, 255))
                            else:
                                vis = draw_text(vis, f"Pocas coincidencias ({good_n})", (10, 30), (0, 0, 255))
                        else:
                            vis = draw_text(vis, "Sin keypoints en imagen", (10, 30), (0, 0, 255))

                        results.append(dict(nombre=name, status=status, good=good_n, inliers=inliers, modelo=kind or "-"))
                        imgs_out.append((name, vis))

                    # Mostrar resultados
                    st.markdown("#### Resultados")
                    for name, vis in imgs_out:
                        st.image(bgr_to_rgb(vis), caption=name, use_container_width=True)

                    # Tabla
                    st.markdown("#### Resumen")
                    import pandas as pd
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)

                    # Descarga ZIP
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for name, vis in imgs_out:
                            _, png = cv.imencode(".png", vis[:, :, ::-1])
                            zf.writestr(f"{tsel.value}_{name}.png", png.tobytes())
                        zf.writestr("resumen.csv", df.to_csv(index=False).encode("utf-8"))
                    st.download_button("Descargar resultados (ZIP)", data=buf.getvalue(), file_name=f"resultados_{tsel.value}.zip", mime="application/zip")
