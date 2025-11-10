# app.py
import io
import zipfile

import numpy as np
from PIL import Image
import streamlit as st

from utils import (
    DEFAULT_SIFT,
    file_to_bgr, bgr_to_rgb, to_gray, draw_text, norm_int,
    rotate_image, scale_image, translate_image, perspective_transform,
    deform_barrel, deform_pincushion,
    extract_sift, match_flann, estimate_geom, project_box, is_valid_quad
)

# ----------------------------
# Estado de sesión (solo UI)
# ----------------------------
def ensure_ss():
    if "current_image" not in st.session_state:
        st.session_state.current_image = None   # np.ndarray (BGR)
    if "sift" not in st.session_state:
        st.session_state.sift = DEFAULT_SIFT.copy()
    if "roi_data" not in st.session_state:
        st.session_state.roi_data = None  # {"image": roi_bgr, "bbox": (x,y,w,h)}
    if "roi_saved" not in st.session_state:
        st.session_state.roi_saved = False

ensure_ss()

# ----------------------------
# Sidebar: carga y SIFT
# ----------------------------
st.sidebar.title("Parámetros y datos")

# Uploader único
upl = st.sidebar.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg", "bmp"])
if upl is not None:
    new_img = file_to_bgr(upl)
    # Solo actualizar si es una imagen diferente
    if st.session_state.current_image is None or not np.array_equal(new_img, st.session_state.current_image):
        st.session_state.current_image = new_img
        # Limpiar ROI al cargar nueva imagen
        st.session_state.roi_data = None
        st.session_state.roi_saved = False

# Mostrar estado del ROI en sidebar
if st.session_state.roi_data is not None:
    st.sidebar.success("ROI guardada")
    bbox = st.session_state.roi_data["bbox"]
    st.sidebar.caption(f"Tamaño: {bbox[2]}x{bbox[3]} px")
else:
    st.sidebar.info("Sin ROI guardada")

# Controles SIFT
st.sidebar.subheader("Parámetros SIFT")
nfeatures = st.sidebar.slider("nfeatures", 0, 10000, int(st.session_state.sift["nfeatures"]), 50)
nOctaveLayers = st.sidebar.slider("nOctaveLayers", 1, 10, int(st.session_state.sift["nOctaveLayers"]), 1)
contrast = st.sidebar.slider("contrastThreshold", 0.001, 0.1, float(st.session_state.sift["contrastThreshold"]), 0.001)
edge = st.sidebar.slider("edgeThreshold", 1.0, 100.0, float(st.session_state.sift["edgeThreshold"]), 1.0)
sigma = st.sidebar.slider("sigma", 0.5, 5.0, float(st.session_state.sift["sigma"]), 0.05)

colA, colB = st.sidebar.columns(2)
with colA:
    if st.sidebar.button("Guardar parámetros SIFT"):
        st.session_state.sift = dict(
            nfeatures=int(nfeatures),
            nOctaveLayers=int(nOctaveLayers),
            contrastThreshold=float(contrast),
            edgeThreshold=float(edge),
            sigma=float(sigma),
        )
        st.sidebar.success("Parámetros guardados")
with colB:
    if st.sidebar.button("Restaurar valores"):
        st.session_state.sift = DEFAULT_SIFT.copy()
        st.rerun()

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
    img = st.session_state.current_image
    if img is None:
        st.info("Sube una imagen en el sidebar.")
    else:
        gray = to_gray(img)
        (kp, des), _ = extract_sift(gray, st.session_state.sift)
        import cv2 as cv
        vis = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        st.image(bgr_to_rgb(vis), caption=f"Keypoints detectados: {len(kp)}", use_container_width=True)

# ----------------------------
# Tab 2: Seleccionar ROI (cropper o coordenadas)
# ----------------------------
try:
    from streamlit_cropper import st_cropper
    HAS_CROPPER = True
except Exception:
    HAS_CROPPER = False

with tab2:
    st.subheader("Definir ROI (Región de Interés)")
    img = st.session_state.current_image
    if img is None:
        st.info("Sube una imagen en el sidebar.")
    else:
        # Mostrar ROI guardada primero
        if st.session_state.roi_data is not None:
            st.success("ROI guardada correctamente")
            col1, col2 = st.columns([2, 1])
            with col1:
                saved_roi = st.session_state.roi_data["image"]
                saved_bbox = st.session_state.roi_data["bbox"]
                st.image(bgr_to_rgb(saved_roi), caption=f"ROI guardada - Posición: {saved_bbox}", use_container_width=True)
            with col2:
                st.metric("Ancho", f"{saved_roi.shape[1]} px")
                st.metric("Alto", f"{saved_roi.shape[0]} px")
                if st.button("Limpiar ROI", key="clear_roi"):
                    st.session_state.roi_data = None
                    st.rerun()
            st.markdown("---")

        rgb = bgr_to_rgb(img)
        h, w = rgb.shape[:2]

        st.write("Selecciona una región de interés en la imagen:")
        modo = st.radio(
            "Modo de selección",
            options=["Coordenadas", "Arrastrar"],
            help="Elige cómo quieres indicar la ROI",
            horizontal=True
        )

        if modo == "Coordenadas":
            st.image(rgb, use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                x = st.number_input("x", min_value=0, max_value=max(0, w-1), value=0, step=1, key="x_coord")
            with c2:
                y = st.number_input("y", min_value=0, max_value=max(0, h-1), value=0, step=1, key="y_coord")
            with c3:
                ww = st.number_input("ancho", min_value=1, max_value=w, value=min(100, w), step=1, key="w_coord")
            with c4:
                hh = st.number_input("alto", min_value=1, max_value=h, value=min(100, h), step=1, key="h_coord")

            if st.button("Guardar ROI", key="save_roi_coords", type="primary"):
                try:
                    x_int = int(norm_int(x, 0, w-1))
                    y_int = int(norm_int(y, 0, h-1))
                    ww_int = int(norm_int(ww, 1, w - x_int))
                    hh_int = int(norm_int(hh, 1, h - y_int))
                    bbox = (x_int, y_int, ww_int, hh_int)
                    roi_img = img[y_int:y_int+hh_int, x_int:x_int+ww_int].copy()

                    if roi_img.size == 0:
                        st.error("El ROI está vacío. Ajusta las coordenadas.")
                    else:
                        st.session_state.roi_data = {"image": roi_img, "bbox": bbox}
                        st.session_state.roi_saved = True
                        st.success(f"ROI guardado: {bbox}")
                        st.balloons()
                        st.rerun()
                except Exception as e:
                    st.error(f"Error al guardar ROI: {str(e)}")

        elif modo == "Arrastrar":
            if not HAS_CROPPER:
                st.error("Instala streamlit-cropper: pip install streamlit-cropper")
                st.info("Mientras tanto, usa el modo 'Coordenadas'")
            else:
                st.write("Arrastra para seleccionar la región:")
                pil = Image.fromarray(rgb)
                cropped = st_cropper(
                    pil,
                    box_color="red",
                    realtime_update=True,
                    aspect_ratio=None,
                    return_type="image",
                )
                if st.button("Guardar ROI", key="save_roi_drag", type="primary"):
                    try:
                        if cropped is not None and cropped.size[0] > 0 and cropped.size[1] > 0:
                            roi_bgr = np.array(cropped)[:, :, ::-1].copy()
                            h2, w2 = roi_bgr.shape[:2]
                            if h2 > 0 and w2 > 0:
                                st.session_state.roi_data = {"image": roi_bgr, "bbox": (0, 0, w2, h2)}
                                st.session_state.roi_saved = True
                                st.success(f"ROI guardado: {w2}x{h2} px")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("El ROI está vacío.")
                        else:
                            st.error("No se obtuvo un recorte válido. Intenta de nuevo.")
                    except Exception as e:
                        st.error(f"Error al guardar ROI: {str(e)}")

# ----------------------------
# Tab 3: Detectar ROI
# ----------------------------
with tab3:
    st.subheader("Detección de ROI en transformaciones y deformaciones")

    if st.session_state.current_image is None:
        st.info("Sube una imagen en el sidebar.")
    elif st.session_state.roi_data is None:
        st.warning("Primero define una ROI en la pestaña 'Seleccionar ROI'.")
    else:
        import cv2 as cv
        base = st.session_state.current_image
        roi = st.session_state.roi_data["image"]

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

        st.markdown("---")
        st.markdown("#### Parámetros de detección")
        modo = st.selectbox("Modo de detección", options=["rigido", "deformacion"])
        col1, col2, col3 = st.columns(3)
        with col1:
            min_matches = st.slider("Mínimo de coincidencias", 4, 40, 10 if modo == "rigido" else 8, 1)
        with col2:
            ransac = st.slider("Umbral RANSAC (px)", 1, 15, 5 if modo == "rigido" else 8, 1)
        with col3:
            ratio = st.slider("Ratio test", 0.55, 0.95, 0.70 if modo == "rigido" else 0.80, 0.01)

        run = st.button("Ejecutar detección", type="primary")

        if run:
            roi_gray = to_gray(roi)
            (kp_roi, des_roi), sift = extract_sift(roi_gray, st.session_state.sift)

            if des_roi is None or len(kp_roi) == 0:
                st.error("La ROI no tiene descriptores con los parámetros SIFT actuales. Ajusta SIFT en el sidebar.")
            else:
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

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, (name, img_t) in enumerate(todo):
                        status_text.text(f"Procesando: {name}...")

                        gray_t = to_gray(img_t)
                        kp_t, des_t = sift.detectAndCompute(gray_t, None)
                        status = "NO DETECTADA"
                        good_n = 0
                        inliers = 0
                        kind = None
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
                                        vis = draw_text(vis, "Homografia/Afin invalida", (10, 30), (0, 165, 255))
                                else:
                                    vis = draw_text(vis, "Homografia/Afin no estimable", (10, 30), (0, 0, 255))
                            else:
                                vis = draw_text(vis, f"Pocas coincidencias ({good_n})", (10, 30), (0, 0, 255))
                        else:
                            vis = draw_text(vis, "Sin keypoints en imagen", (10, 30), (0, 0, 255))

                        results.append(dict(nombre=name, status=status, good=good_n, inliers=inliers, modelo=kind or "-"))
                        imgs_out.append((name, vis))

                        progress_bar.progress((idx + 1) / len(todo))

                    status_text.text("Procesamiento completado")
                    progress_bar.empty()

                    st.markdown("---")
                    st.markdown("#### Resultados visuales")
                    for name, vis in imgs_out:
                        st.image(bgr_to_rgb(vis), caption=name, use_container_width=True)

                    st.markdown("---")
                    st.markdown("#### Resumen de detecciones")
                    import pandas as pd
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)

                    # Estadísticas
                    detectadas = sum(1 for r in results if r["status"] == "DETECTADA")
                    total = len(results)
                    st.metric("Tasa de detección", f"{detectadas}/{total} ({100*detectadas/total:.1f}%)")

                    # Descargar resultados
                    import cv2 as cv2_np
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for name, vis in imgs_out:
                            _, png = cv2_np.imencode(".png", vis[:, :, ::-1])
                            zf.writestr(f"{name}.png", png.tobytes())
                        zf.writestr("resumen.csv", df.to_csv(index=False).encode("utf-8"))

                    st.download_button(
                        "Descargar resultados (ZIP)",
                        data=buf.getvalue(),
                        file_name="resultados_deteccion.zip",
                        mime="application/zip"
                    )
