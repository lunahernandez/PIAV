# app.py
import io
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import cv2 as cv

from utils import (
    DEFAULT_SIFT,
    file_to_bgr, bgr_to_rgb, to_gray, draw_text, norm_int,
    rotate_image, scale_image, translate_image, perspective_transform,
    deform_barrel, deform_pincushion,
    extract_sift, match_flann, estimate_geom, project_box, is_valid_quad,
    make_canvas_centered, warp_affine_on_canvas, apply_transform_spec, match_bf_crosscheck,
    draw_matches_panel, to_gray, bgr_to_rgb, match_knn_ratio,
    affine_matrix_RS,          # NUEVA
    affine_update_t_for_pivot, # NUEVA
    affine_build_2x3,          # NUEVA
    affine_preview_on_canvas,
    apply_distortion_full
)

# ----------------------------
# Estado de sesión (solo UI)
# ----------------------------
def ensure_ss():
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "sift" not in st.session_state:
        st.session_state.sift = DEFAULT_SIFT.copy()
    if "roi_data" not in st.session_state:
        st.session_state.roi_data = None
    if "roi_saved" not in st.session_state:
        st.session_state.roi_saved = False
    if "transform_specs" not in st.session_state:
        st.session_state.transform_specs = []  # lista de specs personalizados

ensure_ss()

# ----------------------------
# Sidebar: carga y SIFT
# ----------------------------
st.sidebar.title("Parámetros y datos")

upl = st.sidebar.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg", "bmp"])
if upl is not None:
    new_img = file_to_bgr(upl)
    if st.session_state.current_image is None or not np.array_equal(new_img, st.session_state.current_image):
        st.session_state.current_image = new_img
        st.session_state.roi_data = None
        st.session_state.roi_saved = False

if st.session_state.roi_data is not None:
    st.sidebar.success("ROI guardado")
    bbox = st.session_state.roi_data["bbox"]
    st.sidebar.caption(f"Tamaño: {bbox[2]}x{bbox[3]} px")
else:
    st.sidebar.info("Sin ROI guardado")

st.sidebar.subheader("Parámetros SIFT")
nfeatures = st.sidebar.slider("nfeatures", 0, 10000, int(st.session_state.sift["nfeatures"]), 50)
nOctaveLayers = st.sidebar.slider("nOctaveLayers", 1, 10, int(st.session_state.sift["nOctaveLayers"]), 1)
contrast = st.sidebar.slider("contrastThreshold", 0.001, 0.1, float(st.session_state.sift["contrastThreshold"]), 0.001)
edge = st.sidebar.slider("edgeThreshold", 1.0, 100.0, float(st.session_state.sift["edgeThreshold"]), 1.0)
sigma = st.sidebar.slider("sigma", 0.5, 5.0, float(st.session_state.sift["sigma"]), 0.05)

colA, colB = st.sidebar.columns(2)
with colA:
    if st.sidebar.button("Guardar SIFT"):
        st.session_state.sift = dict(
            nfeatures=int(nfeatures),
            nOctaveLayers=int(nOctaveLayers),
            contrastThreshold=float(contrast),
            edgeThreshold=float(edge),
            sigma=float(sigma),
        )
        st.sidebar.success("Parámetros guardados")
with colB:
    if st.sidebar.button("Restaurar SIFT"):
        st.session_state.sift = DEFAULT_SIFT.copy()
        st.rerun()

# ----------------------------
# Tabs
# ----------------------------
st.title("Detección de características SIFT")
tab1, tab2, tab3, tab4 = st.tabs([
    "Características SIFT",
    "Seleccionar ROI",
    "Detectar ROI",
    "Transformaciones personalizadas"
])

# ----------------------------
# Tab 1: Características
# ----------------------------
with tab1:
    st.subheader("Vista de keypoints SIFT")
    img = st.session_state.current_image
    if img is None:
        st.info("Sube una imagen en el sidebar.")
    else:
        gray = to_gray(img)
        (kp, des), _ = extract_sift(gray, st.session_state.sift)
        vis = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        st.image(bgr_to_rgb(vis), caption=f"Keypoints detectados: {len(kp)}", use_container_width=True)

# ----------------------------
# Tab 2: Selección de ROI
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

        st.write("Selecciona una región de interés:")
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
                                st.rerun()
                            else:
                                st.error("El ROI está vacío.")
                        else:
                            st.error("No se obtuvo un recorte válido. Intenta de nuevo.")
                    except Exception as e:
                        st.error(f"Error al guardar ROI: {str(e)}")

# ----------------------------
# Tab 3: Detección ROI (solo BFMatcher)
# ----------------------------
with tab3:
    st.subheader("Detección de ROI")

    if st.session_state.current_image is None:
        st.info("Sube una imagen en el sidebar.")
    elif st.session_state.roi_data is None:
        st.warning("Primero define una ROI en la pestaña 'Seleccionar ROI'.")
    else:
        base = st.session_state.current_image
        roi = st.session_state.roi_data["image"]

        fuente = st.radio(
            "Fuente de transformaciones",
            options=["Predefinidas", "Personalizadas"],
            horizontal=True
        )

        if fuente == "Predefinidas":
            st.markdown("Transformaciones a generar")
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
        else:
            if len(st.session_state.transform_specs) == 0:
                st.warning("No hay transformaciones personalizadas definidas.")
            else:
                st.success(f"Usando {len(st.session_state.transform_specs)} transformaciones personalizadas.")

        st.markdown("---")
        st.markdown("Parámetros de detección")

        # --- Emparejador ---
        matcher = st.selectbox("Emparejador", ["BF (crossCheck)", "KNN (ratio test)"], index=0)
        ratio = None
        if matcher == "KNN (ratio test)":
            ratio = st.slider("Ratio test (Lowe)", 0.55, 0.95, 0.75, 0.01)

        col1, col2 = st.columns(2)
        with col1:
            min_matches = st.slider("Mínimo de coincidencias", 3, 40, 5, 1)
        with col2:
            ransac = st.slider("Umbral RANSAC (px)", 1, 15, 5, 1)

        topN = st.slider("Top N líneas de matches a dibujar", 5, 300, 20, 5)
        run = st.button("Ejecutar detección", type="primary")

        if run:
            roi_gray = to_gray(roi)
            (kp_roi, des_roi), sift = extract_sift(roi_gray, st.session_state.sift)

            if des_roi is None or len(kp_roi) == 0:
                st.error("La ROI no tiene descriptores con los parámetros SIFT actuales.")
            else:
                todo = []
                if fuente == "Predefinidas":
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
                else:
                    for i, spec in enumerate(st.session_state.transform_specs):
                        try:
                            name, img_t = apply_transform_spec(base, spec)
                            todo.append((f"{i:02d}_{name}", img_t))
                        except Exception as e:
                            st.warning(f"Error en spec {i}: {e}")

                if not todo:
                    st.warning("Selecciona al menos una transformación.")
                else:
                    results = []
                    imgs_out = []

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

                        if des_t is not None and len(kp_t) > 1:
                            if matcher == "BF (crossCheck)":
                                matches = match_bf_crosscheck(des_roi, des_t)
                            else:
                                matches = match_knn_ratio(des_roi, des_t, ratio=ratio)

                            good_n = len(matches)

                            if good_n >= min_matches:
                                M, mask, inl, kind = estimate_geom(kp_roi, kp_t, matches, ransac_thresh=ransac)
                                inliers = inl
                                if M is not None and inliers >= 4:
                                    poly = project_box(M, kind, roi_gray.shape[:2])
                                    if is_valid_quad(poly, img_t.shape):
                                        status = "DETECTADA"
                                        vis = draw_matches_panel(roi, img_t, kp_roi, kp_t, matches, topN=topN, poly_right=poly)
                                    else:
                                        vis = draw_matches_panel(roi, img_t, kp_roi, kp_t, matches, topN=topN,
                                                                 banner=("Homografia/Afin invalida", (0,165,255)))
                                else:
                                    vis = draw_matches_panel(roi, img_t, kp_roi, kp_t, matches, topN=topN,
                                                             banner=("Homografia/Afin no estimable", (0,0,255)))
                            else:
                                vis = draw_matches_panel(roi, img_t, kp_roi, kp_t, matches, topN=topN,
                                                         banner=(f"Pocas coincidencias ({good_n})", (0,0,255)))
                        else:
                            vis = draw_matches_panel(roi, img_t, [], [], [], topN=0,
                                                     banner=("Sin keypoints en imagen", (0,0,255)))

                        results.append(dict(nombre=name, status=status, good=good_n, inliers=inliers, modelo=kind or "-"))
                        imgs_out.append((name, vis))
                        progress_bar.progress((idx + 1) / len(todo))

                    status_text.text("Procesamiento completado")
                    progress_bar.empty()

                    st.markdown("---")
                    st.markdown("Resultados visuales (ROI a la izquierda, imagen transformada a la derecha)")
                    for name, vis in imgs_out:
                        st.image(bgr_to_rgb(vis), caption=name, use_container_width=True)

                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)

                    detectadas = sum(1 for r in results if r["status"] == "DETECTADA")
                    total = len(results)
                    st.metric("Tasa de detección", f"{detectadas}/{total} ({100*detectadas/total:.1f}%)")

                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for name, vis in imgs_out:
                            _, png = cv.imencode(".png", vis[:, :, ::-1])
                            zf.writestr(f"{name}.png", png.tobytes())
                        zf.writestr("resumen.csv", df.to_csv(index=False).encode("utf-8"))

                    st.download_button(
                        "Descargar resultados (ZIP)",
                        data=buf.getvalue(),
                        file_name="resultados_deteccion.zip",
                        mime="application/zip"
                    )

# ----------------------------
# Tab 4: Transformaciones personalizadas
# ----------------------------
# ----------------------------
# Tab 4: Transformaciones personalizadas
# ----------------------------
from utils import (
    make_canvas_centered,
    affine_matrix_RS,
    affine_update_t_for_pivot,
    affine_build_2x3,
    affine_preview_on_canvas,
    apply_distortion_full,
    bgr_to_rgb,
)
import numpy as np
import pandas as pd
import cv2 as cv
import streamlit as st

with tab4:
    st.subheader("Diseñador de transformaciones")
    if st.session_state.current_image is None:
        st.info("Sube una imagen en el sidebar.")
    else:
        img = st.session_state.current_image
        h, w = img.shape[:2]

        mode = st.radio(
            "Tipo de transformación",
            options=["Afin (Tx,Ty,Ángulo,Escala,Centro)", "Distorsión (k1,k2,p1,p2,k3)"],
            horizontal=True
        )

        # =====================================================================
        # --------------------------- AFIN -----------------------------------
        # =====================================================================
        if "Afin" in mode:
            st.markdown("Parámetros de afin con pivote y compensación exacta (b = t + (I - A)·c)")

            col = st.columns(3)
            with col[0]:
                sf = st.slider("Factor lienzo", 1.2, 4.0, 2.0, 0.1)
                ang = st.slider("Ángulo (°)", 0, 360, 0, 1)
                sx = st.slider("Escala X", 0.10, 3.00, 1.00, 0.01)
                uniform = st.checkbox("Escalado uniforme", value=False)
            with col[1]:
                if uniform:
                    sy = sx
                    st.write(f"Escala Y = {sy:.2f} (uniforme)")
                else:
                    sy = st.slider("Escala Y", 0.10, 3.00, 1.00, 0.01)

                CW, CH = int(w * sf), int(h * sf)

                # --- Estado inicial si no existe ---
                if "tab4_t" not in st.session_state:
                    st.session_state.tab4_t = np.array([0.0, 0.0], np.float32)
                if "tab4_prev_c" not in st.session_state:
                    st.session_state.tab4_prev_c = np.array([CW // 2, CH // 2], np.float32)

                t_state = st.session_state.tab4_t.astype(np.float32)
                prev_c = st.session_state.tab4_prev_c.astype(np.float32)
            with col[2]:
                cx_ui = st.slider("Centro Cx (px lienzo)", 0, CW, CW // 2, 1)
                cy_ui = st.slider("Centro Cy (px lienzo)", 0, CH, CH // 2, 1)
                name = st.text_input("Nombre", value=f"affine_{ang}deg")

            # --- Matriz A y centro actual ---
            A, I = affine_matrix_RS(angle_deg=float(ang), sx=float(sx), sy=float(sy))
            c_now = np.array([float(cx_ui), float(cy_ui)], np.float32)

            # --- Compensar cambio de pivote antes de crear sliders ---
            if not np.allclose(c_now, prev_c):
                t_state = affine_update_t_for_pivot(t_state, prev_c, c_now, A, I)
                st.session_state.tab4_t = t_state.copy()
                st.session_state.tab4_prev_c = c_now.copy()

            # --- Sliders de traslación (sin key para evitar conflictos) ---
            tx_ui = st.slider(
                "Tx (px lienzo)",
                -CW // 2, CW // 2,
                value=int(round(t_state[0])),
                step=1
            )
            ty_ui = st.slider(
                "Ty (px lienzo)",
                -CH // 2, CH // 2,
                value=int(round(t_state[1])),
                step=1
            )

            # --- Si el usuario mueve los sliders, actualizar t_state ---
            if (tx_ui != int(round(t_state[0]))) or (ty_ui != int(round(t_state[1]))):
                t_state = np.array([float(tx_ui), float(ty_ui)], np.float32)
                st.session_state.tab4_t = t_state.copy()

            # --- Construcción de la matriz completa ---
            M, b = affine_build_2x3(A=A, t=t_state, c=c_now, I=I)

            # --- Vista previa ---
            view, pc, CW, CH, ox, oy = affine_preview_on_canvas(
                img=img, scale_factor=sf, M=M, A=A, b=b, c_now=c_now
            )

            # Dibujar pivote en la vista
            disp = view.copy()
            cv.circle(disp, (int(pc[0]-ox), int(pc[1]-oy)), 6, (0,0,255), -1)
            cv.putText(disp, f"C=({int(cx_ui)},{int(cy_ui)})",
                       (int(pc[0]-ox)+10, int(pc[1]-oy)-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            st.image(bgr_to_rgb(disp), caption="Vista previa (recorte central)", use_container_width=True)

            # --- Botones ---
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Añadir a lista"):
                    st.session_state.transform_specs.append(dict(
                        type="affine", name=name, scale_factor=float(sf),
                        tx=float(t_state[0]), ty=float(t_state[1]),
                        angle=float(ang), cx=float(c_now[0]), cy=float(c_now[1]),
                        sx=float(sx), sy=float(sy)
                    ))
                    st.success("Transformación añadida")
            with c2:
                if st.button("Vaciar lista"):
                    st.session_state.transform_specs = []
                    st.info("Lista vaciada")

        # =====================================================================
        # ------------------------- DISTORSIÓN -------------------------------
        # =====================================================================
        else:
            st.markdown("Parámetros de distorsión radial/tangencial")
            col = st.columns(3)
            with col[0]:
                k1 = st.slider("k1 ×1e-3", -100, 100, 0, 1) / 1000.0
                k2 = st.slider("k2 ×1e-3", -100, 100, 0, 1) / 1000.0
            with col[1]:
                p1 = st.slider("p1 ×1e-3", -100, 100, 0, 1) / 1000.0
                p2 = st.slider("p2 ×1e-3", -100, 100, 0, 1) / 1000.0
            with col[2]:
                k3 = st.slider("k3 ×1e-3", -100, 100, 0, 1) / 1000.0
                focal = st.slider("Focal", 1.0, 50.0, 10.0, 0.5)
            use_center = st.checkbox("Fijar centro manual", value=False)
            if use_center:
                cx = st.slider("Cx (px)", 0, w, w//2, 1)
                cy = st.slider("Cy (px)", 0, h, h//2, 1)
                center = (cx, cy)
            else:
                center = None

            prev = apply_distortion_full(img, k1, k2, p1, p2, k3, center=center, focal=focal)
            st.image(bgr_to_rgb(prev), caption="Vista previa distorsión", use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Nombre", value="distortion_custom")
                if st.button("Añadir a lista"):
                    spec = dict(type="distortion", name=name, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3, focal=focal)
                    if center is not None:
                        spec["cx"], spec["cy"] = center[0], center[1]
                    st.session_state.transform_specs.append(spec)
                    st.success("Transformación añadida")
            with c2:
                if st.button("Vaciar lista"):
                    st.session_state.transform_specs = []
                    st.info("Lista vaciada")

        # =====================================================================
        # --------------------------- LISTA ----------------------------------
        # =====================================================================
        st.markdown("---")
        st.markdown("Transformaciones en la lista")
        if len(st.session_state.transform_specs) == 0:
            st.info("No hay transformaciones añadidas.")
        else:
            df = pd.DataFrame(st.session_state.transform_specs)
            st.dataframe(df, use_container_width=True)
            for i, spec in enumerate(st.session_state.transform_specs):
                c1, c2 = st.columns([8, 1])
                with c1:
                    st.code(str(spec))
                with c2:
                    if st.button("Eliminar", key=f"del_{i}"):
                        st.session_state.transform_specs.pop(i)
                        st.experimental_rerun()
