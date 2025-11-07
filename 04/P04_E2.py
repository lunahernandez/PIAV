import io
import numpy as np
import streamlit as st
import soundfile as sf
from scipy.io import wavfile

from utils import (
    to_mono, normalize, make_note, process_make_noisy_original,
    plot_signals_like_example
)


st.set_page_config(page_title="Filtrado de Audio", layout="wide")
st.title("Filtrado de Señales de Audio")

st.sidebar.header("Fuente de audio")
src = st.sidebar.selectbox("Origen", ["Nota sintética", "Cargar WAV", "Micrófono"])

if src != "Micrófono":
    fs = st.sidebar.number_input("Fs (Hz)", 8000, 48000, value=44100, step=1000)
else:
    fs = 48000


# Parámetros de fuente de audio
if src == "Nota sintética":
    f0 = st.sidebar.slider("Frecuencia de la nota (Hz)", 100, 2000, 440, 1)
    duration = st.sidebar.slider("Duración (s)", 0.5, 10.0, 2.0, 0.1)
    harmonics = st.sidebar.slider("Armónicos", 0, 10, 3, 1)
    add_noise_flag = st.sidebar.checkbox("Añadir ruido y usarla como 'Original'", value=True)
    snr_db = st.sidebar.slider("SNR (dB) del ruido", -10, 40, 10, 1) if add_noise_flag else None

elif src == "Cargar WAV":
    uploaded_file = st.sidebar.file_uploader("Archivo .wav", type=["wav"])
    add_noise_flag = st.sidebar.checkbox("Añadir ruido adicional al WAV", value=False)
    snr_db = st.sidebar.slider("SNR (dB) del ruido adicional", -10, 40, 10, 1) if add_noise_flag else None

else:
    add_noise_flag = st.sidebar.checkbox("Añadir ruido adicional al micrófono", value=False)
    snr_db = st.sidebar.slider("SNR (dB) del ruido adicional", -10, 40, 20, 1) if add_noise_flag else None


# Parámetros de filtro
st.sidebar.header("Filtro")
filter_type = st.sidebar.selectbox("Tipo de Filtro", ["Pasa-Bajo", "Pasa-Alto", "Pasa-Banda", "Rechaza-Banda"])
impl = st.sidebar.selectbox(
    "Implementación",
    [
        "Butterworth (lfilter)", "Butterworth (filtfilt)",
        "Chebyshev I (lfilter)", "Chebyshev I (filtfilt)",
        "Chebyshev II (lfilter)", "Chebyshev II (filtfilt)",
        "Elíptico (lfilter)", "Elíptico (filtfilt)"
    ]
)

cutoff1 = st.sidebar.slider("Frecuencia de Corte 1 (Hz)", 20, int(fs/2)-20, 1000, 10)
cutoff2 = None
if filter_type in ("Pasa-Banda", "Rechaza-Banda"):
    cutoff2 = st.sidebar.slider("Frecuencia de Corte 2 (Hz)", 20, int(fs/2)-20, 3000, 10)

# Parámetros de implementación
if "Butterworth" in impl:
    order = st.sidebar.slider("Orden", 1, 10, 5)
    rp = 1.0
    rs = 40.0
elif "Chebyshev I" in impl:
    col1, col2 = st.sidebar.columns(2)
    order = col1.slider("Orden", 1, 10, 5)
    rp = col2.slider("rp (dB) - Banda de paso", 0.1, 5.0, 1.0, 0.1)
    rs = 40.0
elif "Chebyshev II" in impl:
    col1, col2 = st.sidebar.columns(2)
    order = col1.slider("Orden", 1, 10, 5)
    rs = col2.slider("rs (dB) - Banda de parada", 20, 80, 40, 5)
    rp = 1.0
elif "Elíptico" in impl:
    col1, col2, col3 = st.sidebar.columns(3)
    order = col1.slider("Orden", 1, 10, 5)
    rp = col2.slider("rp (dB) - Banda de paso", 0.1, 5.0, 1.0, 0.1)
    rs = col3.slider("rs (dB) - Banda de parada", 20, 100, 40, 5)


# Flujo por fuente de audio
if src != "Micrófono":
    if src == "Nota sintética":
        clean = make_note(fs, f0, duration, harmonics)
        original, filtered = process_make_noisy_original(
            clean, fs, add_noise_flag, snr_db,
            filter_type, impl, cutoff1, cutoff2, order, rp, rs
        )
    else:
        if uploaded_file is None:
            st.info("Carga un archivo WAV para comenzar.")
            st.stop()
        fs_file, data = wavfile.read(uploaded_file)
        data = normalize(to_mono(data))
        fs = fs_file # Actualizar fs al del archivo cargado
        original, filtered = process_make_noisy_original(
            data, fs, add_noise_flag, snr_db,
            filter_type, impl, cutoff1, cutoff2, order, rp, rs
        )

    st.subheader("Visualización de señales")
    fig = plot_signals_like_example(original, filtered, fs)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Reproducción de Audio")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Original")
        b1 = io.BytesIO()
        wavfile.write(b1, fs, (original * 32767).astype(np.int16))
        st.audio(b1.getvalue(), format="audio/wav")
    with c2:
        st.write("Filtrada")
        b2 = io.BytesIO()
        wavfile.write(b2, fs, (filtered * 32767).astype(np.int16))
        st.audio(b2.getvalue(), format="audio/wav")

    st.download_button("Descargar audio filtrado", b2.getvalue(), file_name="audio_filtrado.wav", mime="audio/wav")

else:
    from st_audiorec import st_audiorec

    st.subheader("Micrófono en vivo")
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.success("Audio grabado")

        audio_array, fs_real = sf.read(io.BytesIO(wav_audio_data), dtype='int16')
        audio_array = to_mono(audio_array)
        audio_original = audio_array / 32768.0

        try:
            audio_noisy, audio_filtered = process_make_noisy_original(
                audio_original, fs_real, add_noise_flag, snr_db,
                filter_type, impl, cutoff1, cutoff2, order, rp, rs
            )
        except Exception as e:
            st.error(f"Error al aplicar el filtro: {e}")
            audio_noisy = audio_original
            audio_filtered = audio_original

        buffer_original = io.BytesIO()
        buffer_filtered = io.BytesIO()
        wavfile.write(buffer_original, fs_real, (audio_noisy * 32767).astype(np.int16))
        wavfile.write(buffer_filtered, fs_real, (audio_filtered * 32767).astype(np.int16))

        st.subheader("Visualización de las señales")
        fig = plot_signals_like_example(audio_noisy, audio_filtered, fs_real)
        st.pyplot(fig, clear_figure=True)

        st.subheader("Reproducción de Audio")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original")
            st.audio(buffer_original.getvalue(), format="audio/wav")
        with col2:
            st.write("Filtrada")
            st.audio(buffer_filtered.getvalue(), format="audio/wav")

        duration = len(audio_filtered) / fs_real
        st.caption(f"Frecuencia real detectada: {fs_real} Hz | Duración: {duration:.2f} s")

        st.download_button(
            label="Descargar audio filtrado",
            data=buffer_filtered.getvalue(),
            file_name="audio_filtrado.wav",
            mime="audio/wav"
        )
