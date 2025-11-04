import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import (
    butter, cheby1, cheby2, filtfilt, lfilter, firwin
)
from scipy.fft import rfft, rfftfreq
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import queue
import threading
import soundfile as sf

st.set_page_config(page_title="Filtrado de Audio", layout="wide")
st.title("Aplicación de Filtrado de Señales de Audio – Ruidosa vs Filtrada")

# Convierte una señal estéreo a mono promediando sus canales
def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float64)
    return x.mean(axis=1).astype(np.float64)

def normalize(x: np.ndarray) -> np.ndarray:
    mx = np.max(np.abs(x)) if np.max(np.abs(x)) != 0 else 1.0
    return (x / mx).astype(np.float64) # Evita saturación en el audio

def add_white_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    if np.all(x == 0):
        return x.copy()
    x = x.astype(np.float64)
    p_signal = np.mean(x**2) # Potencia de la señal
    p_noise = p_signal / (10**(snr_db/10) ) # Potencia del ruido
    noise = np.random.normal(0, np.sqrt(p_noise), size=x.shape)
    return normalize(x + noise)

def make_note(fs: int, f0: float, dur: float, harmonics: int) -> np.ndarray:
    t = np.linspace(0, dur, int(fs*dur), endpoint=False)
    x = np.sin(2*np.pi*f0*t)
    for k in range(2, harmonics+1):
        x += (1.0/k) * np.sin(2*np.pi*f0*k*t)
    return normalize(x)

# Asegura que las frecuencias de corte estén dentro de límites seguros
def safe_cutoffs(c1, c2, fs):
    nyq = fs * 0.5
    c1 = max(10.0, min(c1, nyq*0.999))
    c2 = c1 if c2 is None else max(10.0, min(c2, nyq*0.999))
    if c2 < c1:
        c1, c2 = c2, c1
    return c1, c2

# Crea los coeficientes del filtro según elección del usuario
def design_filter(filter_kind, impl, fs, cutoff1, cutoff2=None, order=5, rp=1, rs=40, fir_taps=401, window="hamming"):
    nyq = fs*0.5
    w1 = cutoff1/nyq
    w2 = None if cutoff2 is None else cutoff2/nyq

    if impl == "FIR (ventana)":
        btype = {
            "Pasa-Bajo": "lowpass",
            "Pasa-Alto": "highpass",
            "Pasa-Banda": "bandpass",
            "Rechaza-Banda": "bandstop",
        }[filter_kind]
        # firwin genera los coeficientes del filtro FIR
        if btype in ("lowpass", "highpass"):
            taps = firwin(fir_taps, w1, pass_zero=(btype=="lowpass"), window=window)
        else:
            taps = firwin(fir_taps, [w1, w2], pass_zero=(btype=="bandstop"), window=window)
        return taps, np.array([1.0]), "FIR"
    # filtros IRR tipo Butterworth
    if impl in ("Butterworth (filtfilt)", "Butterworth (lfilter)"):
        if filter_kind == "Pasa-Bajo":
            b, a = butter(order, w1, btype="low")
        elif filter_kind == "Pasa-Alto":
            b, a = butter(order, w1, btype="high")
        elif filter_kind == "Pasa-Banda":
            b, a = butter(order, [w1, w2], btype="band")
        else:
            b, a = butter(order, [w1, w2], btype="bandstop")
        return b, a, impl
    # filtros IRR tipo Chebyshev
    if impl == "Chebyshev I (filtfilt)":
        if filter_kind == "Pasa-Bajo":
            b, a = cheby1(order, rp, w1, btype="low")
        elif filter_kind == "Pasa-Alto":
            b, a = cheby1(order, rp, w1, btype="high")
        elif filter_kind == "Pasa-Banda":
            b, a = cheby1(order, rp, [w1, w2], btype="band")
        else:
            b, a = cheby1(order, rp, [w1, w2], btype="bandstop")
        return b, a, impl
    # filtros IRR tipo Chebyshev II
    if impl == "Chebyshev II (filtfilt)":
        if filter_kind == "Pasa-Bajo":
            b, a = cheby2(order, rs, w1, btype="low")
        elif filter_kind == "Pasa-Alto":
            b, a = cheby2(order, rs, w1, btype="high")
        elif filter_kind == "Pasa-Banda":
            b, a = cheby2(order, rs, [w1, w2], btype="band")
        else:
            b, a = cheby2(order, rs, [w1, w2], btype="bandstop")
        return b, a, impl

    raise ValueError("Implementación de filtro no soportada")

# Aplica el filtro diseñado a la señal ruidosa
def apply_filter(noisy, fs, filter_kind, impl, cutoff1, cutoff2, order, rp, rs, fir_taps, window):
    c1, c2 = safe_cutoffs(cutoff1, cutoff2, fs)
    b, a, tag = design_filter(filter_kind, impl, fs, c1, c2, order, rp, rs, fir_taps, window)
    # Distintas funciones de filtrado
    if tag == "FIR":
        y = lfilter(b, a, noisy).astype(np.float64) # FIR es solo con coef b
    elif impl.endswith("(lfilter)"):
        y = lfilter(b, a, noisy).astype(np.float64) # IIR causal
    else:
        y = filtfilt(b, a, noisy).astype(np.float64) # IIR acausal (sin desfase)
    return normalize(y)

# Calcula el espectro de magnitud positivo
def compute_fft_pos(x, fs):
    n = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(n, d=1.0/fs)
    sel = f >= 0
    return f[sel], np.abs(X[sel])

def plot_signals_like_example(original_noisy, filtered, fs):
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    t = np.arange(len(original_noisy)) / fs

    # Señal original (ruidosa)
    axes[0].plot(t, original_noisy, linewidth=0.5)
    axes[0].set_title('Señal Original (Ruidosa) - Dominio Temporal', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tiempo (s)')
    axes[0].set_ylabel('Amplitud')
    axes[0].grid(True, alpha=0.3)

    # Señal filtrada
    axes[1].plot(t, filtered, linewidth=0.5, color='orange')
    axes[1].set_title('Señal Filtrada - Dominio Temporal', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Tiempo (s)')
    axes[1].set_ylabel('Amplitud')
    axes[1].grid(True, alpha=0.3)

    # Espectro original (ruidosa)
    f1, X1 = compute_fft_pos(original_noisy, fs)
    axes[2].plot(f1, X1, linewidth=0.5)
    axes[2].set_title('Señal Original (Ruidosa) - Dominio Frecuencia', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Frecuencia (Hz)')
    axes[2].set_ylabel('Magnitud')
    axes[2].set_xlim([0, fs/2])
    axes[2].grid(True, alpha=0.3)

    # Espectro filtrada
    f2, X2 = compute_fft_pos(filtered, fs)
    axes[3].plot(f2, X2, linewidth=0.5, color='orange')
    axes[3].set_title('Señal Filtrada - Dominio Frecuencia', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Frecuencia (Hz)')
    axes[3].set_ylabel('Magnitud')
    axes[3].set_xlim([0, fs/2])
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


st.sidebar.header("Fuente de audio")
src = st.sidebar.selectbox("Origen", ["Nota sintética", "Cargar WAV", "Micrófono (en vivo)"])

if src != "Micrófono (en vivo)":
    fs = st.sidebar.number_input("Fs (Hz)", 8000, 48000, value=44100, step=1000)
else:
    fs = 48000

if src == "Nota sintética":
    f0 = st.sidebar.slider("Frecuencia de la nota (Hz)", 100, 2000, 440, 1)
    dur = st.sidebar.slider("Duración (s)", 0.5, 10.0, 2.0, 0.1)
    harms = st.sidebar.slider("Armónicos", 0, 10, 3, 1)
    add_noise_flag = st.sidebar.checkbox("Añadir ruido y usarla como 'Original'", value=True)
    if add_noise_flag:
        # SNR es la relación señal a ruido en decibelios
        snr_db = st.sidebar.slider("SNR (dB) del ruido", -10, 40, 10, 1)
    else:
        snr_db = None

elif src == "Cargar WAV":
    uploaded_file = st.sidebar.file_uploader("Archivo .wav", type=["wav"])
    add_noise_flag = st.sidebar.checkbox("Añadir ruido adicional al WAV", value=False)
    if add_noise_flag:
        snr_db = st.sidebar.slider("SNR (dB) del ruido adicional", -10, 40, 10, 1)
    else:
        snr_db = None
# Caso de cuando se usa micrófono en vivo
else:
    add_noise_flag = st.sidebar.checkbox("Añadir ruido adicional al mic", value=False)
    if add_noise_flag:
        snr_db = st.sidebar.slider("SNR (dB) del ruido adicional", -10, 40, 20, 1)
    else:
        snr_db = None

st.sidebar.header("Filtro")
filter_kind = st.sidebar.selectbox("Tipo de Filtro",
    ["Pasa-Bajo", "Pasa-Alto", "Pasa-Banda", "Rechaza-Banda"])

impl = st.sidebar.selectbox("Implementación",
    [
        "Butterworth (lfilter)", 
        "Butterworth (filtfilt)", 
        "Chebyshev I (filtfilt)", 
        "Chebyshev II (filtfilt)", 
        "FIR (ventana)"
        ]
    )

cutoff1 = st.sidebar.slider("Frecuencia de Corte 1 (Hz)", 20, int(fs/2)-20, 1000, 10)
cutoff2 = None
if filter_kind in ("Pasa-Banda", "Rechaza-Banda"):
    cutoff2 = st.sidebar.slider("Frecuencia de Corte 2 (Hz)", 20, int(fs/2)-20, 3000, 10)

colp1, colp2, colp3 = st.sidebar.columns(3)
order = colp1.slider("Orden", 1, 10, 5, 1)
rp = colp2.slider("rp (dB) Cheby I", 0.1, 5.0, 1.0, 0.1)
rs = colp3.slider("rs (dB) Cheby II", 20, 80, 40, 5)

fir_taps = st.sidebar.slider("Taps FIR", 51, 2001, 401, 10)
window = st.sidebar.selectbox("Ventana FIR", ["hamming", "hann", "blackman", "bartlett"])

def process_make_noisy_original(input_signal, fs, add_noise_flag, snr_db):
    if add_noise_flag and snr_db is not None:
        original_noisy = add_white_noise(input_signal, snr_db)
    else:
        original_noisy = input_signal
    filtered = apply_filter(original_noisy, fs, filter_kind, impl, cutoff1, cutoff2, order, rp, rs, fir_taps, window)
    return original_noisy, filtered

if src != "Micrófono (en vivo)":
    if src == "Nota sintética":
        clean = make_note(fs, f0, dur, harms)
        original, filtered = process_make_noisy_original(clean, fs, add_noise_flag, snr_db)
    else:
        if uploaded_file is None:
            st.info("Carga un archivo WAV para comenzar.")
            st.stop()
        fs_file, data = wavfile.read(uploaded_file)
        if data.ndim > 1:
            data = data[:, 0]
        data = normalize(data.astype(np.float64))
        fs = fs_file
        original, filtered = process_make_noisy_original(data, fs, add_noise_flag, snr_db)

    st.subheader("Visualización de señales")
    fig = plot_signals_like_example(original, filtered, fs)
    st.pyplot(fig, clear_figure=True)


    st.subheader("Reproducción de Audio")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Original (ruidosa)")
        b1 = io.BytesIO()
        wavfile.write(b1, fs, (original*32767).astype(np.int16))
        st.audio(b1.getvalue(), format="audio/wav")
    with c2:
        st.write("Filtrada")
        b2 = io.BytesIO()
        wavfile.write(b2, fs, (filtered*32767).astype(np.int16))
        st.audio(b2.getvalue(), format="audio/wav")

    st.download_button("Descargar audio filtrado", b2.getvalue(),
                       file_name="audio_filtrado.wav", mime="audio/wav")

else:
    st.subheader("Micrófono en vivo")

    from st_audiorec import st_audiorec
    import soundfile as sf
    from scipy.io import wavfile

    st.info("Presiona el botón para empezar/detener la grabación")

    # Grabar audio
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.success("Audio grabado")

        # Leer WAV y obtener la frecuencia real del micrófono
        audio_array, fs_real = sf.read(io.BytesIO(wav_audio_data), dtype='int16')
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1).astype(np.int16)  # convertir a mono si es estéreo

        # Normalizar
        audio_original = audio_array.astype(np.float64) / 32768.0

        # Añadir ruido (si está activado)
        if add_noise_flag and snr_db is not None:
            audio_noisy = add_white_noise(audio_original, snr_db)
        else:
            audio_noisy = audio_original

        # Aplicar filtro
        try:
            audio_filtered = apply_filter(
                audio_noisy,
                fs_real, # usa la frecuencia real del micrófono
                filter_kind,
                impl,
                cutoff1,
                cutoff2,
                order,
                rp,
                rs,
                fir_taps,
                window
            )
        except Exception as e:
            st.error(f"Error al aplicar el filtro: {e}")
            audio_filtered = audio_noisy

        # Generar WAVs válidos para reproducir
        buffer_original = io.BytesIO()
        buffer_filtered = io.BytesIO()
        wavfile.write(buffer_original, fs_real, (audio_original * 32767).astype(np.int16))
        wavfile.write(buffer_filtered, fs_real, (audio_filtered * 32767).astype(np.int16))

        # Mostrar visualizaciones
        st.subheader("Visualización de las señales")
        fig = plot_signals_like_example(audio_noisy, audio_filtered, fs_real)
        st.pyplot(fig, clear_figure=True)

        # Reproductores de audio
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.audio(buffer_original.getvalue(), format="audio/wav")

        with col2:
            st.subheader("Con ruido y filtro")
            st.audio(buffer_filtered.getvalue(), format="audio/wav")

        # Mostrar info técnica
        dur = len(audio_filtered) / fs_real
        st.caption(f"Frecuencia real detectada: {fs_real} Hz | Duración: {dur:.2f} s")

        # Botón de descarga
        st.download_button(
            label="Descargar audio filtrado",
            data=buffer_filtered.getvalue(),
            file_name="audio_filtrado.wav",
            mime="audio/wav"
        )
