import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq
import streamlit as st
import io

st.set_page_config(page_title="Filtrado de Audio", layout="wide")

st.title("Aplicación de Filtrado de Señales de Audio")

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a

def apply_filter(data, sample_rate, filter_type, cutoff1, cutoff2, order):
    if filter_type == 'Pasa-Bajo':
        b, a = butter_lowpass(cutoff1, sample_rate, order)
    elif filter_type == 'Pasa-Alto':
        b, a = butter_highpass(cutoff1, sample_rate, order)
    elif filter_type == 'Pasa-Banda':
        b, a = butter_bandpass(cutoff1, cutoff2, sample_rate, order)
    elif filter_type == 'Rechaza-Banda':
        b, a = butter_bandstop(cutoff1, cutoff2, sample_rate, order)
    
    filtered_data = lfilter(b, a, data)
    return filtered_data

def compute_fft(signal, sample_rate):
    n = len(signal)
    fft_vals = fft(signal)
    fft_freq = fftfreq(n, 1/sample_rate)
    
    positive_freq_idx = fft_freq > 0
    return fft_freq[positive_freq_idx], np.abs(fft_vals[positive_freq_idx])

def plot_signals(data, filtered_data, sample_rate):
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    time = np.arange(len(data)) / sample_rate
    
    axes[0].plot(time, data, linewidth=0.5)
    axes[0].set_title('Señal Original - Dominio Temporal', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tiempo (s)')
    axes[0].set_ylabel('Amplitud')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time, filtered_data, linewidth=0.5, color='orange')
    axes[1].set_title('Señal Filtrada - Dominio Temporal', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Tiempo (s)')
    axes[1].set_ylabel('Amplitud')
    axes[1].grid(True, alpha=0.3)
    
    freq_orig, fft_orig = compute_fft(data, sample_rate)
    axes[2].plot(freq_orig, fft_orig, linewidth=0.5)
    axes[2].set_title('Señal Original - Dominio Frecuencia', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Frecuencia (Hz)')
    axes[2].set_ylabel('Magnitud')
    axes[2].set_xlim([0, sample_rate/2])
    axes[2].grid(True, alpha=0.3)
    
    freq_filt, fft_filt = compute_fft(filtered_data, sample_rate)
    axes[3].plot(freq_filt, fft_filt, linewidth=0.5, color='orange')
    axes[3].set_title('Señal Filtrada - Dominio Frecuencia', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Frecuencia (Hz)')
    axes[3].set_ylabel('Magnitud')
    axes[3].set_xlim([0, sample_rate/2])
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

st.sidebar.header("Configuración del Filtro")

uploaded_file = st.sidebar.file_uploader("Cargar archivo de audio (.wav)", type=['wav'])

if uploaded_file is not None:
    sample_rate, data = wavfile.read(uploaded_file)
    
    if len(data.shape) > 1:
        data = data[:, 0]
    
    data = data.astype(np.float32)
    
    st.sidebar.success(f"Audio cargado: {sample_rate} Hz")
    
    filter_type = st.sidebar.selectbox(
        "Tipo de Filtro",
        ['Pasa-Bajo', 'Pasa-Alto', 'Pasa-Banda', 'Rechaza-Banda']
    )
    
    cutoff1 = st.sidebar.slider(
        "Frecuencia de Corte 1 (Hz)",
        min_value=20,
        max_value=int(sample_rate/2),
        value=1000,
        step=10
    )
    
    if filter_type in ['Pasa-Banda', 'Rechaza-Banda']:
        cutoff2 = st.sidebar.slider(
            "Frecuencia de Corte 2 (Hz)",
            min_value=20,
            max_value=int(sample_rate/2),
            value=3000,
            step=10
        )
    else:
        cutoff2 = None
    
    order = st.sidebar.slider(
        "Orden del Filtro",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )
    
    if st.sidebar.button("Aplicar Filtro", type="primary"):
        with st.spinner("Aplicando filtro..."):
            filtered_data = apply_filter(data, sample_rate, filter_type, cutoff1, cutoff2, order)
            
            st.session_state.filtered_data = filtered_data
            st.session_state.original_data = data
            st.session_state.sample_rate = sample_rate
    
    if 'filtered_data' in st.session_state:
        st.subheader("Visualización de Señales")
        
        fig = plot_signals(
            st.session_state.original_data,
            st.session_state.filtered_data,
            st.session_state.sample_rate
        )
        st.pyplot(fig)
        
        st.subheader("Reproducción de Audio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Audio Original (Ruidoso)**")
            original_bytes = io.BytesIO()
            wavfile.write(original_bytes, st.session_state.sample_rate, 
                         st.session_state.original_data.astype(np.int16))
            st.audio(original_bytes, format='audio/wav')
        
        with col2:
            st.write("**Audio Filtrado**")
            filtered_bytes = io.BytesIO()
            wavfile.write(filtered_bytes, st.session_state.sample_rate, 
                         st.session_state.filtered_data.astype(np.int16))
            st.audio(filtered_bytes, format='audio/wav')
        
        st.download_button(
            label="Descargar Audio Filtrado",
            data=filtered_bytes.getvalue(),
            file_name="audio_filtrado.wav",
            mime="audio/wav"
        )
        
        with st.expander("Información Técnica"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frecuencia de Muestreo", f"{st.session_state.sample_rate} Hz")
            with col2:
                st.metric("Duración", f"{len(st.session_state.original_data)/st.session_state.sample_rate:.2f} s")
            with col3:
                st.metric("Muestras", len(st.session_state.original_data))
    
else:
    st.info("Por favor, carga un archivo de audio .wav desde la barra lateral para comenzar.")
    
    st.markdown("""
    ### Instrucciones de Uso
    
    1. **Carga un archivo de audio** en formato .wav (preferiblemente con ruido)
    2. **Selecciona el tipo de filtro** que deseas aplicar
    3. **Ajusta los parámetros** según tus necesidades:
       - Frecuencia(s) de corte
       - Orden del filtro
    4. **Haz clic en "Aplicar Filtro"** para procesar el audio
    5. **Compara** las señales original y filtrada en tiempo y frecuencia
    6. **Reproduce** ambos audios para escuchar la diferencia
    
    ### Tipos de Filtros Disponibles
    
    - **Pasa-Bajo**: Permite pasar frecuencias por debajo del umbral
    - **Pasa-Alto**: Permite pasar frecuencias por encima del umbral
    - **Pasa-Banda**: Permite pasar frecuencias entre dos umbrales
    - **Rechaza-Banda**: Bloquea frecuencias entre dos umbrales
    """)