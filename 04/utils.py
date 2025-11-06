import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, cheby2, ellip, filtfilt, lfilter

def to_mono(x):
    """Convierte señal estéreo a mono promediando canales."""
    if x.ndim == 1:
        return x.astype(np.float64)
    return x.mean(axis=1).astype(np.float64)

def normalize(x):
    """Normaliza por máximo absoluto para evitar saturación."""
    mx = np.max(np.abs(x)) if np.max(np.abs(x)) != 0 else 1.0
    return (x / mx).astype(np.float64)

def add_white_noise(x, signal_to_noise_ratio_db):
    """Añade ruido blanco con SNR (Signal-to-Noise Ratio) especificado en dB."""
    if np.all(x == 0):
        return x.copy()
    x = x.astype(np.float64)
    signal_power = np.mean(x**2)
    noise_power = signal_power / (10**(signal_to_noise_ratio_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=x.shape)
    return normalize(x + noise)

def make_note(fs, f0, duration, harmonics):
    """Genera una nota sintética con armónicos y la normaliza."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    x = np.sin(2 * np.pi * f0 * t)
    for k in range(2, harmonics + 1):
        x += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
    return normalize(x)

def safe_cutoffs(c1, c2, fs):
    """Asegura que las frecuencias de corte estén dentro de límites seguros."""
    nyquist = fs * 0.5
    c1 = max(10.0, min(c1, nyquist * 0.999))
    c2 = c1 if c2 is None else max(10.0, min(c2, nyquist * 0.999))
    if c2 < c1:
        c1, c2 = c2, c1
    return c1, c2

def design_filter(filter_type, impl, fs, cutoff1, cutoff2=None, 
                  order=5, rp=1, rs=40):
    """Diseña un filtro digital y devuelve sus coeficientes."""
    nyquist = fs * 0.5
    normal_cutoff1 = cutoff1 / nyquist
    normal_cutoff2 = None if cutoff2 is None else cutoff2 / nyquist

    # Elíptico (Cauer): lfilter y filtfilt
    if impl in ("Elíptico (filtfilt)", "Elíptico (lfilter)"):
        if filter_type == "Pasa-Bajo":
            b, a = ellip(order, rp, rs, normal_cutoff1, btype="low")
        elif filter_type == "Pasa-Alto":
            b, a = ellip(order, rp, rs, normal_cutoff1, btype="high")
        elif filter_type == "Pasa-Banda":
            b, a = ellip(order, rp, rs, [normal_cutoff1, normal_cutoff2], btype="band")
        else:
            b, a = ellip(order, rp, rs, [normal_cutoff1, normal_cutoff2], btype="bandstop")
        return b, a, impl

    # Butterworth: lfilter y filtfilt
    if impl in ("Butterworth (filtfilt)", "Butterworth (lfilter)"):
        if filter_type == "Pasa-Bajo":
            b, a = butter(order, normal_cutoff1, btype="low")
        elif filter_type == "Pasa-Alto":
            b, a = butter(order, normal_cutoff1, btype="high")
        elif filter_type == "Pasa-Banda":
            b, a = butter(order, [normal_cutoff1, normal_cutoff2], btype="band")
        else:
            b, a = butter(order, [normal_cutoff1, normal_cutoff2], btype="bandstop")
        return b, a, impl

    # Chebyshev I: lfilter y filtfilt
    if impl in ("Chebyshev I (filtfilt)", "Chebyshev I (lfilter)"):
        if filter_type == "Pasa-Bajo":
            b, a = cheby1(order, rp, normal_cutoff1, btype="low")
        elif filter_type == "Pasa-Alto":
            b, a = cheby1(order, rp, normal_cutoff1, btype="high")
        elif filter_type == "Pasa-Banda":
            b, a = cheby1(order, rp, [normal_cutoff1, normal_cutoff2], btype="band")
        else:
            b, a = cheby1(order, rp, [normal_cutoff1, normal_cutoff2], btype="bandstop")
        return b, a, impl

    # Chebyshev II: lfilter y filtfilt
    if impl in ("Chebyshev II (filtfilt)", "Chebyshev II (lfilter)"):
        if filter_type == "Pasa-Bajo":
            b, a = cheby2(order, rs, normal_cutoff1, btype="low")
        elif filter_type == "Pasa-Alto":
            b, a = cheby2(order, rs, normal_cutoff1, btype="high")
        elif filter_type == "Pasa-Banda":
            b, a = cheby2(order, rs, [normal_cutoff1, normal_cutoff2], btype="band")
        else:
            b, a = cheby2(order, rs, [normal_cutoff1, normal_cutoff2], btype="bandstop")
        return b, a, impl

    raise ValueError("Implementación de filtro no soportada")


def apply_filter(noisy, fs, filter_type, impl, cutoff1, cutoff2, 
                 order, rp, rs):
    """Aplica el filtro diseñado a la señal ruidosa y normaliza la salida."""
    c1, c2 = safe_cutoffs(cutoff1, cutoff2, fs)
    b, a, _ = design_filter(filter_type, impl, fs, c1, c2, order, rp, rs)
    if impl.endswith("(lfilter)"):
        y = lfilter(b, a, noisy).astype(np.float64)
    else:
        y = filtfilt(b, a, noisy).astype(np.float64)
    return normalize(y)

def compute_fft_pos(x, fs):
    """Calcula el espectro de magnitud positivo."""
    n = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(n, d=1.0 / fs)
    sel = f >= 0
    return f[sel], np.abs(X[sel])

def plot_signals_like_example(original_noisy, filtered, fs):
    """Grafica las señales original y filtrada en los dominios temporal y de frecuencia."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    t = np.arange(len(original_noisy)) / fs

    axes[0].plot(t, original_noisy, linewidth=0.5)
    axes[0].set_title('Señal Original (Ruidosa) - Dominio Temporal', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tiempo (s)')
    axes[0].set_ylabel('Amplitud')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, filtered, linewidth=0.5, color='orange')
    axes[1].set_title('Señal Filtrada - Dominio Temporal', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Tiempo (s)')
    axes[1].set_ylabel('Amplitud')
    axes[1].grid(True, alpha=0.3)

    f1, X1 = compute_fft_pos(original_noisy, fs)
    axes[2].plot(f1, X1, linewidth=0.5)
    axes[2].set_title('Señal Original (Ruidosa) - Dominio Frecuencia', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Frecuencia (Hz)')
    axes[2].set_ylabel('Magnitud')
    axes[2].set_xlim([0, 2000])
    axes[2].grid(True, alpha=0.3)

    f2, X2 = compute_fft_pos(filtered, fs)
    axes[3].plot(f2, X2, linewidth=0.5, color='orange')
    axes[3].set_title('Señal Filtrada - Dominio Frecuencia', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Frecuencia (Hz)')
    axes[3].set_ylabel('Magnitud')
    axes[3].set_xlim([0, 2000])
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def process_make_noisy_original(input_signal, fs,add_noise_flag, snr_db, 
                                filter_type, impl, cutoff1, cutoff2, 
                                order, rp, rs):
    """Aplica el filtro seleccionado a una señal de entrada. Opcionalmente le añade ruido."""
    if add_noise_flag and snr_db is not None:
        original_noisy = add_white_noise(input_signal, snr_db)
    else:
        original_noisy = input_signal
    filtered = apply_filter(original_noisy, fs, filter_type, impl, cutoff1, cutoff2, order, rp, rs)
    return original_noisy, filtered
