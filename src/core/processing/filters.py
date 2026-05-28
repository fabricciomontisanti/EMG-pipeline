import numpy as np
from scipy.signal import butter, iirnotch



def bandpass_filter(sampling_rate: int, lowcut: float, highcut: float, order: int):
    """
    Crea los coeficientes de un filtro paso banda Butterworth.

    """

    nyquist = sampling_rate / 2.0

    low = lowcut / nyquist
    high = highcut / nyquist

    if low <= 0:
        raise ValueError("lowcut debe ser mayor que 0")

    if high >= 1:
        raise ValueError("highcut debe ser menor que la frecuencia de Nyquist")

    if low >= high:
        raise ValueError("lowcut debe ser menor que highcut")

    b, a = butter(
        N=order,
        Wn=[low, high],
        btype="bandpass"
    )

    return b, a


def notch_filter(sampling_rate: int, notch_freq: float, quality_factor: float):
    """
    Crea los coeficientes de un filtro notch.

    """

    nyquist = sampling_rate / 2.0
    normalized_freq = notch_freq / nyquist

    if normalized_freq <= 0 or normalized_freq >= 1:
        raise ValueError("notch_freq debe estar entre 0 y Nyquist")

    b, a = iirnotch(
        w0=normalized_freq,
        Q=quality_factor
    )

    return b, a