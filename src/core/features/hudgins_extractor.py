import numpy as np

from core import SignalData

class FeaturesExtractor:
    """
    Extractor de Hudgins set características  para señales EMG.

    Recibe una ventana de señal EMG en formato SignalData:

        signal_data.samples.shape = (n_muestras, n_canales)

    y devuelve un vector de características:

        features.shape = (n_canales * n_caracteristicas,)
    """

    def __init__(
        self,
        zc_threshold: float= 0.01,
        ssc_threshold: float= 0.01
        ):
        self.zc_threshold = zc_threshold
        self.ssc_threshold = ssc_threshold

    def extract(self, signal_data: SignalData) -> np.ndarray:
        """
        Extrae características de todos los canales de una ventana EMG.
        """

        samples = signal_data.get_samples()

        if samples.ndim != 2:
            raise ValueError("La señal debe tener forma (n_muestras, n_canales)")

        features: list[float] = []

        n_channels = samples.shape[1]

        for channel_index in range(n_channels):
            channel_signal = samples[:, channel_index]

            mav = self.mean_absolute_value(channel_signal)
            rms = self.root_mean_square(channel_signal)
            wl = self.waveform_length(channel_signal)
            zc = self.zero_crossings(channel_signal)
            ssc = self.slope_sign_changes(channel_signal)

            features.extend([mav, rms, wl, zc, ssc])

        return np.array(features, dtype=float)

    def mean_absolute_value(self, signal: np.ndarray) -> float:
        """
        Calcula el valor absoluto medio (MAV) de una señal.
        """
        return float(np.mean(np.abs(signal)))

    def root_mean_square(self, signal: np.ndarray) -> float:
        """
        Calcula la raíz del valor medio cuadrático (RMS) de una señal.
        """
        return float(np.sqrt(np.mean(signal ** 2)))

    def waveform_length(self, signal: np.ndarray) -> float:
        """
        Calcula la longitud de la forma de onda (WL) de una señal.
        """
        return float(np.sum(np.abs(np.diff(signal))))

    def zero_crossings(self, signal: np.ndarray) -> int:
        """
        Calcula el número de cruces por cero (ZC) de una señal.
        """
        count = 0

        for i in range(1, len(signal)):
            sign_change = signal[i - 1] * signal[i] < 0
            amplitude_change = abs(signal[i] - signal[i - 1]) >= self.zc_threshold

            if sign_change and amplitude_change:
                count += 1

        return count

    def slope_sign_changes(self, signal: np.ndarray) -> int:
        """
        Calcula el número de cambios de signo de pendiente (SSC) de una señal.
        """
        count = 0

        for i in range(1, len(signal) - 1):
            previous_diff = signal[i] - signal[i - 1]
            next_diff = signal[i] - signal[i + 1]

            sign_change = previous_diff * next_diff > 0
            amplitude_change = (
                abs(previous_diff) >= self.ssc_threshold
                or abs(next_diff) >= self.ssc_threshold
            )

            if sign_change and amplitude_change:
                count += 1

        return count