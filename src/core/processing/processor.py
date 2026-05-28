import numpy as np
from scipy.signal import lfilter, lfilter_zi

from .filters import bandpass_filter, notch_filter

class SignalProcessor: 
    """
    Preprocesador EMG para procesamiento por bloques.

    Recibe bloques pequeños de señal con forma:

        block.shape = (n_muestras_bloque, n_canales)

    y devuelve bloques procesados con la misma forma.

    """

    def __init__(
        self,
        sampling_rate: int,
        n_channels: int,
        apply_bandpass: bool,
        apply_notch: bool,
        bandpass_low: float,
        bandpass_high: float,
        bandpass_order: int,
        notch_freq: float,
        notch_quality_factor: float,
    ):
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels

        self.apply_bandpass = apply_bandpass
        self.apply_notch = apply_notch

        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.bandpass_order = bandpass_order

        self.notch_freq = notch_freq
        self.notch_quality_factor = notch_quality_factor

        self.bandpass_b = None
        self.bandpass_a = None
        self.bandpass_state = None

        self.notch_b = None
        self.notch_a = None
        self.notch_state = None

        self._initialize_filters()

    def _initialize_filters(self):
        """
        Metodo que crea los filtros y sus estados

        """
        if self.apply_bandpass:
            # Crear filtro paso banda
            self.bandpass_b, self.bandpass_a = bandpass_filter(
                sampling_rate=self.sampling_rate,
                lowcut=self.bandpass_low,
                highcut=self.bandpass_high,
                order=self.bandpass_order,
                )
            #Inicializar estado
            self.bandpass_state = self._create_initial_state(
                self.bandpass_b,
                self.bandpass_a
                )           

        if self.apply_notch:
            # Crear filtro notch
            self.notch_b, self.notch_a = notch_filter(
                sampling_rate=self.sampling_rate,
                notch_freq=self.notch_freq,
                quality_factor=self.notch_quality_factor,
            )
            #Inicializar estado
            self.notch_state = self._create_initial_state(
                self.notch_b,
                self.notch_a
            )


    def _create_initial_state(self, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Crea el estado inicial para un filtro a partir decoeficientes b y a.

        """
        zi = lfilter_zi(b, a) # Estado inicial para un canal
        state = np.tile(zi[:, np.newaxis], (1, self.n_channels)) # Repetir para todos los canales
        return state

    def process_block(self, block: np.ndarray) -> np.ndarray:
        """
        Procesa un bloque de muestras.

        """

        self._validate_block(block) # Validar forma del bloque

        processed = block.copy() # Copiar bloque para no modificar el original

        if self.apply_bandpass:
            processed, self.bandpass_state = lfilter(
                self.bandpass_b,
                self.bandpass_a,
                processed,
                axis=0, #se filtra a lo largo de las filas, es decir, a lo largo del tiempo
                zi=self.bandpass_state
            )

        if self.apply_notch:
            processed, self.notch_state = lfilter(
                self.notch_b,
                self.notch_a,
                processed,
                axis=0, #se filtra a lo largo de las filas, es decir, a lo largo del tiempo 
                zi=self.notch_state
            )

        return processed

    def reset(self):
        """
        Reinicia el estado de los filtros.

        """

        self._initialize_filters()

    def _validate_block(self, block: np.ndarray):
        """
        Comprueba que el bloque tiene el formato esperado.
        """

        if not isinstance(block, np.ndarray):
            raise TypeError("block debe ser un np.ndarray")

        if block.ndim != 2:
            raise ValueError(
                "block debe tener forma (n_muestras_bloque, n_canales)"
            )

        if block.shape[1] != self.n_channels:
            raise ValueError(
                f"El bloque tiene {block.shape[1]} canales, "
                f"pero se esperaban {self.n_channels}"
            )

        if block.shape[0] == 0:
            raise ValueError("El bloque no puede estar vacío")