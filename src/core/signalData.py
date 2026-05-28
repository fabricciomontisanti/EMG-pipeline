from dataclasses import dataclass
import numpy as np

"""
    Representación interna común de una señal EMG.

    La señal se almacena como una matriz NumPy con forma:

        samples = [n_muestras x n_canales]
"""

@dataclass
class SignalData:
    samples: np.ndarray 
    sampling_rate: int
    channel_names: list[str] 

    
    def get_samples(self) -> np.ndarray:
        """
        Devuelve la matriz de muestras EMG.
        """
        return self.samples
    
    def get_sampling_rate(self) -> int:
        """
        Devuelve la frecuencia de muestreo.
        """
        return self.sampling_rate

    def get_channel_names(self) -> list[str]:
        """
        Devuelve los nombres de los canales.
        """
        return self.channel_names



    