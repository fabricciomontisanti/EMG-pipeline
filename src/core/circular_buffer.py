from collections import deque

import numpy as np


class CircularBuffer:
    """
    Buffer circular para almacenar las últimas muestras EMG recibidas.
    """
    max_samples: int
    buffer: deque

    def __init__(b, max_samples: int):
        """
        Constructor del buffer.
        """

        # Variables
        b.max_samples= max_samples
        b.buffer= deque(maxlen=max_samples)

    def add_sample(b, sample: list[float]):
        """
        Añade una muestra EMG al buffer.
        """

        b.buffer.append(sample)


    def clear(b):
        """
        Vacía el buffer.
        """

        b.buffer.clear()


    def get_last_samples(self, number_of_samples: int) -> np.ndarray:
        data: np.ndarray = np.array(self.buffer)

        if number_of_samples > len(data):
            raise ValueError("No hay suficientes muestras en el buffer")

        return data[-number_of_samples:]

    def __len__(b) -> int:
        """
        Permite usar len(buffer).
        """

        return len(b.buffer)