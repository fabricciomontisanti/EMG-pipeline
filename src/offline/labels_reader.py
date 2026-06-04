from pathlib import Path
from typing import List, Tuple

def read_label_ranges_txt(file_path: str) -> list[tuple[float, float, str]]:
    """
    Lee un fichero de etiquetas tipo TXT con rangos por línea.

    Formato esperado por línea:
        start end label
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero de etiquetas: {file_path}")

    ranges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip() # Elimina espacios en blanco al inicio y al final
            if not line or line.startswith("#"):
                continue  # ignorar líneas vacías o comentarios
            parts = line.split(",")  # dividir por comas
            if len(parts) != 3:
                raise ValueError(f"Línea mal formada: {line}")
            start, end, label = float(parts[0]), float(parts[1]), parts[2]
            ranges.append((start, end, label))

    return ranges


def get_label_for_window_realtime_fixed(
    window_index: int,
    label_ranges: List[Tuple[float, float, str]],
    window_size_first: float,
    window_size_hop: float,
    hop_size: float
    ) -> str:
    """
    Devuelve la etiqueta correspondiente a una ventana concreta en tiempo real.
    Si el centro de la ventana queda fuera de los rangos, devuelve 'ignore'.
    """
    if window_index == 0:
        center_time = window_size_first / 2
    else:
        center_time = window_index * hop_size + window_size_hop / 2

    for start, end, label in label_ranges:
        if start <= center_time < end:
            return label

    return "ignore"