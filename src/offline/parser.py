from pathlib import Path
import pandas as pd


def read_cometa_ascii(file_path) -> pd.DataFrame | None:
    """
    Lee un fichero ASCII exportado por Cometa.

    El fichero esperado tiene:
    - primera columna: tiempo en segundos
      Frecuencia de muestreo: 2000 Hz (cada fila representa incremento 0.0005 segundos)
    - resto de columnas: canales EMG en uV
    """

    #Variables
    path: Path = Path(file_path) 
    clean_columns: list[str] = []

    # Comprobamos que el fichero existe
    if not path.exists():
        print(f"Error: El fichero '{file_path}' no existe.")
        return None

    # Leemos el fichero.
    data_frame: pd.DataFrame= pd.read_csv(path, sep="\t")

    # Limpiamos los nombres de las columnas.
    for column_name in data_frame.columns:
        clean_name: str= column_name.replace(":", "")   # Elimina los dos puntos
        clean_name = clean_name.strip()            # Elimina espacios en blanco al inicio y al final
        clean_columns.append(clean_name)           # Agrega el nombre limpio a la lista

    data_frame.columns = clean_columns

    return data_frame


def get_sample(data_frame, row_index)-> tuple[float, list[float]]:
    """
    Devuelve una muestra concreta del fichero.

    Una muestra está formada por:
    - tiempo
    - valores EMG de todos los canales
    """
    #Variables
    row: pd.Series = data_frame.iloc[row_index] #Obtener fila correspondiente
    time_value: float = row.iloc[0]             
    emg_values: list[float] = []

    for value in row.iloc[1:]:
        emg_values.append(float(value))

    return time_value, emg_values



def get_depth(data_frame)-> int:
    """
    Devuelve la profundidad del fichero, es decir, el número de filas.
    """
    return len(data_frame)