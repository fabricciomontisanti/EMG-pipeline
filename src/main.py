from core import CircularBuffer, SignalData
from offline import read_cometa_ascii, get_sample, get_depth
from core.processing import SignalProcessor 
from visualizer import plot_multichannel_signal, ControlVisualizer
from core.features import FeaturesExtractor


from dataclasses import dataclass

import numpy as np
np.set_printoptions(threshold=np.inf)



SAMPLING_RATE: int = 2000
N_CHANNELS: int = 6

ACQUISITION_BLOCK_SIZE: int = 40    # 20 ms a 2000 Hz
BUFFER_DURATION_SECONDS: int = 2
WINDOW_SIZE: int = 400              # 200 ms a 2000 Hz
HOP_SIZE: int = 200                 # 100 ms a 2000 Hz

#Configuracion de los canales -> grupos musculares
@dataclass
class config:
    c1: str
    c2: str
    c3: str
    c4: str
    c5: str
    c6: str

def show_window(window: np.ndarray, contador: int, output_file):

    print(f"Ventana {contador}:", file=output_file)
    print("Caracteristicas:", file=output_file)
    print(window, file=output_file)

def get_block(data_frame, start_index: int, block_size: int) -> np.ndarray:
    """
    Devuelve un bloque de muestras EMG a partir del DataFrame.

    El bloque tiene forma:

        block.shape = (n_muestras_bloque, n_canales)

    """

    block_samples: list[list[float]] = []

    end_index: int = min(start_index + block_size, len(data_frame))

    for row_index in range(start_index, end_index):
        time_value, emg_values = get_sample(data_frame, row_index)
        block_samples.append(emg_values)

    return np.array(block_samples, dtype=float)

def main():
    """
    Programa principal
    """

    # Configuración
    file_name: str = input("Introduce el nombre del fichero a cargar: ")
    file_path: str = "data/raw/" + file_name
    max_samples: int = SAMPLING_RATE * BUFFER_DURATION_SECONDS

    conf: config = config(c1= "FD", c2= "FP", c3= "ED", c4= "EC", c5= "FC", c6= "P")


    # Lectura del fichero
    data_frame = read_cometa_ascii(file_path)

    if data_frame is None:
        print("No se pudo cargar el fichero. Saliendo del programa.")
        return

    print("Fichero cargado correctamente")
    print("Número de muestras:", len(data_frame))
    print("Columnas detectadas:", list(data_frame.columns))

    # Creación del buffer circular
    buffer: CircularBuffer = CircularBuffer(max_samples)

    #Visualizador 
    raw_signal: list[np.ndarray] = []
    processed_signal: list[np.ndarray] = []

    # Creación del procesador streaming
    processor: SignalProcessor = SignalProcessor(
        sampling_rate=SAMPLING_RATE,
        n_channels=N_CHANNELS,
        apply_bandpass=True,
        apply_notch=True,
        bandpass_low=20.0,
        bandpass_high=450.0,
        bandpass_order=4,
        notch_freq=50.0,
        notch_quality_factor=30.0,
    )

    # Creación del extractor de características
    feature_extractor: FeaturesExtractor = FeaturesExtractor(
        zc_threshold=5.0,  # Umbral para cruces por cero
        ssc_threshold=5.0  # Umbral para cambios de pendiente
    )


    # Recorrido muestra a muestra
    window_c: int = 0
    first: bool = True
    samples_since_last_window: int = 0

    with open("output.txt", "w") as output_file:
        for start_index in range(0, len(data_frame), ACQUISITION_BLOCK_SIZE):
            #Obtener bloque crudo
            raw_block: np.ndarray = get_block(data_frame, start_index, ACQUISITION_BLOCK_SIZE)

            if raw_block.shape[0] == 0:
                continue

            if raw_block.shape[1] != N_CHANNELS:
                raise ValueError(
                    f"El bloque tiene {raw_block.shape[1]} canales, "
                    f"pero se esperaban {N_CHANNELS}"
                )

            #Procesar bloque
            processed_block: np.ndarray = processor.process_block(raw_block)

            #Visualizar
            raw_signal.append(raw_block)
            processed_signal.append(processed_block)

            #Insertar muestras procesadas en el buffer
            for sample in processed_block:
                buffer.add_sample(sample)

            # 4. Actualizar cuántas muestras nuevas han llegado
            samples_since_last_window += processed_block.shape[0]

            # 5. Extraer ventana cuando haya suficiente señal
            if (first and len(buffer) >= WINDOW_SIZE) or (not first and samples_since_last_window >= HOP_SIZE):
                if (first):
                    signal_data = SignalData(samples=buffer.get_last_samples(WINDOW_SIZE), 
                                            sampling_rate=SAMPLING_RATE, 
                                            channel_names=[conf.c1, conf.c2, conf.c3, conf.c4, conf.c5, conf.c6])
                    first = False
                    samples_since_last_window = 0
                    features = feature_extractor.extract(signal_data)

                elif (samples_since_last_window >= HOP_SIZE):
                    signal_data = SignalData(samples=buffer.get_last_samples(WINDOW_SIZE), 
                                            sampling_rate=SAMPLING_RATE, 
                                            channel_names=[conf.c1, conf.c2, conf.c3, conf.c4, conf.c5, conf.c6])
                    samples_since_last_window = 0
                    features = feature_extractor.extract(signal_data)

                #Mostrar ventana procesada
                #show_window(signal_data.get_samples(), window_c, output_file)
                window_c += 1

                #Mostrar características extraídas
                show_window(features, window_c, output_file)
                
            

    #Visualizar señal completa
    visualizer = ControlVisualizer(
        raw_signal=raw_signal,
        processed_signal=processed_signal,
        sampling_rate=SAMPLING_RATE,
        channel_names=[conf.c1, conf.c2, conf.c3, conf.c4, conf.c5, conf.c6],
        file_name=file_name
    )


if __name__ == "__main__":
    main()