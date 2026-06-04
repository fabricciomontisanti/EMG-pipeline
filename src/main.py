from core import CircularBuffer, SignalData
from core.processing import SignalProcessor 
from core.classification.lda_classifier import LDAClassifier
from core.features import FeaturesExtractor
from offline import read_cometa_ascii, get_sample, get_depth, read_label_ranges_txt, get_label_for_window_realtime_fixed
from visualizer import plot_multichannel_signal, ControlVisualizer
from testing import show_window, show_features



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
    file_path: str = "data/raw/" + file_name + ".txt"
    max_samples: int = SAMPLING_RATE * BUFFER_DURATION_SECONDS
    mode: str = input ("Introduce el modo ('entrenamiento' o 'prediccion'): ")


    conf: config = config(c1= "FD", c2= "FP", c3= "ED", c4= "EC", c5= "FC", c6= "P")

    windows_file = open("results/windows.txt", "w")
    features_file = open("results/features.txt", "w")


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

    # Lectura de etiquetas
    label_ranges = read_label_ranges_txt("data/labels/" + file_name + "_labels.txt")

    # Creación del clasificador LDA
    lda: LDAClassifier = LDAClassifier()
    if mode == "entrenamiento":
        try:
            lda.load_training_memory("models/training_memory.joblib")
            print("Memoria de entrenamiento cargada.")
        except FileNotFoundError:
            print("No hay memoria previa. Se empieza desde cero.")

    if mode == "prediccion":
        lda.load_model("models/lda_model.joblib")

    # Recorrido muestra a muestra
    window_c: int = 0
    samples_since_last_window: int = 0

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

        #Actualizar cuántas muestras nuevas han llegado
        samples_since_last_window += processed_block.shape[0]

        #Extraer ventana cuando haya suficiente señal
        if len(buffer) >= WINDOW_SIZE and samples_since_last_window >= HOP_SIZE:
            signal_data = SignalData(samples=buffer.get_last_samples(WINDOW_SIZE), 
                                    sampling_rate=SAMPLING_RATE, 
                                    channel_names=[conf.c1, conf.c2, conf.c3, conf.c4, conf.c5, conf.c6])
            
            samples_since_last_window = 0
            features = feature_extractor.extract(signal_data)
            label = get_label_for_window_realtime_fixed(
                window_index=window_c,
                label_ranges=label_ranges,
                window_size_first=WINDOW_SIZE / SAMPLING_RATE,
                window_size_hop=HOP_SIZE / SAMPLING_RATE,
                hop_size=HOP_SIZE / SAMPLING_RATE
            )
            
            if mode == "entrenamiento":
                if label != "ignore":
                    lda.add_training_sample(features, label=label)
            elif mode == "prediccion":
                predicted_label = lda.predict(features)
                print(f"Ventana {window_c}: etiqueta real={label}, predicha={predicted_label}")
            

            #Mostrar ventana procesada
            show_window(signal_data.get_samples(), window_c, windows_file)
                

            #Mostrar características extraídas
            show_features(features, window_c, features_file)

            window_c += 1


    windows_file.close()
    features_file.close()

    
    #Entrenar modelo 
    if (mode == "entrenamiento"):
        lda.train_from_memory()

        lda.save_training_memory("models/training_memory.joblib")
        lda.save_model("models/lda_model.joblib")
    
    """
    #Visualizador de etiquetas
    for i in range(len(window_labels)):
        print(f"Ventana {i}: etiqueta={window_labels[i]}")
    """

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