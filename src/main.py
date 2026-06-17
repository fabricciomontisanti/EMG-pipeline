#Librerias
from core import CircularBuffer, SignalData
from core.processing import SignalProcessor 
from core.classification.lda_classifier import LDAClassifier
from core.features import FeaturesExtractor
from offline import read_cometa_ascii, get_sample, get_depth, read_label_ranges_txt, get_label_for_window_realtime_fixed
from visualizer import plot_multichannel_signal, ControlVisualizer
from testing import show_window, show_features, show_metrics
from metrics import load_results, calculate_metrics

from dataclasses import dataclass
import numpy as np
import threading as th 
np.set_printoptions(threshold=np.inf)


#METADATOS
SAMPLING_RATE: int = 2000
N_CHANNELS: int = 6

ACQUISITION_BLOCK_SIZE: int = 40    # 20 ms a 2000 Hz
BUFFER_DURATION_SECONDS: int = 2
WINDOW_SIZE: int = 400              # 200 ms a 2000 Hz
HOP_SIZE: int = 200                 # 100 ms a 2000 Hz

#VARIABLES GLOBALES

#Configuracion de los canales -> grupos musculares
@dataclass
class config:
    c1: str
    c2: str
    c3: str
    c4: str
    c5: str
    c6: str

conf: config = config(c1= "P", c2= "FC", c3= "FD", c4= "FP", c5= "ED", c6= "EC")
muscle_order= ["FD", "FP", "ED", "EC", "FC", "P"]



#Funciones auxiliares
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

def launch_visualizer(raw_signal, processed_signal, sampling_rate, channel_names, file_name):
    ControlVisualizer(
        raw_signal=raw_signal,
        processed_signal=processed_signal,
        sampling_rate=sampling_rate,
        channel_names=channel_names,
        file_name=file_name
    )


def offline():
    """
    Programa para procesamiento offline de un fichero EMG.
    Permite procesar múltiples ficheros en un bucle infinito.
    """
    while True:  # bucle infinito

        # Solicitar al usuario si desea limpiar la memoria de entrenamiento
        clear: str = input("¿Limpiar memoria? (s/n): ")
        # Clasificador LDA
        lda: LDAClassifier = LDAClassifier()

        # Limpieza de memoria si se desea
        if clear.lower() == 's':
            contrasena: str = input("Introduce la contraseña para limpiar la memoria: ")
            if contrasena == "hnp":
                lda.clear_training_memory()
                print("Memoria de entrenamiento limpiada.")
                continue  # vuelve al inicio del bucle
            else:
                print("Contraseña incorrecta.")
                continue

        # Configuración
        file_name: str = input("Introduce el nombre del fichero a cargar: ")
        file_path: str = "data/raw/" + file_name + ".txt"
        max_samples: int = SAMPLING_RATE * BUFFER_DURATION_SECONDS
        

        windows_file = open("results/windows.txt", "w")
        features_file = open("results/features.txt", "w")
        labels_file = open("results/labels.txt", "w")
        metrics_file = open("results/metrics.txt", "w")

        # Lectura del fichero
        data_frame = read_cometa_ascii(file_path)
        if data_frame is None:
            print("No se pudo cargar el fichero.")
            continue  # vuelve al inicio del bucle

        # Creación del buffer circular
        buffer: CircularBuffer = CircularBuffer(max_samples)

        # Visualizador
        raw_signal: list[np.ndarray] = []
        processed_signal: list[np.ndarray] = []

        # Procesador
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

        # Extractor de características
        feature_extractor: FeaturesExtractor = FeaturesExtractor(
            zc_threshold=5.0,
            ssc_threshold=5.0
        )

        # Lectura de etiquetas
        label_ranges = read_label_ranges_txt("data/labels/" + file_name + "_labels.txt")


        print(f"Fichero '{file_name}' cargado correctamente, {len(data_frame)} muestras.")

        # Selección de modo
        mode: str = input("Introduce el modo ('entrenamiento' o 'prediccion'): ")
        if mode == "entrenamiento":
            try:
                lda.load_training_memory("models/training_memory.joblib")
                print("Memoria de entrenamiento cargada.")
            except FileNotFoundError:
                print("No hay memoria previa. Se empieza desde cero.")
        elif mode == "prediccion":
            lda.load_model("models/lda_model.joblib")

        # Variables de ventana
        window_c: int = 0
        samples_since_last_window: int = 0

        # Bucle de procesamiento por bloques
        for start_index in range(0, len(data_frame), ACQUISITION_BLOCK_SIZE):
            raw_block: np.ndarray = get_block(data_frame, start_index, ACQUISITION_BLOCK_SIZE)
            if raw_block.shape[0] == 0:
                continue
            if raw_block.shape[1] != N_CHANNELS:
                raise ValueError(f"El bloque tiene {raw_block.shape[1]} canales, pero se esperaban {N_CHANNELS}")

            processed_block: np.ndarray = processor.process_block(raw_block)
            raw_signal.append(raw_block)
            processed_signal.append(processed_block)

            for sample in processed_block:
                buffer.add_sample(sample)

            samples_since_last_window += processed_block.shape[0]

            if len(buffer) >= WINDOW_SIZE and samples_since_last_window >= HOP_SIZE:
                # Ventana ordenada
                signal_data = SignalData(
                    samples=buffer.get_last_samples(WINDOW_SIZE),
                    sampling_rate=SAMPLING_RATE,
                    channel_names=[conf.c1, conf.c2, conf.c3, conf.c4, conf.c5, conf.c6]
                )
                window_samples = signal_data.get_samples()
                channel_names = signal_data.get_channel_names()
                indices = [channel_names.index(m) for m in muscle_order]
                window_samples_ordered = window_samples[:, indices]
                samples_since_last_window = 0

                # Extraer características
                features = feature_extractor.extract(SignalData(
                    samples=window_samples_ordered,
                    sampling_rate=SAMPLING_RATE,
                    channel_names=muscle_order
                ))

                # Etiqueta de la ventana
                label = get_label_for_window_realtime_fixed(
                    window_index=window_c,
                    label_ranges=label_ranges,
                    window_size_first=WINDOW_SIZE / SAMPLING_RATE,
                    window_size_hop=HOP_SIZE / SAMPLING_RATE,
                    hop_size=HOP_SIZE / SAMPLING_RATE
                )

                if mode == "entrenamiento" and label != "ignore":
                    lda.add_training_sample(features, label=label)
                elif mode == "prediccion":
                    predicted_label = lda.predict(features)
                    labels_file.write(f"Ventana {window_c}: etiqueta real={label}, predicha={predicted_label}\n")

                # Hilos para mostrar ventana y características
                window_copy = np.copy(window_samples_ordered)
                th.Thread(target=show_window, args=(window_copy, window_c, windows_file)).start()
                th.Thread(target=show_features, args=(features, window_c, features_file)).start()

                window_c += 1

        # Guardar métricas si es predicción
        if mode == "prediccion":
            labels_real_arr, labels_pred_arr = load_results("results/labels.txt")
            metrics = calculate_metrics(labels_real_arr, labels_pred_arr, ignore_label="ignore")
            th.Thread(target=show_metrics, args=(metrics, metrics_file)).start()

        windows_file.close()
        features_file.close()
        labels_file.close()
        metrics_file.close()

        # Entrenamiento final si es modo entrenamiento
        if mode == "entrenamiento":
            lda.train_from_memory()
            lda.save_training_memory("models/training_memory.joblib")
            lda.save_model("models/lda_model.joblib")

        # Visualización completa
        t3 = th.Thread(target=launch_visualizer, args=(raw_signal, processed_signal, SAMPLING_RATE, [conf.c1, conf.c2, conf.c3, conf.c4, conf.c5, conf.c6], file_name))
        t3.start()

        t3.join()  


#main
def main():
    """
    Programa principal
    """
    modo: str = input("Introduce el modo de operación ('offline' o 'online'): ")
    if modo == "offline":
        print("=== Programa de procesamiento offline de señales EMG ===")
        offline()



if __name__ == "__main__":
    main()