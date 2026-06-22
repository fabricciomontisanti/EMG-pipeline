import numpy as np
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_multichannel_signal(
    signal: np.ndarray,
    sampling_rate: int,
    channel_names: list[str],
    title: str,
    output_path: str
):
    """
    Visualiza una señal EMG multicanal en el dominio temporal.

    Parámetros:
        signal:
            Matriz con forma (n_muestras, n_canales).

        sampling_rate:
            Frecuencia de muestreo en Hz.

        channel_names:
            Nombres de los canales.

        title:
            Título de la figura.

        output_path:
            Ruta para guardar la figura. Si es None, se muestra en pantalla.
    """

    if signal.ndim != 2:
        raise ValueError("signal debe tener forma (n_muestras, n_canales)")

    n_samples = signal.shape[0]
    n_channels = signal.shape[1]

    if len(channel_names) != n_channels:
        raise ValueError(
            "La longitud de channel_names debe coincidir con el número de canales"
        )


    time_axis = np.arange(n_samples) / sampling_rate

    max_value = np.max(np.abs(signal)) #maximo valor de la señal
    if max_value == 0:
        y_limit = 1 #evitar división por cero 
    else:
        y_limit = max_value * 1.1 #limite de y para mejor visualización

    fig, axes = plt.subplots(
        n_channels,
        1,
        sharex=True,
        sharey=True,
        figsize=(12, 2 * n_channels)
    )

    if n_channels == 1:
        axes = [axes]

    fig.suptitle(title)

    for channel_index in range(n_channels):
        axes[channel_index].plot(
            time_axis,
            signal[:, channel_index]
        )

        axes[channel_index].set_ylabel(channel_names[channel_index])
        axes[channel_index].set_ylim(-y_limit, y_limit)
        axes[channel_index].grid(True)

    axes[-1].set_xlabel("Tiempo (s)")

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        print(f"Figura guardada en: {output_path}")
    else:
        plt.show()

    plt.close()
