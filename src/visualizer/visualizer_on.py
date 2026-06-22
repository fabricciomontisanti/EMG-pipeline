import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class LiveMultichannelVisualizer:
    """
    Visualizador online para señales multicanal.
    Mantiene una ventana abierta y actualiza las curvas en directo.
    """

    def __init__(
        self,
        sampling_rate: int,
        channel_names: list[str],
        title: str,
        seconds_to_show: float = 5.0,
        y_limit: int = 500.0
    ):
        self.sampling_rate = sampling_rate
        self.channel_names = channel_names
        self.title = title
        self.seconds_to_show = seconds_to_show
        self.y_limit= y_limit

        self.n_channels = len(channel_names)

        plt.ion()

        self.fig, self.axes = plt.subplots(
            self.n_channels,
            1,
            sharex=True,
            figsize=(12, 2 * self.n_channels)
        )

        if self.n_channels == 1:
            self.axes = [self.axes]

        self.lines = []

        for ax, channel_name in zip(self.axes, self.channel_names):
            line, = ax.plot([], [])
            ax.set_ylabel(channel_name)
            ax.set_ylim(-self.y_limit, self.y_limit)
            ax.grid(True)
            self.lines.append(line)

        self.axes[-1].set_xlabel("Tiempo (s)")
        self.fig.suptitle(self.title)

        plt.tight_layout()
        plt.show(block=False)

    def update(self, signal_blocks: list[np.ndarray]):
        if not signal_blocks:
            return

        signal = np.vstack(signal_blocks)

        if signal.ndim != 2:
            raise ValueError("signal debe tener forma (n_muestras, n_canales)")

        if signal.shape[1] != self.n_channels:
            raise ValueError(
                f"La señal tiene {signal.shape[1]} canales, "
                f"pero se esperaban {self.n_channels}"
            )

        samples_to_show = int(self.seconds_to_show * self.sampling_rate)

        if signal.shape[0] > samples_to_show:
            signal = signal[-samples_to_show:]

        time_axis = np.arange(signal.shape[0]) / self.sampling_rate

        for channel_index, line in enumerate(self.lines):
            channel_signal = signal[:, channel_index]

            line.set_data(time_axis, channel_signal)

            self.axes[channel_index].set_xlim(
                time_axis[0],
                time_axis[-1] if len(time_axis) > 1 else 1
            )

            ymin = np.min(channel_signal)
            ymax = np.max(channel_signal)

            if ymin == ymax:
                ymin -= 1
                ymax += 1

            margin = 0.1 * (ymax - ymin)
            self.axes[channel_index].set_ylim(-self.y_limit, self.y_limit)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.close(self.fig)