from .visualizer import plot_multichannel_signal

import numpy as np


class ControlVisualizer:
    def __init__(self, 
                raw_signal: np.ndarray,
                processed_signal: np.ndarray,
                sampling_rate: int, 
                channel_names: list[str], 
                file_name: str):
                
        self.raw_signal = raw_signal
        self.processed_signal = processed_signal
        self.sampling_rate = sampling_rate
        self.channel_names = channel_names
        self.file_name = file_name

        self.visualize()

    def visualize(self):
        raw_signal= np.vstack(self.raw_signal)
        processed_signal = np.vstack(self.processed_signal)

        plot_multichannel_signal(
            signal=raw_signal,
            sampling_rate=self.sampling_rate,
            channel_names=self.channel_names,
            title= self.file_name + " - Señal cruda",
            output_path="data/processed/"+self.file_name+"_raw_signal.png")

        plot_multichannel_signal(
            signal=processed_signal,
            sampling_rate=self.sampling_rate,
            channel_names=self.channel_names,
            title= self.file_name + " - Señal procesada",
            output_path="data/processed/"+self.file_name+"_processed_signal.png" )


