from .visualizer_off import plot_multichannel_signal
from .visualizer_on import LiveMultichannelVisualizer


import numpy as np


class ControlVisualizer:
    def __init__(self, 
                raw_signal: np.ndarray,
                processed_signal: np.ndarray,
                sampling_rate: int, 
                channel_names: list[str], 
                file_name: str,
                mode: str):
                
        self.raw_signal = raw_signal
        self.processed_signal = processed_signal
        self.sampling_rate = sampling_rate
        self.channel_names = channel_names
        self.file_name = file_name
        self.mode = mode; 

        if mode=="offline":
            self.visualize_off()
        elif mode == "online":
            self.visualize_on()
        else: 
            raise ValueError("mode debe ser 'offline' u 'online'")
        


    def visualize_off(self):
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


    def visualize_on(self):
        self.live_visualizer = LiveMultichannelVisualizer(
            sampling_rate=self.sampling_rate,
            channel_names=self.channel_names,
            title=self.file_name + " - Señal procesada online",
            seconds_to_show=5.0,
            y_limit=500.0
        )

    def update_online(self):
        self.live_visualizer.update(self.processed_signal)

    def close(self):
        if self.live_visualizer is not None:
            self.live_visualizer.close()