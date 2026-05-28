# Módulo de procesamiento para el pipeline de EMG.
from .filters import bandpass_filter, notch_filter
from .processor import SignalProcessor