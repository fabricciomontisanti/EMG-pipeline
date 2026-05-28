import numpy as np

def show_window(window: np.ndarray, contador: int, output_file):

    print(f"Ventana {contador}:", file=output_file)
    print(window, file=output_file)

def show_features(window: np.ndarray, contador: int, output_file):

    print(f"Ventana {contador}:", file=output_file)
    print("Caracteristicas:", file=output_file)
    print(window, file=output_file)