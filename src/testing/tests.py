import numpy as np

def show_window(window: np.ndarray, contador: int, output_file):

    print(f"Ventana {contador}:", file=output_file)
    print(window, file=output_file)

def show_features(window: np.ndarray, contador: int, output_file):

    print(f"Ventana {contador}:", file=output_file)
    print("Caracteristicas:", file=output_file)
    print(window, file=output_file)

def show_metrics(metrics: dict, output_file):
    print("="*40, file=output_file)
    print("          MÉTRICAS DE CLASIFICACIÓN", file=output_file)
    print("="*40, file=output_file)
    
    # Accuracy global
    print(f"Accuracy global: {metrics['accuracy']:.4f}", file=output_file)
    print("-"*40, file=output_file)
    
    # Métricas por clase
    print("Métricas por clase:", file=output_file)
    for cls, m in metrics.items():
        if cls not in ["accuracy", "confusion_matrix"]:
            print(f"\nClase: {cls}", file=output_file)
            print(f"  Precision : {m['precision']:.4f}", file=output_file)
            print(f"  Recall    : {m['recall']:.4f}", file=output_file)
            print(f"  F1-score  : {m['f1_score']:.4f}", file=output_file)
    
    # Matriz de confusión
    print("\n" + "-"*40, file=output_file)
    print("Matriz de confusión:", file=output_file)
    print(metrics["confusion_matrix"], file=output_file)
    print("="*40 + "\n", file=output_file)