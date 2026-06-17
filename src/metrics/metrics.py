# metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_results(file_path: str):
    """
    Lee un fichero de resultados con formato:
    Ventana N: etiqueta real=X, predicha=Y
    Devuelve listas de etiquetas reales y predichas.
    """
    labels_real = []
    labels_pred = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split("etiqueta real=")[1].split(", predicha=")
                labels_real.append(parts[0].strip())
                labels_pred.append(parts[1].strip())
            except IndexError:
                continue

    return np.array(labels_real), np.array(labels_pred)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def calculate_metrics(labels_real: np.ndarray, labels_pred: np.ndarray, ignore_label="ignore"):
    """
    Calcula métricas por clase individual: accuracy, precision, recall, F1-score.
    Ignora las ventanas con etiqueta `ignore`.
    """
    mask = labels_real != ignore_label
    y_true = labels_real[mask]
    y_pred = labels_pred[mask]

    classes = np.unique(y_true)
    
    metrics = {}
    # Accuracy global
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Métricas por clase
    precision = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    
    # Guardar resultados en diccionario por clase
    for i, cls in enumerate(classes):
        metrics[cls] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1_score": f1[i]
        }
    
    # Matriz de confusión
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=classes)
    
    return metrics
