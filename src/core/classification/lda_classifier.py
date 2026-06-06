import os # Para manejo de archivos y directorios
import joblib # Para guardar y cargar modelos entrenados
import numpy as np # Para manejo de arrays y cálculos numéricos

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # Para el modelo LDA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Para evaluación del modelo


class LDAClassifier:
    """
    Clasificador LDA para señales EMG.

    Recibe vectores de características extraídos de ventanas EMG.

    X.shape = (n_ventanas, n_caracteristicas) Matriz de características
    y.shape = (n_ventanas,)                   Vector de etiquetas  
    """

    def __init__(self):
        self.model = LinearDiscriminantAnalysis()
        self.is_trained = False

        self.training_features: list[np.ndarray] = []
        self.training_labels: list[str] = []

        self.prediction_true_labels: list[str] = []
        self.prediction_outputs: list[str] = []

    def add_training_sample(self, features: np.ndarray, label: str):
        """
        Añade una ventana etiquetada a la memoria de entrenamiento.
        """

        features = np.asarray(features, dtype=float)

        if features.ndim != 1:
            raise ValueError("features debe ser un vector 1D")

        self.training_features.append(features)
        self.training_labels.append(label)

    def train_from_memory(self):
        """
        Entrena LDA usando todas las ventanas almacenadas.
        """

        if len(self.training_features) == 0:
            raise RuntimeError("No hay muestras de entrenamiento.")

        X = np.vstack(self.training_features)
        y = np.array(self.training_labels)

        self._validate_training_data(X, y)

        self.model.fit(X, y)
        self.is_trained = True

        print("Modelo LDA entrenado.")
        print("X shape:", X.shape)
        print("Clases:", self.model.classes_)

    def predict(self, features: np.ndarray) -> str:
        """
        Predice la clase de una única ventana.
        """

        if not self.is_trained:
            raise RuntimeError("El clasificador no está entrenado.")

        features = np.asarray(features, dtype=float)

        if features.ndim != 1:
            raise ValueError("features debe ser un vector 1D")

        features = features.reshape(1, -1)

        prediction = self.model.predict(features)

        return str(prediction[0])

    def add_prediction_result(self, true_label: str, predicted_label: str):
        """
        Guarda una predicción junto a su etiqueta real para evaluación posterior.
        """

        self.prediction_true_labels.append(true_label)
        self.prediction_outputs.append(predicted_label)

    def evaluate_prediction_history(self):
        """
        Evalúa las predicciones acumuladas durante una simulación online.
        """

        if len(self.prediction_true_labels) == 0:
            raise RuntimeError("No hay predicciones para evaluar.")

        y_true = np.array(self.prediction_true_labels)
        y_pred = np.array(self.prediction_outputs)

        accuracy = accuracy_score(y_true, y_pred)

        print("Accuracy:", accuracy)
        print()
        print("Classification report:")
        print(classification_report(y_true, y_pred))
        print()
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))

        return accuracy

    def save_model(self, model_path: str):
        """
        Guarda solo el modelo entrenado.
        """

        if not self.is_trained:
            raise RuntimeError("No se puede guardar un modelo no entrenado.")

        directory = os.path.dirname(model_path)

        if directory != "":
            os.makedirs(directory, exist_ok=True)

        joblib.dump(
            {
                "model": self.model,
                "is_trained": self.is_trained,
            },
            model_path
        )

    def load_model(self, model_path: str):
        """
        Carga un modelo entrenado.
        """

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No existe el modelo: {model_path}")

        data = joblib.load(model_path)

        self.model = data["model"]
        self.is_trained = data["is_trained"]

    def save_training_memory(self, memory_path: str):
        """
        Guarda las características y etiquetas acumuladas.
        """

        directory = os.path.dirname(memory_path)

        if directory != "":
            os.makedirs(directory, exist_ok=True)

        joblib.dump(
            {
                "training_features": self.training_features,
                "training_labels": self.training_labels,
            },
            memory_path
        )

    def load_training_memory(self, memory_path: str):
        """
        Carga características y etiquetas acumuladas anteriormente.
        """

        if not os.path.exists(memory_path):
            raise FileNotFoundError(f"No existe la memoria: {memory_path}")

        data = joblib.load(memory_path)

        self.training_features = data["training_features"]
        self.training_labels = data["training_labels"]

    def clear_prediction_history(self):
        """
        Limpia las predicciones acumuladas.
        """

        self.prediction_true_labels = []
        self.prediction_outputs = []

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray):
        """
        Comprueba que X e y tienen formato correcto.
        """

        if X.ndim != 2:
            raise ValueError("X debe tener forma (n_ventanas, n_caracteristicas)")

        if y.ndim != 1:
            raise ValueError("y debe tener forma (n_ventanas,)")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X e y deben tener el mismo número de ventanas"
            )

        unique_classes = np.unique(y)

        if len(unique_classes) < 2:
            raise ValueError(
                "LDA necesita al menos dos clases distintas para entrenar."
            )

    def clear_training_memory(self):
        """
        Limpia las muestras de entrenamiento acumuladas.
        """

        memory_file = "models/training_memory.joblib"
        model_file = "models/lda_model.joblib"

        if os.path.exists(memory_file):
            os.remove(memory_file)
        if os.path.exists(model_file):
            os.remove(model_file)