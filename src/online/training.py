import random
import time
import os

class TrainingOnline:
    def __init__(
        self,
        labels: list[str],
        min_duration: float = 3.0,
        max_duration: float = 10.0,
        transition_ignore: float = 0.3
    ):
        self.labels = labels
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.transition_ignore = transition_ignore

        self.current_label: str | None = None
        self.segment_start_time: float = 0.0
        self.segment_end_time: float = 0.0

        self._choose_next_label()

    def _choose_next_label(self):
        previous_label = self.current_label

        possible_labels = [
            label for label in self.labels
            if label != previous_label
        ]

        self.current_label = random.choice(possible_labels)

        now = time.monotonic()
        duration = random.uniform(self.min_duration, self.max_duration)

        self.segment_start_time = now
        self.segment_end_time = now + duration

        self._show_label(duration)

    def _show_label(self, duration: float):
        os.system("cls" if os.name == "nt" else "clear")

        print("=" * 50)
        print("          ENTRENAMIENTO ONLINE")
        print("=" * 50)
        print()
        print(f"GESTO ACTUAL: {self.current_label.upper()}")
        print()
        print(f"Duración aproximada: {duration:.2f} s")
        print()
        print("Realiza el gesto indicado hasta que cambie la etiqueta.")
        print("Pulsa Ctrl+C para finalizar el entrenamiento.")
        print("=" * 50)

    def update(self):
        now = time.monotonic()

        if now >= self.segment_end_time:
            self._choose_next_label()

    def get_training_label(self) -> str:
        self.update()

        now = time.monotonic()
        elapsed = now - self.segment_start_time

        if elapsed < self.transition_ignore:
            return "ignore"

        return self.current_label