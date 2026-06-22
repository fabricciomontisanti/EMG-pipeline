import socket
import time
import numpy as np
import threading as th
import queue
import os

np.set_printoptions(threshold=np.inf)

from core import CircularBuffer
from testing import show_window

HOST = "127.0.0.1"
PORT = 5000


def connect():
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        print(f"Conectado a CometaBridge en {HOST}:{PORT}")
        return sock



def receive_blocks(
    buff: CircularBuffer,
    block_queue: queue.Queue,
    stop_event: th.Event,
    buffer_lock: th.Lock
):
    sock = connect()

    try:
        with sock:
            file = sock.makefile("r", encoding="utf-8")

            while not stop_event.is_set():
                header = file.readline()

                if not header:
                    print("Conexión cerrada por CometaBridge.")
                    break

                header = header.strip()

                if not header:
                    continue

                parts = header.split()

                if len(parts) != 4 or parts[0] != "BLOCK":
                    print("Cabecera inesperada:", header)
                    continue

                block_id = int(parts[1])
                expected_samples = int(parts[2])
                expected_channels = int(parts[3])

                rows = []

                while True:
                    line = file.readline()

                    if not line:
                        print("Conexión cerrada durante un bloque.")
                        return

                    line = line.strip()

                    if line == "END":
                        break

                    values = [float(x) for x in line.split(",")]
                    rows.append(values)

                block = np.array(rows, dtype=np.float32)

                if block.shape != (expected_samples, expected_channels):
                    print(
                        f"Advertencia: bloque {block_id} con shape={block.shape}, "
                        f"esperado=({expected_samples}, {expected_channels})"
                    )
                    continue

                # Guardamos en el buffer circular
                with buffer_lock:
                    for sample in block:
                        buff.add_sample(sample.tolist())

                # Mandamos el bloque completo al hilo principal
                block_queue.put((block_id, block.copy()))

    finally:
        stop_event.set()