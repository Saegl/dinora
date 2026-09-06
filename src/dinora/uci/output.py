import sys
import threading

_stdout_lock = threading.Lock()


def send(s: str) -> None:
    with _stdout_lock:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()
