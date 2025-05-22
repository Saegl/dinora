import sys
import time


def send(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.write("\n")
    sys.stdout.flush()


class UCILogger:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.start_time = time.time()
        self.prev_iter_time = time.time()
        self.frequency = 1  # in seconds

    def should_log(self) -> bool:
        if not self.enabled:
            return False

        now = time.time()
        delta_time = now - self.prev_iter_time
        return delta_time >= self.frequency

    def on_search_iter(self, nodes: int, depth: int, pv: str, cp: int) -> None:
        if not self.enabled:
            return

        now = time.time()
        self.prev_iter_time = now

        time_searched = int((now - self.start_time) * 1000)
        nps = int(nodes / (time_searched / 1000)) if time_searched > 0 else 0
        send(
            f"info depth {depth} time {time_searched} nodes {nodes} score cp {cp} nps {nps} pv {pv}"
        )

    def on_search_finish(self, nodes: int, depth: int, pv: str, cp: int) -> None:
        if not self.enabled:
            return

        now = time.time()

        time_searched = int((now - self.start_time) * 1000)
        nps = int(nodes / (time_searched / 1000)) if time_searched > 0 else 0
        send(
            f"info depth {depth} time {time_searched} nodes {nodes} score cp {cp} nps {nps} pv {pv}"
        )
