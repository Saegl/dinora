from dataclasses import dataclass


@dataclass
class UciGoParams:
    wtime: int | None = None
    btime: int | None = None
    winc: int | None = None
    binc: int | None = None
    nodes: int | None = None
    infinite: bool = False

    def is_time(self, turn: bool) -> tuple[int, int] | None:
        # Can return time only if both wtime and btime specified
        if not (self.wtime and self.btime):
            return None
        engine_time = self.wtime if turn else self.btime
        engine_inc = (self.winc if turn else self.binc) or 0
        return engine_time, engine_inc


def parse_go_params(tokens: list[str]) -> UciGoParams:
    params = UciGoParams()

    # Skip go
    assert tokens[0] == "go"
    i = 1

    def end() -> bool:
        return i >= len(tokens)

    def consume_tok() -> str | None:
        nonlocal i
        if end():
            return None
        tok = tokens[i]
        i += 1
        return tok

    while not end():
        currtok = consume_tok()

        # tokens with integer value
        if currtok in ["wtime", "btime", "winc", "binc", "nodes"]:
            val = consume_tok()
            setattr(params, currtok, int(val))  # type: ignore

        if currtok == "infinite":
            params.infinite = True

    return params
