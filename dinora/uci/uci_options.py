# TODO: I don't know how to properly annotate convert
from typing import Any


class Check:
    uci_type = "check"

    @staticmethod
    def convert(val: str) -> Any:
        pass


class Spin:
    uci_type = "spin"
    minvalue: int
    maxvalue: int

    @staticmethod
    def convert(val: str) -> Any:
        pass


class String:
    uci_type = "string"

    @staticmethod
    def convert(val: str) -> Any:
        pass


class FloatString:
    """Just a UCI string for GUI, converted to float in engine"""

    uci_type = "string"

    @staticmethod
    def convert(val: str) -> Any:
        return float(val)


UciOption = Check | Spin | String | FloatString
