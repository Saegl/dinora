class Check:
    uci_type = "check"

    @staticmethod
    def convert(val: str):
        pass


class Spin:
    uci_type = "spin"
    minvalue: int
    maxvalue: int

    @staticmethod
    def convert(val: str):
        pass


class String:
    uci_type = "string"

    @staticmethod
    def convert(val: str):
        pass


class FloatString:
    """Just a UCI string for GUI, converted to float in engine"""

    uci_type = "string"

    @staticmethod
    def convert(val: str):
        return float(val)


UciOptions = Check | Spin | String | FloatString
