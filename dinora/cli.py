import argparse

from dinora.uci import start_uci


def cli():
    parser = argparse.ArgumentParser(
        prog='dinora',
        description='Chess engine'
    )

    parser.parse_args()
    start_uci()
