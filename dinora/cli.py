import click


class UnimplementedCommand(Exception):
    pass


@click.group()
def cli():
    """Dinora is a chess engine"""


@cli.command()
def uci():
    """Start UCI"""
    from dinora.uci import start_uci

    start_uci()


@cli.command()
def healthcheck():
    """Check if Dinora is installed properly"""
    raise UnimplementedCommand("Healthcheck is not yet implemented")
    # TODO check tf gpu is available
    # TODO check that keras model is reachable
    # click.secho("OK: Everything is working", fg="green")


@cli.command()
def download():
    """Download model weights"""
    raise UnimplementedCommand("Download is not yet implemented")


@cli.group()
def dev():
    """Commands for developers"""


@dev.command()
def metrics():
    """Generate engine metrics"""
    raise UnimplementedCommand("Metrics are not yet implemented")


if __name__ == "__main__":
    cli()
