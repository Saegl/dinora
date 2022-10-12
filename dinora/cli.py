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
    import tensorflow as tf

    ok = True
    click.secho("Checking gpu...")
    if tf.config.list_physical_devices("GPU"):
        click.secho("GPU: Found", fg="green")
    else:
        ok = False
        click.secho("GPU: Not found", fg="red")

    # TODO check that keras model is reachable

    if ok:
        click.secho("OK: Everything is working", fg="green")
    else:
        click.secho("NOT OK: errors above", fg="red")


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
