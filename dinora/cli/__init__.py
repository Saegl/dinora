import click


class UnimplementedCommand(Exception):
    pass


@click.group()
def cli() -> None:
    """Dinora is a chess engine"""


@cli.command()
@click.option(
    "--model",
    default="cached_dnn",
    help=(
        "Model used to evaluate position, predict future moves."
        "Available options: 'dnn', 'badgyal', 'handcrafted' "
        "and same models but cached (use 'cached_' prefix)"
    ),
)
@click.option("--override-go", default="", help="Override uci go command")
def uci(model: str, override_go: str) -> None:
    """Start UCI"""
    from dinora.cli.uci import start_uci

    start_uci(model, override_go)


@cli.command()
def healthcheck() -> None:
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
def download() -> None:
    """Download model weights"""
    raise UnimplementedCommand("Download is not yet implemented")


@cli.group()
def dev() -> None:
    """Commands for developers"""


@dev.command()
def metrics() -> None:
    """Generate engine metrics"""
    raise UnimplementedCommand("Metrics are not yet implemented")
