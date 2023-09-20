import dinora.elo_estimator.cli as elo_estimator_cli
import dinora.viz.cli as treeviz_cli
import dinora.uci.cli as uci_cli


def run_cli():
    parser = build_root_cli()
    args = parser.parse_args()

    if args.subcommand == "elo_estimator":
        elo_estimator_cli.run_cli(args)
    elif args.subcommand == "treeviz":
        treeviz_cli.run_cli(args)
    else:
        uci_cli.run_cli(args)


def build_root_cli():
    parser = uci_cli.build_parser()
    subparsers = parser.add_subparsers(title="Subcommands", dest="subcommand")

    elo_estimator_cli.build_parser(subparsers)
    treeviz_cli.build_parser(subparsers)

    return parser
