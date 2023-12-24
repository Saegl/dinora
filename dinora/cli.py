import dinora.elo_estimator.cli as elo_estimator_cli

import dinora.viz.cli as treeviz_cli
import dinora.uci.cli as uci_cli
import dinora.train.compact_dataset.make_cli as make_dataset_cli
import dinora.train.compact_dataset.upload_cli as upload_dataset_cli
import dinora.bench.cli as bench_cli
import dinora.onnx.cli as onnx_cli


def run_cli():
    parser = build_root_cli()
    args = parser.parse_args()

    if args.subcommand == "elo_estimator":
        elo_estimator_cli.run_cli(args)
    elif args.subcommand == "treeviz":
        treeviz_cli.run_cli(args)
    elif args.subcommand == "make_dataset":
        make_dataset_cli.run_cli(args)
    elif args.subcommand == "upload_dataset":
        upload_dataset_cli.run_cli(args)
    elif args.subcommand == "bench":
        bench_cli.run_cli(args)
    elif args.subcommand == "export_onnx":
        onnx_cli.run_cli(args)
    else:
        uci_cli.run_cli(args)


def build_root_cli():
    parser = uci_cli.build_parser()
    subparsers = parser.add_subparsers(title="Subcommands", dest="subcommand")

    elo_estimator_cli.build_parser(subparsers)
    treeviz_cli.build_parser(subparsers)
    make_dataset_cli.build_parser(subparsers)
    upload_dataset_cli.build_parser(subparsers)
    bench_cli.build_parser(subparsers)
    onnx_cli.build_parser(subparsers)

    return parser
