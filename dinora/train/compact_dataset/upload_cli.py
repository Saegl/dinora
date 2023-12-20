import pathlib


def build_parser(subparsers):
    parser = subparsers.add_parser(
        name="upload_dataset", help="Tool to upload dataset to wandb"
    )
    parser.add_argument(
        "dataset_dir",
        help="Dataset directory",
        type=pathlib.Path,
    )
    parser.add_argument(
        "wandb_label",
        help="Wandb dataset label",
        type=str,
    )


def run_cli(args):
    from dinora.train.compact_dataset.upload import upload_dataset

    upload_dataset(args.dataset_dir, args.wandb_label)
