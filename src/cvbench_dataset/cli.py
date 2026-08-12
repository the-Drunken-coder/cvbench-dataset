from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import DatasetError
from .initializer import init_dataset
from .intake import DEFAULT_MAX_CONTRIBUTION_BYTES, import_contribution
from .manifest import build_release, verify_release
from .source_recipe import hydrate_source_recipe, validate_source_recipe
from .validator import validate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cvbench-dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create a new canonical draft dataset")
    init.add_argument("root", type=Path)
    init.add_argument("--id", dest="dataset_id", required=True)
    init.add_argument("--title", required=True)
    init.add_argument("--description", required=True)
    init.add_argument(
        "--class",
        dest="classes",
        action="append",
        required=True,
        metavar="ID=DESCRIPTION",
    )
    init.add_argument(
        "--data-role",
        choices=["training_only", "benchmark_candidate", "benchmark_truth"],
        default="training_only",
    )
    init.add_argument(
        "--annotation-scope",
        choices=["exhaustive_visible", "class_exhaustive", "sparse", "activity_bounded"],
        default="exhaustive_visible",
    )

    validate = subparsers.add_parser("validate", help="validate a canonical dataset root")
    validate.add_argument("root", type=Path)

    build = subparsers.add_parser(
        "build-release",
        help="fail-closed manifest validation and deterministic tar.gz construction",
    )
    build.add_argument("root", type=Path)
    build.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify-release", help="verify a release manifest and optional archive")
    verify.add_argument("root", type=Path)
    verify.add_argument("--archive", type=Path)

    intake = subparsers.add_parser(
        "import-contribution",
        help="safely import one Studio contribution into a draft dataset",
    )
    intake.add_argument("root", type=Path)
    intake.add_argument("studio_zip", type=Path)
    intake.add_argument(
        "--max-total-bytes",
        type=int,
        default=DEFAULT_MAX_CONTRIBUTION_BYTES,
    )

    recipe = subparsers.add_parser(
        "validate-source-recipe",
        help="validate a public source-referenced training dataset",
    )
    recipe.add_argument("root", type=Path)

    hydrate = subparsers.add_parser(
        "hydrate-source-recipe",
        help="copy hash-pinned local source videos into a canonical dataset",
    )
    hydrate.add_argument("root", type=Path)
    hydrate.add_argument("--source-dir", type=Path, required=True)
    hydrate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            result = init_dataset(
                args.root,
                dataset_id=args.dataset_id,
                title=args.title,
                description=args.description,
                classes=args.classes,
                data_role=args.data_role,
                annotation_scope=args.annotation_scope,
            )
        elif args.command == "validate":
            result = validate_dataset(args.root).to_dict()
        elif args.command == "build-release":
            result = build_release(args.root, args.output)
        elif args.command == "verify-release":
            result = verify_release(args.root, args.archive)
        elif args.command == "validate-source-recipe":
            result = validate_source_recipe(args.root).to_dict()
        elif args.command == "hydrate-source-recipe":
            result = hydrate_source_recipe(args.root, args.source_dir, args.output)
        else:
            result = import_contribution(
                args.root,
                args.studio_zip,
                max_total_bytes=args.max_total_bytes,
            )
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
