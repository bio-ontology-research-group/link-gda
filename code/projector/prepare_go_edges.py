"""Prepare a validated GO edge cache for LinkGDA training."""

import argparse
import logging
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from link_gda.go_projection import MODES, ensure_go_edges


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--go-projection-mode", choices=MODES, default="upheno-first")
    parser.add_argument("--rebuild", action="store_true",
                        help="Explicitly replace the selected cache and its metadata")
    parser.add_argument("--memory", default="10g", help="JVM maximum heap (default: 10g)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import mowl
    mowl.init_jvm(args.memory)
    from mowl.datasets import PathDataset
    from mowl.projection import OWL2VecStarProjector
    path = ensure_go_edges(
        args.data_dir,
        args.go_projection_mode,
        lambda: OWL2VecStarProjector(bidirectional_taxonomy=True),
        PathDataset,
        rebuild=args.rebuild,
    )
    print(f"GO projection ready: mode={args.go_projection_mode} path={path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
