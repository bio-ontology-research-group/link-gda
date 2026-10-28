#!/usr/bin/env python3
"""Convert an existing per-association prediction TSV to an identity-checked NPZ."""
import argparse
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from link_gda.score_io import convert_prediction_tsv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--candidates', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert_prediction_tsv(args.predictions, args.out, args.candidates)))
