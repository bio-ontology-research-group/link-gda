"""Generate the prepared-input count table from the retained annotation pairs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


ROWS = [
    ('Gene--Phenotype', 'gene_phenotypes.csv', ('Gene', 'Phenotype'), r'MGI\_GenePheno.rpt'),
    ('Disease--Phenotype', 'disease_phenotypes.csv', ('Disease', 'Phenotype'), 'phenotype.hpoa'),
    ('Gene--Function', 'gene_functions.csv', ('Gene', 'Function'), r'goa\_human.gaf.gz'),
    ('Gene--Expression site', 'gene_site.csv', ('Gene', 'Tissue'), 'tpmss.tsv'),
    ('Gene--Disease (total)', 'gene_diseases.csv', ('Gene', 'Disease'), r'genes\_to\_disease.txt'),
]


def generate(data, output):
    records = {}
    lines = [r'\begin{table}[htbp]', r'  \centering',
             r'  \caption{Data sources and prepared input counts before graph-term coverage filtering.',
             r'    Gene--disease pairs require non-empty represented phenotype profiles for both entities.}',
             r'  \label{tab:datasets}', r'  \small', r'  \begin{tabular}{llr}',
             r'    \toprule', r'    Relation & Source file & \# pairs \\', r'    \midrule']
    for label, name, columns, provider in ROWS:
        path = data / name
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or ()) != set(columns):
                raise ValueError(f'Unexpected columns: {name}')
            pairs = [tuple(row[c] for c in columns) for row in reader]
        if not pairs or any(not all(pair) for pair in pairs):
            raise ValueError(f'Empty annotation or identifier: {name}')
        count = len(set(pairs))
        records[name] = dict(pairs=count, source_rows=len(pairs),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        source = r'\texttt{' + provider + '}'
        source = source + ' (E-GTEX-8)' if name == 'gene_site.csv' else source
        # Preserve the manuscript's existing table layout.
        lines.append(f'    {label:<25}& {source:<30} & ' + f'{count:,}'.replace(',', '{,}') + r' \\')
    lines.extend([r'    \bottomrule', r'  \end{tabular}', r'\end{table}'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n')
    output.with_suffix('.json').write_text(json.dumps(dict(scope='Distinct prepared annotation pairs; provider labels identify upstream sources, not newly downloaded raw snapshots', inputs=records), indent=2) + '\n')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    generate(args.data_dir, args.out)
