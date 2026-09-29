from pathlib import Path

from graph_statistics import assemble


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_exact_stage_rows_and_candidate_scoped_fallback(tmp_path):
    data = Path(tmp_path)
    write(data / "upheno_edges.tsv", "P\tr\tQ\nP\tr\tQ\n")
    write(data / "go_edges.tsv", "F\tsub\tG\n")
    disease_rows = "".join(f"D{i},P\n" for i in range(1, 11))
    write(data / "disease_phenotypes.csv", "Disease,Phenotype\n" + disease_rows + "DT,P\n")
    write(data / "gene_functions.csv", "Gene,Function\nGF,F\n")
    write(data / "gene_phenotypes.csv", "Gene,Phenotype\nGF,P\nGFB,P\nGFB,Q\nGX,P\n")
    write(data / "gene_diseases.csv", "Gene,Disease\nGF,D1\nGFB,D2\n")
    train_rows = "".join(f"GF\tD{i}\n" for i in range(1, 11)) + "GFB\tD2\n"
    write(data / "folds/fold_0/train.csv", "Gene\tDisease\n" + train_rows)
    write(data / "folds/fold_0/test.csv", "Gene\tDisease\nGFB\tDT\nGX\tDT\n")
    result = assemble(data, 0, "f")
    assert result["stages"]["ontology_projection"]["loaded_triple_rows"] == 3
    assert result["stages"]["ontology_projection"]["unique_triples"] == 2
    assert result["annotation_rows_added"]["has_function"] == 1
    assert result["annotation_rows_added"]["has_phenotype"] == 2
    assert result["phenotype_fallback"] == {
        "eligible_genes_in_all_annotations": 2,
        "candidate_genes_affected": 1,
        "candidate_genes_denominator": 2,
        "true_test_genes_affected": 1,
        "true_test_genes_denominator": 2,
        "test_pairs_affected": 1,
        "test_pairs_denominator": 2,
    }
    final = result["stages"]["causes_phenotype"]
    assert final["loaded_training_instances_with_inverses"] == 2 * final["loaded_triple_rows"]
    assert final["unique_training_instances_with_inverses"] == 2 * final["unique_triples"]
